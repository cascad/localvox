//! Per-connection WebSocket handling: config parse, session loop, disconnect.

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::path::Path;
use tokio_util::either::Either;
use tokio_rustls::server::TlsStream;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc;
use tracing::info;

use crate::audio_writer;
use crate::config;
use crate::dispatcher;
use crate::processor;
use crate::session;
use crate::session_registry;

type ServerStream = Either<
    tokio::net::TcpStream,
    TlsStream<tokio::net::TcpStream>,
>;

pub type WsSender = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<ServerStream>,
    tokio_tungstenite::tungstenite::Message,
>;
pub type WsReceiver =
    futures_util::stream::SplitStream<tokio_tungstenite::WebSocketStream<ServerStream>>;

fn scan_session_lag(
    session_dir: &Path,
    transcribed_end_0: f64,
    transcribed_end_1: f64,
) -> (f64, f64, f64, f64) {
    let mut max_end_0 = 0.0f64;
    let mut max_end_1 = 0.0f64;
    let mut dir_size_bytes: u64 = 0;

    let Ok(entries) = std::fs::read_dir(session_dir) else {
        return (0.0, 0.0, 0.0, 0.0);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".meta.json") {
            if let Ok(data) = std::fs::read(&path) {
                if let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&data) {
                    let sid = meta.get("source_id").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
                    let end_sec = meta.get("end_time_sec").and_then(|v| v.as_f64());
                    if let Some(sec) = end_sec {
                        if sid == 0 {
                            max_end_0 = max_end_0.max(sec);
                        } else if sid == 1 {
                            max_end_1 = max_end_1.max(sec);
                        }
                    }
                }
            }
        }
        if let Ok(m) = entry.metadata() {
            if m.is_file() && name.ends_with(".wav") {
                dir_size_bytes += m.len();
            }
        }
    }
    let lag_0 = (max_end_0 - transcribed_end_0).max(0.0);
    let lag_1 = (max_end_1 - transcribed_end_1).max(0.0);
    let lag_max = lag_0.max(lag_1);
    let dir_size_mb = dir_size_bytes as f64 / (1024.0 * 1024.0);
    (lag_0, lag_1, lag_max, dir_size_mb)
}

fn setup_client_bridge() -> (
    crossbeam_channel::Sender<processor::ClientMessage>,
    mpsc::Receiver<processor::ClientMessage>,
) {
    let (cb_tx, cb_rx) = crossbeam_channel::unbounded::<processor::ClientMessage>();
    let (tx_ws, rx_ws) = mpsc::channel::<processor::ClientMessage>(64);
    thread::spawn(move || {
        while let Ok(msg) = cb_rx.recv() {
            if tx_ws.blocking_send(msg).is_err() {
                break;
            }
        }
    });
    (cb_tx, rx_ws)
}

fn spawn_ws_forwarder(
    mut rx: mpsc::Receiver<processor::ClientMessage>,
    mut ws: WsSender,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let json = msg_to_json(&msg);
            if ws
                .send(tokio_tungstenite::tungstenite::Message::Text(json))
                .await
                .is_err()
            {
                break;
            }
        }
    })
}

async fn replay_completed_session(
    mut ws_sender: WsSender,
    session: &session_registry::Session,
    last_seq: u64,
) -> Result<()> {
    let (cb_tx, rx_ws) = setup_client_bridge();
    let _ = ws_sender
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({"type": "session", "state": "resuming", "session_id": &session.id})
                .to_string(),
        ))
        .await;
    session
        .output_sink
        .replay_to_client_no_register(cb_tx, last_seq);
    let send_task = spawn_ws_forwarder(rx_ws, ws_sender);
    let _ = send_task.await;
    Ok(())
}

fn spawn_poll_lag(
    session_dir: std::path::PathBuf,
    output_sink: Arc<processor::OutputSink>,
    processor: Arc<dispatcher::FileProcessor>,
    transcribed_end: processor::TranscribedEnd,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        loop {
            interval.tick().await;
            let te0 = transcribed_end.get(0);
            let te1 = transcribed_end.get(1);
            let dir = session_dir.clone();
            let (lag_0, lag_1, lag_max, size_mb) =
                tokio::task::spawn_blocking(move || scan_session_lag(&dir, te0, te1))
                    .await
                    .unwrap_or((0.0, 0.0, 0.0, 0.0));
            let status = serde_json::json!({
                "type": "status",
                "lag_sec": (lag_max * 10.0).round() / 10.0,
                "lag_sec_0": (lag_0 * 10.0).round() / 10.0,
                "lag_sec_1": (lag_1 * 10.0).round() / 10.0,
                "audio_dir_size_mb": (size_mb * 100.0).round() / 100.0,
                "llm_queue": processor.llm_queue_len(),
                "task_queue_size": processor.asr_task_queue_len(),
            });
            output_sink.send_status(status);
        }
    })
}

fn handle_text_msg(
    text: &str,
    writer: &mut audio_writer::AudioWriter,
    writer_flushed: &mut bool,
    session: &session_registry::Session,
    processor: &dispatcher::FileProcessor,
    output_sink: &processor::OutputSink,
    registry: &session_registry::SessionRegistry,
) -> Result<()> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        let t = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
        if t == "recording" {
            if let Some(enabled) = v.get("enabled").and_then(|x| x.as_bool()) {
                writer.set_recording(enabled);
            }
        } else if t == "config" {
            if let Some(sc) = v.get("source_count").and_then(|x| x.as_u64()) {
                writer.set_source_count((sc as u8).clamp(1, 2));
            }
        } else if (t == "end_of_stream" || t == "end_session") && !*writer_flushed {
            *writer_flushed = true;
            session.end_of_stream.store(true, Ordering::Relaxed);
            for (path, src) in writer.close()? {
                info!(
                    "Segment ready src{}: {} -> queue (final)",
                    src,
                    path.display()
                );
                processor.enqueue(path, src);
            }
            processor.wait_until_empty(3600.0);
            std::thread::sleep(std::time::Duration::from_millis(500));
            output_sink.send(&processor::ClientMessage::Done);
            session.processing_complete.store(true, Ordering::Relaxed);
            if t == "end_session" {
                session::mark_session_done(&session.dir);
                if let Some(ref cid) = session.client_id {
                    registry.remove_client_session(cid);
                }
                registry.remove(&session.id);
                info!(
                    "Session ended by client, marked done and removed: {}",
                    session.id
                );
            }
        }
    }
    Ok(())
}

async fn run_session_loop(
    ws_receiver: &mut WsReceiver,
    writer: &mut audio_writer::AudioWriter,
    session: &session_registry::Session,
    processor: &dispatcher::FileProcessor,
    output_sink: &processor::OutputSink,
    registry: &session_registry::SessionRegistry,
    pending_first_msg: Option<&str>,
) -> Result<bool> {
    let mut writer_flushed = false;
    if let Some(text) = pending_first_msg {
        handle_text_msg(
            text,
            writer,
            &mut writer_flushed,
            session,
            processor,
            output_sink,
            registry,
        )?;
    }
    while let Some(msg) = ws_receiver.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(_) => break,
        };
        match msg {
            tokio_tungstenite::tungstenite::Message::Binary(data) => {
                if data.len() < 2 {
                    continue;
                }
                let source_id = data[0];
                if source_id > 1 {
                    continue;
                }
                let pcm = &data[1..];
                if let Ok(completed) = writer.feed(source_id, pcm) {
                    for (path, src) in completed {
                        info!("Segment ready src{}: {} -> queue", src, path.display());
                        processor.enqueue(path, src);
                    }
                }
            }
            tokio_tungstenite::tungstenite::Message::Text(text) => {
                handle_text_msg(
                    &text,
                    writer,
                    &mut writer_flushed,
                    session,
                    processor,
                    output_sink,
                    registry,
                )?;
            }
            _ => {}
        }
    }
    Ok(writer_flushed)
}

fn handle_disconnect(
    session: &Arc<session_registry::Session>,
    processor: &Arc<dispatcher::FileProcessor>,
    output_sink: &Arc<processor::OutputSink>,
    writer: &mut audio_writer::AudioWriter,
    writer_flushed: bool,
    settings: &config::Settings,
    registry: &Arc<session_registry::SessionRegistry>,
) {
    if writer_flushed {
        return;
    }
    if let Ok(segments) = writer.close() {
        for (path, src) in segments {
            processor.enqueue(path, src);
        }
    }
    if session.mode == session_registry::SessionMode::Live {
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        if let Ok(mut guard) = session.timeout_cancel_tx.lock() {
            *guard = Some(cancel_tx);
        }
        let p = processor.clone();
        let o = output_sink.clone();
        let s = session.clone();
        let reg = Arc::clone(registry);
        let timeout_sec = settings.live_reconnect_timeout_sec;
        tokio::spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(timeout_sec)) => {
                    s.end_of_stream.store(true, Ordering::Relaxed);
                    let s2 = s.clone();
                    tokio::task::spawn_blocking(move || {
                        p.wait_until_empty(3600.0);
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        o.send(&processor::ClientMessage::Done);
                        s2.processing_complete.store(true, Ordering::Relaxed);
                        session::mark_session_done(&s2.dir);
                        if let Some(ref cid) = s2.client_id {
                            reg.remove_client_session(cid);
                        }
                        tracing::info!("Live session timed out, marked done: {}", s2.id);
                    }).await.ok();
                }
                _ = cancel_rx => {}
            }
        });
    } else {
        session.end_of_stream.store(true, Ordering::Relaxed);
        let p = processor.clone();
        let o = output_sink.clone();
        let s = session.clone();
        let reg = Arc::clone(registry);
        thread::spawn(move || {
            p.wait_until_empty(3600.0);
            std::thread::sleep(std::time::Duration::from_millis(500));
            o.send(&processor::ClientMessage::Done);
            s.processing_complete.store(true, Ordering::Relaxed);
            session::mark_session_done(&s.dir);
            if let Some(ref cid) = s.client_id {
                reg.remove_client_session(cid);
            }
        });
    }
}

fn msg_to_json(msg: &processor::ClientMessage) -> String {
    match msg {
        processor::ClientMessage::Transcript {
            text,
            source,
            start_sec,
            end_sec,
            seg_id,
            variants,
            seq,
        } => {
            let mut j = serde_json::json!({
                "type": "transcript",
                "text": text,
                "source": source,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "seg_id": seg_id,
                "seq": seq
            });
            if let Some(ref v) = variants {
                j["variants"] = serde_json::to_value(v).unwrap_or_default();
            }
            j.to_string()
        }
        processor::ClientMessage::Status(v) => v.to_string(),
        processor::ClientMessage::Done => serde_json::json!({"type": "done"}).to_string(),
    }
}

/// Main entry point for handling a client connection.
pub async fn handle_client(
    stream: ServerStream,
    settings: config::Settings,
    audio_dir: std::path::PathBuf,
    asr_dispatcher: Arc<dispatcher::AsrDispatcher>,
    registry: Arc<session_registry::SessionRegistry>,
    _conn_id: u32,
) -> Result<()> {
    let settings_ref = &settings;
    let ws_stream = tokio_tungstenite::accept_hdr_async(
        stream,
        |req: &http::Request<()>, mut res: http::Response<()>| {
        if settings_ref.auth_enabled() {
            let key = req
                .headers()
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer ").map(|s| s.trim()))
                .filter(|s| !s.is_empty());
            if let Some(k) = key {
                if settings_ref.validate_api_key(k).is_some() {
                    return Ok(res);
                }
            }
            tracing::warn!("Auth rejected: missing or invalid API key");
            *res.status_mut() = http::StatusCode::UNAUTHORIZED;
            Err(res.map(|()| Some("Unauthorized".to_string())))
        } else {
            Ok(res)
        }
        },
    )
    .await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // --- Parse first message (config or data) ---

    let first_msg = ws_receiver.next().await;
    let first_text = match first_msg {
        Some(Ok(tokio_tungstenite::tungstenite::Message::Text(t))) => Some(t),
        Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) => return Ok(()),
        Some(Err(_)) => return Ok(()),
        None => return Ok(()),
        _ => None,
    };

    let first_json = first_text
        .as_ref()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok());

    let is_config = first_json
        .as_ref()
        .and_then(|v| v.get("type").and_then(|x| x.as_str()))
        == Some("config");

    let client_id: Option<String> = if is_config {
        first_json.as_ref().and_then(|v| {
            v.get("client_id")
                .and_then(|x| x.as_str())
                .map(String::from)
        })
    } else {
        None
    };

    let last_seq: u64 = if is_config {
        first_json
            .as_ref()
            .and_then(|v| v.get("last_seq").and_then(|x| x.as_u64()))
            .unwrap_or(0)
    } else {
        0
    };

    // Resolve session_id: check if client has an active session, otherwise generate new UUIDv7
    let (session_id, is_resume_candidate) = if let Some(ref cid) = client_id {
        if let Some(existing_sid) = registry.get_active_session_for_client(cid) {
            (existing_sid, true)
        } else {
            (uuid::Uuid::now_v7().simple().to_string(), false)
        }
    } else {
        (uuid::Uuid::now_v7().simple().to_string(), false)
    };

    let source_count = if is_config {
        first_json
            .as_ref()
            .and_then(|v| v.get("source_count").and_then(|x| x.as_u64()))
            .unwrap_or(1) as u8
    } else {
        1
    };

    let priority = if is_config {
        let mode = first_json
            .as_ref()
            .and_then(|v| v.get("mode").and_then(|x| x.as_str()))
            .unwrap_or("batch");
        if mode == "live" {
            dispatcher::Priority::High
        } else {
            dispatcher::Priority::Normal
        }
    } else {
        dispatcher::Priority::Normal
    };

    let pending_first_msg = if !is_config { first_text.clone() } else { None };
    let session_mode = if priority == dispatcher::Priority::High {
        session_registry::SessionMode::Live
    } else {
        session_registry::SessionMode::Batch
    };

    // --- Handle existing session ---

    if let Some(session) = is_resume_candidate
        .then(|| registry.get(&session_id))
        .flatten()
    {
        if session.end_of_stream.load(Ordering::Relaxed) {
            return replay_completed_session(ws_sender, &session, last_seq).await;
        }
        if session.mode == session_registry::SessionMode::Live {
            registry.cancel_live_timeout(&session_id);
            if session.end_of_stream.load(Ordering::Relaxed) {
                return replay_completed_session(ws_sender, &session, last_seq).await;
            }

            let (cb_tx, rx_ws) = setup_client_bridge();
            let _ = ws_sender
                .send(tokio_tungstenite::tungstenite::Message::Text(
                    serde_json::json!({"type": "session", "state": "resuming", "session_id": &session_id}).to_string(),
                ))
                .await;
            session.output_sink.replay_to_client(cb_tx, last_seq);
            let send_task = spawn_ws_forwarder(rx_ws, ws_sender);

            let mut writer = audio_writer::AudioWriter::new(&audio_dir, &settings);
            writer
                .resume_session(session.dir.clone(), source_count.clamp(1, 2))
                .map_err(|e| anyhow::anyhow!("resume_session: {}", e))?;
            info!(
                "Live session resumed: {} (client: {:?})",
                session_id, client_id
            );

            let processor = session.processor.clone();
            let output_sink = session.output_sink.clone();

            let poll_lag = spawn_poll_lag(
                session.dir.clone(),
                output_sink.clone(),
                processor.clone(),
                session.transcribed_end.clone(),
            );

            let writer_flushed = run_session_loop(
                &mut ws_receiver,
                &mut writer,
                &session,
                &processor,
                &output_sink,
                &registry,
                pending_first_msg.as_deref(),
            )
            .await?;

            output_sink.clear_client();
            poll_lag.abort();
            handle_disconnect(
                &session,
                &processor,
                &output_sink,
                &mut writer,
                writer_flushed,
                &settings,
                &registry,
            );
            let _ = send_task.await;
            return Ok(());
        }
    }

    // --- New session ---

    let mut writer = audio_writer::AudioWriter::new(&audio_dir, &settings);
    let session_dir = writer.start_session(Some(&session_id))?;
    info!("New session: {} (client: {:?})", session_id, client_id);
    info!("Session dir: {}", session_dir.display());
    writer.set_source_count(source_count.clamp(1, 2));

    session::write_session_meta(&session_dir, client_id.as_deref());

    let transcript_path = session_dir.join("transcript.jsonl");
    let (cb_tx, rx_ws) = setup_client_bridge();
    let output_sink = Arc::new(processor::OutputSink::new(transcript_path, Some(cb_tx)));

    let session_handle = asr_dispatcher.register_session(
        session_id.clone(),
        priority,
        output_sink.clone(),
        session_dir.clone(),
        settings.language.clone(),
    );
    let processor = Arc::new(dispatcher::FileProcessor::new(
        session_id.clone(),
        asr_dispatcher,
        session_handle,
    ));

    let session = Arc::new(session_registry::Session {
        id: session_id.clone(),
        client_id: client_id.clone(),
        dir: session_dir.clone(),
        mode: session_mode,
        processor: processor.clone(),
        output_sink: output_sink.clone(),
        transcribed_end: processor.transcribed_end.clone(),
        end_of_stream: std::sync::atomic::AtomicBool::new(false),
        processing_complete: std::sync::atomic::AtomicBool::new(false),
        created_at: std::time::Instant::now(),
        timeout_cancel_tx: std::sync::Mutex::new(None),
    });
    registry.insert(session_id.clone(), session.clone());

    if let Some(ref cid) = client_id {
        registry.set_client_session(cid.clone(), session_id.clone());
    }

    let _ = ws_sender
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({"type": "session", "state": "new", "session_id": &session_id})
                .to_string(),
        ))
        .await;

    let send_task = spawn_ws_forwarder(rx_ws, ws_sender);

    let poll_lag = spawn_poll_lag(
        session_dir,
        output_sink.clone(),
        processor.clone(),
        processor.transcribed_end.clone(),
    );

    let writer_flushed = run_session_loop(
        &mut ws_receiver,
        &mut writer,
        &session,
        &processor,
        &output_sink,
        &registry,
        pending_first_msg.as_deref(),
    )
    .await?;

    output_sink.clear_client();
    poll_lag.abort();
    handle_disconnect(
        &session,
        &processor,
        &output_sink,
        &mut writer,
        writer_flushed,
        &settings,
        &registry,
    );
    let _ = send_task.await;

    Ok(())
}
