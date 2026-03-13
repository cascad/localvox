//! Reliable disk-buffered transcription WebSocket server (Rust + GGML Whisper).

mod asr;
mod audio_writer;
mod config;
mod dispatcher;
mod hallucination;
mod llm_corrector;
mod overlap;
mod processor;
mod session_cleanup;
mod session_registry;
mod transcript_postprocess;
mod vad;

use anyhow::Result;
use clap::Parser;
use crossbeam_channel;
use futures_util::{SinkExt, StreamExt};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::accept_async;
use tracing::info;

type WsSender = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    tokio_tungstenite::tungstenite::Message,
>;
type WsReceiver = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
>;

/// Prepend cuda_path (folder with cublas64_12.dll etc.) to PATH so whisper.cpp finds CUDA DLLs.
fn add_cuda_to_path(settings: &config::Settings) {
    let base = match &settings.cuda_path {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => return,
    };
    let base_path = std::path::Path::new(base);
    if !base_path.is_dir() {
        return;
    }

    // Accept either a CUDA root directory or a direct subdirectory (e.g. ...\Release).
    // If cuda_path is explicitly configured, prefer it over CUDA_PATH to avoid mixing
    // incompatible runtime DLL versions from different toolkits.
    let mut roots = vec![base_path.to_path_buf()];
    if let Some(name) = base_path.file_name().and_then(|n| n.to_str()) {
        if (name.eq_ignore_ascii_case("release")
            || name.eq_ignore_ascii_case("bin")
            || name.eq_ignore_ascii_case("lib"))
            && base_path.parent().is_some()
        {
            roots.push(base_path.parent().unwrap().to_path_buf());
        }
    }

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    for root in &roots {
        for sub in ["", "bin", "lib", "lib/x64", "Release"] {
            let dir = if sub.is_empty() {
                root.clone()
            } else {
                root.join(sub)
            };
            if dir.is_dir() {
                candidates.push(dir);
            }
        }
    }

    if cfg!(windows) {
        if let Ok(cuda_env) = std::env::var("CUDA_PATH") {
            let cuda_root = std::path::PathBuf::from(cuda_env);
            if cuda_root.is_dir() {
                for sub in ["", "bin", "lib", "lib/x64"] {
                    let dir = if sub.is_empty() {
                        cuda_root.clone()
                    } else {
                        cuda_root.join(sub)
                    };
                    if dir.is_dir() {
                        candidates.push(dir);
                    }
                }
            }
        }
    }

    let mut unique = std::collections::HashSet::new();
    candidates.retain(|p| unique.insert(p.to_string_lossy().to_ascii_lowercase()));
    if candidates.is_empty() {
        return;
    }

    if let Ok(prev) = std::env::var("PATH") {
        let sep = if cfg!(windows) { ";" } else { ":" };
        let prepend = candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(sep);
        std::env::set_var("PATH", format!("{}{}{}", prepend, sep, prev));
        info!("Added CUDA search dirs to PATH: {}", prepend);
    }
}

#[derive(Parser)]
#[command(name = "live-transcribe-server-reliable")]
#[command(about = "Reliable disk-buffered transcription server (GGML Whisper)")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value = "9745")]
    port: u16,

    /// Path to GGML Whisper model (.bin)
    #[arg(long)]
    model: Option<String>,
}

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

// ---------------------------------------------------------------------------
// Helper: crossbeam→mpsc bridge (OutputSink uses crossbeam, tokio WS needs mpsc)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper: spawn WS forwarder (mpsc → websocket)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper: replay completed session transcript, then close
// ---------------------------------------------------------------------------

async fn replay_completed_session(
    mut ws_sender: WsSender,
    session: &session_registry::Session,
) -> Result<()> {
    let (cb_tx, rx_ws) = setup_client_bridge();
    let _ = ws_sender
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({"type": "session", "state": "resuming"}).to_string(),
        ))
        .await;
    session.output_sink.replay_to_client(cb_tx);
    let send_task = spawn_ws_forwarder(rx_ws, ws_sender);
    let _ = send_task.await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: spawn poll_lag status reporter
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper: process a single text message from client
// ---------------------------------------------------------------------------

fn handle_text_msg(
    text: &str,
    writer: &mut audio_writer::AudioWriter,
    writer_flushed: &mut bool,
    session: &session_registry::Session,
    processor: &dispatcher::FileProcessor,
    output_sink: &processor::OutputSink,
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
        } else if t == "end_of_stream" {
            if !*writer_flushed {
                *writer_flushed = true;
                session.end_of_stream.store(true, Ordering::Relaxed);
                for (path, src) in writer.close()? {
                    info!("Segment ready src{}: {} -> queue (final)", src, path.display());
                    processor.enqueue(path, src);
                }
                processor.wait_until_empty(3600.0);
                std::thread::sleep(std::time::Duration::from_millis(500));
                output_sink.send(&processor::ClientMessage::Done);
                session.processing_complete.store(true, Ordering::Relaxed);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: run the main WS message loop (binary audio + text commands)
// ---------------------------------------------------------------------------

async fn run_session_loop(
    ws_receiver: &mut WsReceiver,
    writer: &mut audio_writer::AudioWriter,
    session: &session_registry::Session,
    processor: &dispatcher::FileProcessor,
    output_sink: &processor::OutputSink,
    pending_first_msg: Option<&str>,
) -> Result<bool> {
    let mut writer_flushed = false;
    if let Some(text) = pending_first_msg {
        handle_text_msg(text, writer, &mut writer_flushed, session, processor, output_sink)?;
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
                    &text, writer, &mut writer_flushed, session, processor, output_sink,
                )?;
            }
            _ => {}
        }
    }
    Ok(writer_flushed)
}

// ---------------------------------------------------------------------------
// Helper: handle disconnect — live timeout or batch finalization
// ---------------------------------------------------------------------------

fn handle_disconnect(
    session: &Arc<session_registry::Session>,
    processor: &Arc<dispatcher::FileProcessor>,
    output_sink: &Arc<processor::OutputSink>,
    writer: &mut audio_writer::AudioWriter,
    writer_flushed: bool,
    settings: &config::Settings,
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
        let timeout_sec = settings.live_reconnect_timeout_sec;
        tokio::spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(timeout_sec)) => {
                    s.end_of_stream.store(true, Ordering::Relaxed);
                    tokio::task::spawn_blocking(move || {
                        p.wait_until_empty(3600.0);
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        o.send(&processor::ClientMessage::Done);
                        s.processing_complete.store(true, Ordering::Relaxed);
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
        thread::spawn(move || {
            p.wait_until_empty(3600.0);
            std::thread::sleep(std::time::Duration::from_millis(500));
            o.send(&processor::ClientMessage::Done);
            s.processing_complete.store(true, Ordering::Relaxed);
        });
    }
}

// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    let settings = config::load_settings(args.model.as_deref())?;

    add_cuda_to_path(&settings);
    let models = asr::load_models(&settings)?;
    if models.is_empty() {
        anyhow::bail!(
            "No ASR models configured. Set model_path (legacy) or models array in settings.json"
        );
    }
    whisper_rs::print_system_info();

    // Таблица: модель → CPU/GPU
    let name_w = models.iter().map(|m| m.name().len()).max().unwrap_or(8).max(6);
    info!("ASR модели (кто на чём):");
    info!("  {:<name_w$} | Через", "Модель", name_w = name_w);
    info!("  {:-<name_w$}-+-------", "", name_w = name_w);
    for m in &models {
        info!("  {:<name_w$} | {}", m.name(), m.backend(), name_w = name_w);
    }

    let llm_dispatcher: Option<Arc<dispatcher::LlmDispatcher>> =
        if settings.llm_correction_enabled {
            info!(
                "LLM correction enabled: {} @ {} (global pool)",
                settings.llm_model, settings.ollama_url
            );
            let llm = Arc::new(llm_corrector::LlmCorrector::new(
                &settings.ollama_url,
                &settings.llm_model,
                &settings.llm_prompt_single,
                &settings.llm_prompt_ensemble,
                settings.llm_timeout_sec,
            ));
            Some(Arc::new(dispatcher::LlmDispatcher::new(
                llm,
                settings.llm_pool_size,
            )))
        } else {
            None
        };

    let asr_dispatcher = Arc::new(dispatcher::AsrDispatcher::new(
        models,
        llm_dispatcher.clone(),
        settings.asr_workers,
    ));
    info!("ASR dispatcher: {} workers (global)", settings.asr_workers);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Server ws://{}", addr);

    let cleanup_audio_dir = std::path::PathBuf::from(&settings.audio_dir);
    let cleanup_ttl = settings.session_ttl_hours;
    let cleanup_max_mb = settings.audio_dir_max_mb;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
        interval.tick().await;
        loop {
            interval.tick().await;
            let dir = cleanup_audio_dir.clone();
            let ttl = cleanup_ttl;
            let max = cleanup_max_mb;
            tokio::task::spawn_blocking(move || {
                session_cleanup::run(&dir, ttl, max, 2);
            })
            .await
            .ok();
        }
    });

    let registry = Arc::new(session_registry::SessionRegistry::new());

    let registry_evict = Arc::clone(&registry);
    let session_idle_timeout = settings.session_idle_timeout_sec;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        interval.tick().await;
        loop {
            interval.tick().await;
            let max_age = std::time::Duration::from_secs(session_idle_timeout);
            let evicted = registry_evict.evict_completed(max_age);
            if evicted > 0 {
                tracing::info!("Evicted {} completed session(s) from registry", evicted);
            }
        }
    });
    let mut session_counter: u32 = 0;

    loop {
        let (stream, peer) = listener.accept().await?;
        session_counter += 1;
        let sid = session_counter;
        info!("Client connected: {} (conn {})", peer, sid);

        let settings = settings.clone();
        let audio_dir = std::path::PathBuf::from(&settings.audio_dir);
        let asr_dispatcher_clone = Arc::clone(&asr_dispatcher);
        let registry_clone = Arc::clone(&registry);

        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            rt.block_on(async {
                if let Err(e) = handle_client(
                    stream,
                    settings,
                    audio_dir,
                    asr_dispatcher_clone,
                    registry_clone,
                    sid,
                )
                .await
                {
                    tracing::error!("[conn {}] error: {}", sid, e);
                }
                info!("Client disconnected: {} (conn {})", peer, sid);
            });
        });
    }
}

async fn handle_client(
    stream: tokio::net::TcpStream,
    settings: config::Settings,
    audio_dir: std::path::PathBuf,
    asr_dispatcher: Arc<dispatcher::AsrDispatcher>,
    registry: Arc<session_registry::SessionRegistry>,
    _conn_id: u32,
) -> Result<()> {
    let ws_stream = accept_async(stream).await?;
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

    let session_id = if is_config {
        first_json
            .as_ref()
            .and_then(|v| v.get("session_id").and_then(|x| x.as_str()).map(String::from))
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string())
    } else {
        uuid::Uuid::new_v4().simple().to_string()
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

    if let Some(session) = registry.get(&session_id) {
        if session.end_of_stream.load(Ordering::Relaxed) {
            return replay_completed_session(ws_sender, &session).await;
        }
        if session.mode == session_registry::SessionMode::Live {
            registry.cancel_live_timeout(&session_id);
            if session.end_of_stream.load(Ordering::Relaxed) {
                return replay_completed_session(ws_sender, &session).await;
            }

            let (cb_tx, rx_ws) = setup_client_bridge();
            let _ = ws_sender
                .send(tokio_tungstenite::tungstenite::Message::Text(
                    serde_json::json!({"type": "session", "state": "resuming"}).to_string(),
                ))
                .await;
            session.output_sink.replay_to_client(cb_tx);
            let send_task = spawn_ws_forwarder(rx_ws, ws_sender);

            let mut writer = audio_writer::AudioWriter::new(&audio_dir, &settings);
            writer
                .resume_session(session.dir.clone(), source_count.clamp(1, 2))
                .map_err(|e| anyhow::anyhow!("resume_session: {}", e))?;
            info!("Live session resumed: {}", session_id);

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
                pending_first_msg.as_deref(),
            )
            .await?;

            output_sink.clear_client();
            poll_lag.abort();
            handle_disconnect(&session, &processor, &output_sink, &mut writer, writer_flushed, &settings);
            let _ = send_task.await;
            return Ok(());
        }
    }

    // --- New session ---

    let mut writer = audio_writer::AudioWriter::new(&audio_dir, &settings);
    let session_dir = writer.start_session(Some(&session_id))?;
    info!("Session dir: {}", session_dir.display());
    writer.set_source_count(source_count.clamp(1, 2));

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

    let _ = ws_sender
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({"type": "session", "state": "new"}).to_string(),
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
        pending_first_msg.as_deref(),
    )
    .await?;

    output_sink.clear_client();
    poll_lag.abort();
    handle_disconnect(&session, &processor, &output_sink, &mut writer, writer_flushed, &settings);
    let _ = send_task.await;

    Ok(())
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
        } => {
            let mut j = serde_json::json!({
                "type": "transcript",
                "text": text,
                "source": source,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "seg_id": seg_id
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
