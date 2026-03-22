//! Worker: download → convert → transcribe → save.

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::overlap::merge_overlap;
use crate::state::{QueueState, ServerStatus};

fn merge_transcript_chunks(transcripts: &[(f64, String)]) -> String {
    let mut merged = String::new();
    let mut prev_tail = String::new();
    for (_, text) in transcripts {
        let (to_emit, new_tail, no_space_before) = merge_overlap(&prev_tail, text);
        if !merged.is_empty() && !to_emit.is_empty() && !no_space_before {
            merged.push(' ');
        }
        merged.push_str(&to_emit);
        prev_tail = new_tail;
    }
    merged
}

const CHUNK_BYTES: usize = 8192;
const POLL_MS: u64 = 500;

async fn read_session_state(
    read: &mut futures_util::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
) -> Result<String, anyhow::Error> {
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(s)) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    if v.get("type").and_then(|x| x.as_str()) == Some("session") {
                        return Ok(v
                            .get("state")
                            .and_then(|x| x.as_str())
                            .unwrap_or("new")
                            .to_string());
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(Message::Binary(_))
            | Ok(Message::Ping(_))
            | Ok(Message::Pong(_))
            | Ok(Message::Frame(_)) => {}
            Err(_) => break,
        }
    }
    Ok("new".to_string())
}

pub async fn run_worker(
    state: Arc<std::sync::Mutex<QueueState>>,
    state_path: PathBuf,
    _output_dir: PathBuf,
    server: String,
    api_key: Option<String>,
    yt_dlp: &str,
    ffmpeg: &str,
    ffmpeg_location: Option<String>,
    js_runtime: Option<String>,
    raw_mode: bool,
) {
    loop {
        let idx = {
            let mut s = state.lock().unwrap();
            s.next_pending()
        };

        let Some(idx) = idx else {
            tokio::time::sleep(Duration::from_millis(POLL_MS)).await;
            continue;
        };

        let (url, out_path, session_id) = {
            let mut s = state.lock().unwrap();
            s.set_in_progress(idx);
            s.save(&state_path).ok();
            let j = &s.jobs[idx];
            let sid = if j.session_id.is_empty() {
                j.id.clone()
            } else {
                j.session_id.clone()
            };
            (j.url.clone(), PathBuf::from(&j.output_path), sid)
        };

        let result = process_one(
            &url,
            &out_path,
            &session_id,
            &server,
            api_key.as_deref(),
            yt_dlp,
            ffmpeg,
            ffmpeg_location.as_deref(),
            js_runtime.as_deref(),
            state.clone(),
            idx,
            &state_path,
            raw_mode,
        )
        .await;

        {
            let mut s = state.lock().unwrap();
            match result {
                Ok(()) => s.set_done(idx),
                Err(e) => s.set_failed(idx, e.to_string()),
            }
            s.save(&state_path).ok();
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn process_one(
    url: &str,
    output_path: &PathBuf,
    session_id: &str,
    server: &str,
    api_key: Option<&str>,
    yt_dlp: &str,
    ffmpeg: &str,
    ffmpeg_location: Option<&str>,
    js_runtime: Option<&str>,
    state: Arc<std::sync::Mutex<QueueState>>,
    job_idx: usize,
    state_path: &PathBuf,
    raw_mode: bool,
) -> Result<()> {
    let temp = download_audio(yt_dlp, url, ffmpeg_location, js_runtime)?;
    let pcm = convert_to_pcm(ffmpeg, &temp)?;
    let _ = std::fs::remove_file(&temp);

    let mut request =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(server)
            .map_err(|e| anyhow::anyhow!("invalid server URL: {}", e))?;
    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        request
            .headers_mut()
            .insert("Authorization", format!("Bearer {key}").parse().unwrap());
    }
    let (ws_stream, _) = connect_async(request).await.context("connect to server")?;
    let (mut write, mut read) = ws_stream.split();

    write
        .send(Message::Text(
            serde_json::json!({
                "type": "config",
                "source_count": 1,
                "client_id": session_id,
                "mode": "batch"
            })
            .to_string(),
        ))
        .await?;

    let session_state = read_session_state(&mut read).await?;
    let resuming = session_state == "resuming";

    let send_handle = if resuming {
        tokio::spawn(async { Ok::<(), anyhow::Error>(()) })
    } else {
        write
            .send(Message::Text(
                serde_json::json!({"type": "recording", "enabled": true}).to_string(),
            ))
            .await?;

        let total_bytes = pcm.len();
        let pcm = std::sync::Arc::new(pcm);
        let pcm_send = pcm.clone();
        let state_send = state.clone();
        let path_send = state_path.clone();

        tokio::spawn(async move {
            let mut sent_bytes = 0usize;
            for chunk in pcm_send.chunks(CHUNK_BYTES) {
                let mut msg = Vec::with_capacity(1 + chunk.len());
                msg.push(0u8);
                msg.extend_from_slice(chunk);
                if write.send(Message::Binary(msg.into())).await.is_err() {
                    break;
                }
                sent_bytes += chunk.len();
                if sent_bytes % (CHUNK_BYTES * 32) == 0 || sent_bytes == total_bytes {
                    if let Ok(mut s) = state_send.lock() {
                        s.set_progress(job_idx, sent_bytes, total_bytes);
                        let _ = s.save(&path_send);
                    }
                }
            }
            if let Ok(mut s) = state_send.lock() {
                s.set_progress(job_idx, total_bytes, total_bytes);
            }
            let _ = write
                .send(Message::Text(
                    serde_json::json!({"type": "end_of_stream"}).to_string(),
                ))
                .await;
            Ok(())
        })
    };

    let mut transcripts: Vec<(f64, String)> = Vec::new();
    let mut done = false;
    while !done {
        match read.next().await {
            Some(Ok(Message::Text(s))) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    match v.get("type").and_then(|x| x.as_str()) {
                        Some("transcript") => {
                            if let Some(text) = v.get("text").and_then(|x| x.as_str()) {
                                let start_sec =
                                    v.get("start_sec").and_then(|x| x.as_f64()).unwrap_or(0.0);
                                transcripts.push((start_sec, text.to_string()));
                            }
                        }
                        Some("done") => done = true,
                        Some("status") => {
                            let status = ServerStatus {
                                task_queue_size: v
                                    .get("task_queue_size")
                                    .and_then(|x| x.as_u64())
                                    .unwrap_or(0)
                                    as usize,
                                llm_queue: v.get("llm_queue").and_then(|x| x.as_u64()).unwrap_or(0)
                                    as usize,
                                lag_sec: v.get("lag_sec").and_then(|x| x.as_f64()).unwrap_or(0.0),
                            };
                            if let Ok(mut s) = state.lock() {
                                s.set_server_status(job_idx, status);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some(Ok(Message::Close(_))) => break,
            Some(Err(_)) => break,
            None => break,
            _ => {}
        }
    }

    let _ = send_handle.await;

    transcripts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let output = if raw_mode {
        transcripts
            .iter()
            .map(|(t, text)| format!("[{:.1}] {}", t, text))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        merge_transcript_chunks(&transcripts)
    };
    if let Some(parent) = output_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(output_path, &output).context("write transcript")?;

    Ok(())
}

fn download_audio(
    yt_dlp: &str,
    url: &str,
    ffmpeg_location: Option<&str>,
    js_runtime: Option<&str>,
) -> Result<PathBuf> {
    let base = std::env::temp_dir().join(format!("yt_transcribe_{}", std::process::id()));
    let out_template = base.with_extension("%(ext)s");
    let mut args = vec![
        "-x",
        "-f",
        "bestaudio",
        "-o",
        out_template.to_str().unwrap(),
    ];
    if let Some(loc) = ffmpeg_location {
        args.push("--ffmpeg-location");
        args.push(loc);
    }
    if let Some(rt) = js_runtime {
        args.push("--js-runtime");
        args.push(rt);
    }
    args.push(url);
    let status = Command::new(yt_dlp)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .context("yt-dlp (install: pip install yt-dlp)")?;
    if !status.success() {
        anyhow::bail!("yt-dlp failed");
    }
    for ext in ["webm", "m4a", "opus", "ogg", "mp3"] {
        let p = base.with_extension(ext);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!("yt-dlp produced no file")
}

fn convert_to_pcm(ffmpeg: &str, input: &PathBuf) -> Result<Vec<u8>> {
    let mut output = Command::new(ffmpeg)
        .args([
            "-i",
            input.to_str().unwrap(),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("ffmpeg (install ffmpeg)")?;

    let mut stdout = output.stdout.take().context("ffmpeg stdout")?;
    let mut pcm = Vec::new();
    stdout.read_to_end(&mut pcm)?;
    let _ = output.wait();
    Ok(pcm)
}
