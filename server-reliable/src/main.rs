//! Reliable disk-buffered transcription WebSocket server (Rust + GGML Whisper).

mod audio_writer;
mod config;
mod gigaam;
mod hallucination;
mod llm_corrector;
mod overlap;
mod processor;
mod session_cleanup;
mod transcript_postprocess;
mod vad;

use anyhow::Result;
use clap::Parser;
use crossbeam_channel;
use futures_util::{SinkExt, StreamExt};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::accept_async;
use tracing::info;

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

    // Keep insertion order and remove duplicates.
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
            if m.is_file() {
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
    let model_path = settings
        .model_path
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("model_path required in settings or --model"))?;

    info!("Loading Whisper: {} (use_gpu: {})", model_path, settings.use_gpu);
    add_cuda_to_path(&settings);
    let context = processor::load_whisper(model_path, settings.use_gpu)?;
    whisper_rs::print_system_info();
    let context = Arc::new(context);
    info!("Model loaded (check stderr above for CUBLAS/GPU — if CUBLAS=0, GPU not used)");

    let gigaam: Option<Arc<gigaam::GigaAM>> = if settings.ensemble_enabled {
        if let Some(dir) = settings.gigaam_model_dir.as_ref().filter(|s| !s.trim().is_empty()) {
            let path = std::path::Path::new(dir);
            match gigaam::GigaAM::new(path) {
                Ok(g) => {
                    info!("GigaAM ensemble enabled: {}", dir);
                    Some(Arc::new(g))
                }
                Err(e) => {
                    tracing::warn!("GigaAM load failed (ensemble disabled): {}", e);
                    None
                }
            }
        } else {
            tracing::warn!("ensemble_enabled but gigaam_model_dir not set");
            None
        }
    } else {
        None
    };

    let llm: Option<Arc<llm_corrector::LlmCorrector>> = if settings.llm_correction_enabled {
        info!(
            "LLM correction enabled: {} @ {}",
            settings.llm_model, settings.ollama_url
        );
        Some(Arc::new(llm_corrector::LlmCorrector::new(
            &settings.ollama_url,
            &settings.llm_model,
        )))
    } else {
        None
    };

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

    let mut session_id: u32 = 0;

    loop {
        let (stream, peer) = listener.accept().await?;
        session_id += 1;
        let sid = session_id;
        info!("Client connected: {} (session {})", peer, sid);

        let settings = settings.clone();
        let audio_dir = std::path::PathBuf::from(&settings.audio_dir);
        let context_clone = Arc::clone(&context);
        let gigaam_clone = gigaam.clone();
        let llm_clone = llm.clone();

        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            rt.block_on(async {
                if let Err(e) = handle_client(stream, settings, audio_dir, context_clone, gigaam_clone, llm_clone, sid).await {
                    tracing::error!("[session {}] error: {}", sid, e);
                }
                info!("Client disconnected: {} (session {})", peer, sid);
            });
        });
    }
}

async fn handle_client(
    stream: tokio::net::TcpStream,
    settings: config::Settings,
    audio_dir: std::path::PathBuf,
    context: Arc<whisper_rs::WhisperContext>,
    gigaam: Option<Arc<gigaam::GigaAM>>,
    llm: Option<Arc<llm_corrector::LlmCorrector>>,
    _session_id: u32,
) -> Result<()> {
    let state = context
        .create_state()
        .map_err(|e| anyhow::anyhow!("create_state: {:?}", e))?;
    let state = Arc::new(std::sync::Mutex::new(state));

    let ws_stream = accept_async(stream).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    let (cb_tx, cb_rx) = crossbeam_channel::unbounded::<processor::ClientMessage>();
    let cb_tx_poll = cb_tx.clone();
    let (tx_ws, mut rx_ws) = mpsc::channel::<processor::ClientMessage>(64);
    thread::spawn(move || {
        while let Ok(msg) = cb_rx.recv() {
            if tx_ws.blocking_send(msg).is_err() {
                break;
            }
        }
    });

    let processor = processor::FileProcessor::new(
        cb_tx,
        state,
        gigaam,
        llm,
        settings.language.clone(),
    );
    let transcribed_end = processor.transcribed_end.clone();

    let mut writer = audio_writer::AudioWriter::new(&audio_dir, &settings);
    let session_dir = writer.start_session()?;
    info!("Session dir: {}", session_dir.display());

    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx_ws.recv().await {
            let json = match &msg {
                processor::ClientMessage::Transcript { text, source } => {
                    serde_json::json!({"type": "transcript", "text": text, "source": source}).to_string()
                }
                processor::ClientMessage::Status(v) => v.to_string(),
            };
            if ws_sender
                .send(tokio_tungstenite::tungstenite::Message::Text(json))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    let session_dir_for_poll = session_dir.clone();
    let tx_out_poll = cb_tx_poll;
    let poll_lag = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        loop {
            interval.tick().await;
            let te0 = transcribed_end.get(0);
            let te1 = transcribed_end.get(1);
            let dir = session_dir_for_poll.clone();
            let (lag_0, lag_1, lag_max, size_mb) =
                tokio::task::spawn_blocking(move || {
                    scan_session_lag(&dir, te0, te1)
                })
                .await
                .unwrap_or((0.0, 0.0, 0.0, 0.0));
            let status = serde_json::json!({
                "type": "status",
                "lag_sec": (lag_max * 10.0).round() / 10.0,
                "lag_sec_0": (lag_0 * 10.0).round() / 10.0,
                "lag_sec_1": (lag_1 * 10.0).round() / 10.0,
                "audio_dir_size_mb": (size_mb * 100.0).round() / 100.0,
            });
            let _ = tx_out_poll.send(processor::ClientMessage::Status(status));
        }
    });

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
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                    let t = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    if t == "recording" {
                        if let Some(enabled) = v.get("enabled").and_then(|x| x.as_bool()) {
                            writer.set_recording(enabled);
                        }
                    } else if t == "config" {
                        if let Some(sc) = v.get("source_count").and_then(|x| x.as_u64()) {
                            writer.set_source_count((sc as u8).clamp(1, 2));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    poll_lag.abort();
    for (path, src) in writer.close()? {
        processor.enqueue(path, src);
    }
    processor.stop();

    drop(processor);
    let _ = send_task.await;

    Ok(())
}
