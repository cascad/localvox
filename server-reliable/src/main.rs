//! Reliable disk-buffered transcription WebSocket server (Rust + GGML Whisper).

mod asr;
mod audio_writer;
mod client_handler;
mod config;
mod dispatcher;
mod hallucination;
mod llm_corrector;
mod overlap;
mod processor;
mod session;
mod session_cleanup;
mod session_registry;
mod tls;
mod transcript_postprocess;
mod vad;

use anyhow::Result;
use clap::Parser;
use config::sha2_hash_hex;
use std::sync::Arc;
use std::thread;
use tokio::net::TcpListener;
use tokio_util::either::Either;
use tracing::info;

fn generate_api_key() -> String {
    let mut bytes = [0u8; 32];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut bytes);
    hex::encode(bytes)
}

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

    /// Generate a new API key and print it (for adding to settings.json api_keys)
    #[arg(long)]
    generate_key: bool,

    /// Optional name for the generated key
    #[arg(long, requires = "generate_key")]
    key_name: Option<String>,
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

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let args = Args::parse();

    if args.generate_key {
        let key = generate_api_key();
        let name = args.key_name.as_deref().unwrap_or("default");
        let hash = sha2_hash_hex(&key);
        println!("Add this to settings.json api_keys:");
        println!(
            "  {{ \"name\": \"{}\", \"hash\": \"sha256:{}\" }}",
            name, hash
        );
        println!("\nAPI key (give to client, store securely):");
        println!("{}", key);
        return Ok(());
    }

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
    let name_w = models
        .iter()
        .map(|m| m.name().len())
        .max()
        .unwrap_or(8)
        .max(6);
    info!("ASR модели (кто на чём):");
    info!("  {:<name_w$} | Через", "Модель", name_w = name_w);
    info!("  {:-<name_w$}-+-------", "", name_w = name_w);
    for m in &models {
        info!("  {:<name_w$} | {}", m.name(), m.backend(), name_w = name_w);
    }

    asr::warmup(&models);

    let llm_dispatcher: Option<Arc<dispatcher::LlmDispatcher>> = if settings.llm_correction_enabled
    {
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

    let tls_acceptor = match (&settings.tls_cert_path, &settings.tls_key_path) {
        (Some(cert), Some(key)) if !cert.is_empty() && !key.is_empty() => {
            let cert_path = std::path::Path::new(cert);
            let key_path = std::path::Path::new(key);
            Some(tls::load_tls_acceptor(cert_path, key_path)?)
        }
        _ => None,
    };

    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).await?;
    let scheme = if tls_acceptor.is_some() { "wss" } else { "ws" };
    info!(
        "Server {}://{} (auth: {})",
        scheme,
        addr,
        if settings.auth_enabled() { "on" } else { "off" }
    );

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

    session::startup_session_scan(std::path::Path::new(&settings.audio_dir), &registry);

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

        let stream = if let Some(ref acceptor) = tls_acceptor {
            match acceptor.accept(stream).await {
                Ok(tls_stream) => {
                    let rustls_conn = tls_stream.get_ref().1;
                    let tls_ver = rustls_conn
                        .protocol_version()
                        .map(|v| format!("{v:?}"))
                        .unwrap_or_else(|| "unknown".into());
                    let cipher = rustls_conn
                        .negotiated_cipher_suite()
                        .map(|cs| format!("{:?}", cs.suite()))
                        .unwrap_or_else(|| "n/a".into());
                    info!(
                        "Client connected: {} (conn {}, transport: WSS, TLS {}, cipher {})",
                        peer, sid, tls_ver, cipher
                    );
                    tracing::debug!(
                        target: "localvox::tls",
                        conn = sid,
                        peer = %peer,
                        tls_version = %tls_ver,
                        cipher_suite = %cipher,
                        "TLS handshake completed"
                    );
                    Either::Right(tls_stream)
                }
                Err(e) => {
                    tracing::error!("[conn {}] TLS handshake failed: {}", sid, e);
                    continue;
                }
            }
        } else {
            info!(
                "Client connected: {} (conn {}, transport: plain WS, TLS disabled)",
                peer, sid
            );
            Either::Left(stream)
        };

        let settings = settings.clone();
        let audio_dir = std::path::PathBuf::from(&settings.audio_dir);
        let asr_dispatcher_clone = Arc::clone(&asr_dispatcher);
        let registry_clone = Arc::clone(&registry);

        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            rt.block_on(async {
                if let Err(e) = client_handler::handle_client(
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
