//! Проверка `wss://` / `ws://` до сервера LocalVox: TLS + WebSocket handshake,
//! минимальный `config` как у клиента, затем печать входящих кадров.

use std::io;
use std::net::ToSocketAddrs;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use tungstenite::Message;
use url::Url;

#[derive(Parser, Debug)]
#[command(name = "wss-probe")]
#[command(about = "Probe WSS/WebSocket to LocalVox server and print frames from the server")]
struct Cli {
    /// Server URL, e.g. wss://203.0.113.10:9745 or wss://xxx.ngrok-free.app (port 443 implied)
    #[arg(value_name = "URL")]
    server_url: String,

    /// API key (or set LOCALVOX_API_KEY)
    #[arg(long, env = "LOCALVOX_API_KEY")]
    api_key: Option<String>,

    #[arg(long, default_value_t = false)]
    tls_insecure: bool,

    #[arg(long)]
    tls_ca_path: Option<String>,

    /// Optional client_id for config (probe only; default none)
    #[arg(long)]
    client_id: Option<String>,

    /// Stop after this many seconds (0 = until Ctrl+C)
    #[arg(long, default_value_t = 30)]
    for_sec: u64,

    /// Stop after this many messages from server (0 = no limit)
    #[arg(long, default_value_t = 0)]
    max_msgs: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let server_url = cli.server_url.trim().to_string();
    if server_url.is_empty() {
        anyhow::bail!("URL is empty");
    }

    let parsed = Url::parse(&server_url).context("parse URL")?;
    let host = parsed.host_str().context("URL must have a host")?;
    let host = if host == "localhost" {
        "127.0.0.1"
    } else {
        host
    };
    // `url` returns None for default ports (wss→443, ws→80) even if omitted or canonicalized.
    let port = parsed
        .port_or_known_default()
        .context("URL must use ws:// or wss:// (unknown scheme)")?;
    // Hostname (ngrok, DNS) is not a SocketAddr — resolve via DNS.
    let sock_addr = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("DNS lookup failed for {host}:{port}"))?
        .next()
        .with_context(|| format!("no addresses resolved for {host}:{port}"))?;

    let uri: http::Uri = server_url.parse().context("parse URI for WebSocket")?;

    let mut builder = tungstenite::client::ClientRequestBuilder::new(uri);
    if let Some(ref key) = cli.api_key {
        let key = key.trim();
        if !key.is_empty() {
            builder = builder.with_header("Authorization", format!("Bearer {key}"));
        }
    }

    let t0 = Instant::now();
    let tcp = std::net::TcpStream::connect_timeout(&sock_addr, Duration::from_secs(10))
        .with_context(|| format!("TCP connect {sock_addr}"))?;

    let connector = if server_url.starts_with("wss://") {
        let mut tls_builder = native_tls::TlsConnector::builder();
        if cli.tls_insecure {
            tls_builder.danger_accept_invalid_certs(true);
        }
        if let Some(ref ca_path) = cli.tls_ca_path {
            let pem =
                std::fs::read(ca_path).with_context(|| format!("read tls_ca_path {ca_path}"))?;
            let cert =
                native_tls::Certificate::from_pem(&pem).context("parse PEM as certificate")?;
            tls_builder.add_root_certificate(cert);
        }
        let c = tls_builder.build().context("build TLS connector")?;
        Some(tungstenite::Connector::NativeTls(c))
    } else {
        None
    };

    let (mut ws, _resp) = tungstenite::client_tls_with_config(builder, tcp, None, connector)
        .map_err(|e| anyhow::anyhow!("WebSocket handshake: {e}"))?;

    eprintln!(
        "[probe] WebSocket OK after {:?} (url={})",
        t0.elapsed(),
        server_url
    );

    if server_url.starts_with("wss://") {
        match ws.get_ref() {
            tungstenite::stream::MaybeTlsStream::NativeTls(tls) => {
                let verify = if cli.tls_insecure {
                    "insecure (any cert)"
                } else if cli.tls_ca_path.is_some() {
                    "custom CA / PEM"
                } else {
                    "system trust store"
                };
                let der = tls
                    .peer_certificate()
                    .ok()
                    .flatten()
                    .and_then(|c| c.to_der().ok())
                    .map(|d| d.len())
                    .unwrap_or(0);
                eprintln!("[probe] TLS: verify={verify}; peer leaf cert ~{der} bytes DER");
            }
            tungstenite::stream::MaybeTlsStream::Plain(_) => {
                eprintln!("[probe] WARN: wss:// but underlying stream is plain TCP");
            }
            _ => {}
        }
    }

    // Same bootstrap as live client so the server keeps the socket open.
    let config = if let Some(ref cid) = cli.client_id {
        serde_json::json!({
            "type": "config",
            "source_count": 1u8,
            "mode": "live",
            "client_id": cid,
            "last_seq": 0u64,
        })
    } else {
        serde_json::json!({
            "type": "config",
            "source_count": 1u8,
            "mode": "live",
            "last_seq": 0u64,
        })
    };
    ws.send(Message::Text(config.to_string()))
        .context("send config")?;
    let recording = serde_json::json!({"type": "recording", "enabled": false});
    ws.send(Message::Text(recording.to_string()))
        .context("send recording")?;

    eprintln!("[probe] sent config + recording; printing server frames…");

    match ws.get_ref() {
        tungstenite::stream::MaybeTlsStream::Plain(s) => {
            let _ = s.set_read_timeout(Some(Duration::from_millis(500)));
        }
        tungstenite::stream::MaybeTlsStream::NativeTls(s) => {
            let _ = s
                .get_ref()
                .set_read_timeout(Some(Duration::from_millis(500)));
        }
        _ => {}
    }

    let deadline = if cli.for_sec > 0 {
        Some(Instant::now() + Duration::from_secs(cli.for_sec))
    } else {
        None
    };
    let mut count: u64 = 0;

    loop {
        if let Some(d) = deadline {
            if Instant::now() >= d {
                eprintln!("[probe] done (--for-sec)");
                break;
            }
        }
        if cli.max_msgs > 0 && count >= cli.max_msgs {
            eprintln!("[probe] done (--max-msgs)");
            break;
        }

        match ws.read() {
            Ok(msg) => {
                count += 1;
                match msg {
                    Message::Text(t) => println!("{t}"),
                    Message::Binary(b) => {
                        let preview = b.len().min(64);
                        println!("<binary {} bytes> {}", b.len(), hex_preview(&b[..preview]));
                    }
                    Message::Ping(p) => {
                        eprintln!("[probe] << Ping({} bytes)", p.len());
                        let _ = ws.send(Message::Pong(p));
                    }
                    Message::Pong(p) => eprintln!("[probe] << Pong({} bytes)", p.len()),
                    Message::Close(f) => {
                        eprintln!("[probe] << Close {f:?}");
                        break;
                    }
                    Message::Frame(f) => eprintln!("[probe] << Raw frame {:?}", f),
                }
            }
            Err(tungstenite::Error::Io(ref e))
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut => {
            }
            Err(e) => {
                eprintln!("[probe] read error: {e}");
                break;
            }
        }
    }

    let _ = ws.close(None);
    Ok(())
}

fn hex_preview(slice: &[u8]) -> String {
    slice
        .iter()
        .take(32)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}
