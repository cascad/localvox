//! WebSocket I/O thread: connect, config, message loop.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tungstenite::Message;
use url::Url;

use crate::types::{ServerMessage, UiEvent, WsOutgoing};

#[allow(clippy::too_many_arguments)]
pub fn ws_io_thread(
    server_url: String,
    out_rx: mpsc::Receiver<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: Arc<AtomicBool>,
    source_count: u8,
    client_id: Option<String>,
    pending_end_session: Arc<AtomicBool>,
    initial_recording: bool,
    api_key: Option<String>,
    tls_insecure: bool,
    tls_ca_path: Option<String>,
) {
    let last_seq = AtomicU64::new(0);
    let reconnect_delay = Duration::from_secs(2);

    let parsed_url = match Url::parse(&server_url) {
        Ok(u) => u,
        Err(e) => {
            let _ = ui_tx.send(UiEvent::Disconnected {
                reason: format!("некорректный URL: {e}"),
            });
            return;
        }
    };

    while running.load(Ordering::Relaxed) {
        let host = parsed_url
            .host_str()
            .expect("URL сервера должен содержать host");
        let host = if host == "localhost" {
            "127.0.0.1"
        } else {
            host
        };
        let port = parsed_url
            .port()
            .expect("URL сервера должен содержать порт");
        let addr = format!("{}:{}", host, port);

        let sock_addr: std::net::SocketAddr = addr
            .parse()
            .unwrap_or_else(|_| panic!("не удалось распарсить адрес сервера «{addr}»"));

        let uri: http::Uri = match server_url.parse() {
            Ok(u) => u,
            Err(e) => {
                let _ = ui_tx.send(UiEvent::Disconnected {
                    reason: format!("некорректный URI: {e}"),
                });
                return;
            }
        };

        let mut builder = tungstenite::client::ClientRequestBuilder::new(uri.clone());
        if let Some(ref key) = api_key {
            if !key.is_empty() {
                builder = builder.with_header("Authorization", format!("Bearer {key}"));
            }
        }

        let tcp = match std::net::TcpStream::connect_timeout(&sock_addr, Duration::from_secs(5)) {
            Ok(s) => s,
            Err(e) => {
                let _ = ui_tx.send(UiEvent::Disconnected {
                    reason: format!("подключение к {sock_addr}: {e}"),
                });
                for _ in 0..20 {
                    if !running.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                continue;
            }
        };

        let connector = if server_url.starts_with("wss://") {
            let mut tls_builder = native_tls::TlsConnector::builder();
            if tls_insecure {
                tls_builder.danger_accept_invalid_certs(true);
            }
            if let Some(ref ca_path) = tls_ca_path {
                match std::fs::read(ca_path) {
                    Ok(pem) => match native_tls::Certificate::from_pem(&pem) {
                        Ok(cert) => {
                            tls_builder.add_root_certificate(cert);
                        }
                        Err(e) => {
                            let _ = ui_tx.send(UiEvent::Disconnected {
                                reason: format!("TLS CA cert parse: {e}"),
                            });
                            continue;
                        }
                    },
                    Err(e) => {
                        let _ = ui_tx.send(UiEvent::Disconnected {
                            reason: format!("TLS CA read {ca_path}: {e}"),
                        });
                        continue;
                    }
                }
            }
            let c = match tls_builder.build() {
                Ok(c) => c,
                Err(e) => {
                    let _ = ui_tx.send(UiEvent::Disconnected {
                        reason: format!("TLS connector: {e}"),
                    });
                    continue;
                }
            };
            Some(tungstenite::Connector::NativeTls(c))
        } else {
            None
        };

        let connect_result =
            tungstenite::client_tls_with_config(builder, tcp, None, connector);

        let (mut ws, _) = match connect_result {
            Ok(x) => x,
            Err(e) => {
                let _ = ui_tx.send(UiEvent::Disconnected {
                    reason: format!("ws handshake: {e}"),
                });
                for _ in 0..20 {
                    if !running.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                continue;
            }
        };

        // Send config with client_id and last_seq (server assigns/resumes session)
        let current_last_seq = last_seq.load(Ordering::Relaxed);
        let config_msg = if let Some(ref cid) = client_id {
            serde_json::json!({
                "type": "config",
                "source_count": source_count,
                "mode": "live",
                "client_id": cid,
                "last_seq": current_last_seq,
            })
        } else {
            serde_json::json!({
                "type": "config",
                "source_count": source_count,
                "mode": "live",
                "last_seq": current_last_seq,
            })
        };
        if ws.send(Message::Text(config_msg.to_string())).is_err() {
            let _ = ui_tx.send(UiEvent::Disconnected {
                reason: "ошибка отправки config".into(),
            });
            thread::sleep(reconnect_delay);
            continue;
        }
        let recording_msg = serde_json::json!({"type": "recording", "enabled": initial_recording});
        if ws.send(Message::Text(recording_msg.to_string())).is_err() {
            let _ = ui_tx.send(UiEvent::Disconnected {
                reason: "ошибка отправки recording".into(),
            });
            thread::sleep(reconnect_delay);
            continue;
        }

        let client_id_note = client_id
            .as_deref()
            .map(|c| format!("client_id={c}"))
            .unwrap_or_else(|| "client_id=(none)".into());
        let transport = if server_url.starts_with("wss://") {
            match ws.get_ref() {
                tungstenite::stream::MaybeTlsStream::NativeTls(tls) => {
                    let verify = if tls_insecure {
                        "server cert verification OFF (insecure)"
                    } else if tls_ca_path.is_some() {
                        "server cert verified (custom CA / pinned PEM)"
                    } else {
                        "server cert verified (system trust store)"
                    };
                    let peer_cert = tls
                        .peer_certificate()
                        .ok()
                        .flatten()
                        .and_then(|c| c.to_der().ok())
                        .map(|der| format!("peer leaf cert {} bytes DER", der.len()))
                        .unwrap_or_else(|| "peer cert n/a".into());
                    format!(
                        "[ws] WSS: encrypted; {verify}; {peer_cert}; {client_id_note} (TLS version/cipher — в логе сервера)"
                    )
                }
                tungstenite::stream::MaybeTlsStream::Plain(_) => {
                    format!("[ws] WSS URL but stream is plain TCP (unexpected); {client_id_note}")
                }
                _ => format!("[ws] WSS: unknown TLS stream type; {client_id_note}"),
            }
        } else {
            format!(
                "[ws] WS: plaintext (no TLS); {client_id_note} — для шифрования используйте wss://"
            )
        };
        let _ = ui_tx.send(UiEvent::Debug { text: transport });

        let _ = ui_tx.send(UiEvent::Connected);

        match ws.get_ref() {
            tungstenite::stream::MaybeTlsStream::Plain(s) => {
                let _ = s.set_read_timeout(Some(Duration::from_millis(20)));
            }
            tungstenite::stream::MaybeTlsStream::NativeTls(s) => {
                let _ = s.get_ref().set_read_timeout(Some(Duration::from_millis(20)));
            }
            _ => {}
        }

        let mut write_errors = 0u32;
        let mut disconnected = false;

        while running.load(Ordering::Relaxed) && !disconnected {
            match ws.read() {
                Ok(Message::Text(text)) => {
                    if let Ok(srv_msg) = serde_json::from_str::<ServerMessage>(&text) {
                        if matches!(&srv_msg, ServerMessage::Done) {
                            if pending_end_session.swap(false, Ordering::Relaxed) {
                                last_seq.store(0, Ordering::Relaxed);
                                let _ = ui_tx.send(UiEvent::Debug {
                                    text: "[session] Done (end_session), переподключение".into(),
                                });
                                let _ = ui_tx.send(UiEvent::Server(srv_msg));
                                disconnected = true;
                            }
                        } else if let ServerMessage::Session {
                            ref state,
                            ref session_id,
                        } = srv_msg
                        {
                            let sid_str = session_id.as_deref().unwrap_or("?");
                            if state == "new" {
                                last_seq.store(0, Ordering::Relaxed);
                            }
                            let _ = ui_tx.send(UiEvent::Debug {
                                text: format!("[session] state={}, id={}", state, sid_str),
                            });
                        } else if let ServerMessage::Transcript { seq, .. } = &srv_msg {
                            if *seq > 0 {
                                last_seq.store(*seq, Ordering::Relaxed);
                            }
                            let _ = ui_tx.send(UiEvent::Server(srv_msg));
                        } else {
                            let _ = ui_tx.send(UiEvent::Server(srv_msg));
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    disconnected = true;
                    let _ = ui_tx.send(UiEvent::Disconnected {
                        reason: "сервер закрыл соединение".into(),
                    });
                }
                Err(tungstenite::Error::Io(ref e))
                    if e.kind() == io::ErrorKind::WouldBlock
                        || e.kind() == io::ErrorKind::TimedOut => {}
                Err(e) => {
                    disconnected = true;
                    let _ = ui_tx.send(UiEvent::Disconnected {
                        reason: format!("{e}"),
                    });
                }
                _ => {}
            }

            while let Ok(msg) = out_rx.try_recv() {
                let result = match msg {
                    WsOutgoing::Binary(data) => ws.send(Message::Binary(data)),
                    WsOutgoing::Text(text) => ws.send(Message::Text(text)),
                };
                match result {
                    Ok(_) => write_errors = 0,
                    Err(e) => {
                        write_errors += 1;
                        if write_errors >= 10 {
                            disconnected = true;
                            let _ = ui_tx.send(UiEvent::Disconnected {
                                reason: format!("ошибки записи: {e}"),
                            });
                            break;
                        }
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        }

        let _ = ws.close(None);
        if running.load(Ordering::Relaxed) {
            for _ in 0..20 {
                if !running.load(Ordering::Relaxed) {
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    }

    running.store(false, Ordering::Relaxed);
    let _ = ui_tx.send(UiEvent::Quit);
}
