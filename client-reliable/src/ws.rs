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

pub fn ws_io_thread(
    server_url: String,
    out_rx: mpsc::Receiver<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: Arc<AtomicBool>,
    source_count: u8,
    client_id: Option<String>,
    pending_end_session: Arc<AtomicBool>,
    initial_recording: bool,
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
        let host = if host == "localhost" { "127.0.0.1" } else { host };
        let port = parsed_url
            .port()
            .expect("URL сервера должен содержать порт");
        let addr = format!("{}:{}", host, port);

        let sock_addr: std::net::SocketAddr = addr
            .parse()
            .unwrap_or_else(|_| panic!("не удалось распарсить адрес сервера «{addr}»"));

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
        let (mut ws, _) = match tungstenite::client::client_with_config(&server_url, tcp, None) {
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

        let _ = ui_tx.send(UiEvent::Connected);

        let _ = ws.get_ref().set_read_timeout(Some(Duration::from_millis(20)));

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
                        } else if let ServerMessage::Session { ref state, ref session_id } = srv_msg {
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
                    if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut =>
                {
                }
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
                    WsOutgoing::Binary(data) => ws.send(Message::Binary(data.into())),
                    WsOutgoing::Text(text) => ws.send(Message::Text(text.into())),
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
