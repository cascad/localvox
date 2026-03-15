mod audio;
mod config;
mod summarize;
mod transcript;
mod tui;
mod types;
mod ws;
pub use audio::{audio_capture, collect_input_devices, default_device, format_device_display, key_matches, list_output_device_names, loopback_capture, physical_key, resample, resolve_device, to_mono, SAMPLE_RATE};
pub use config::{get_config_save_path, load_config, ClientConfig};
pub use summarize::spawn_summarize_ollama;
pub use transcript::{export_trimmed_transcript, TranscriptStore};
pub use types::*;
pub use tui::run_tui;
pub use ws::ws_io_thread;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};
use url::Url;

// ── CLI ──────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "live-transcribe-client-reliable", about = "Клиент reliable-transcribe с отображением отставания")]
struct Cli {
    /// WebSocket-сервер
    #[arg(short, long)]
    server: Option<String>,

    /// Микрофон (индекс или подстрока имени)
    #[arg(short, long)]
    device: Option<String>,

    /// Системный звук — default-output (дефолт) или индекс/имя устройства (напр. Razer)
    #[arg(long)]
    loopback: Option<String>,

    /// Второй микрофон (индекс или имя). Игнорируется если задан --loopback
    #[arg(long)]
    device2: Option<String>,

    /// Файл для записи транскрипции
    #[arg(short, long)]
    output: Option<String>,

    /// Показать список аудиоустройств и выйти
    #[arg(short, long)]
    list_devices: bool,

    /// Редактор настроек (выбор устройств, сохранение в конфиг)
    #[arg(long)]
    configure: bool,

    /// Путь к конфигу (по умолчанию ищет client-config.json)
    #[arg(short, long)]
    config: Option<String>,
}

fn list_devices() {
    let host = cpal::default_host();

    eprintln!("Устройства ввода (микрофоны):");
    for (i, (dev, name)) in collect_input_devices().into_iter().enumerate() {
        let cfg = dev.default_input_config();
        let info = match cfg {
            Ok(c) => format!("rate {}, ch {}", c.sample_rate(), c.channels()),
            Err(_) => "no config".into(),
        };
        let is_default = host
            .default_input_device()
            .map(|d| d.name().unwrap_or_default() == name)
            .unwrap_or(false);
        let extra = if is_default {
            format!("{info} (по умолчанию)")
        } else {
            info
        };
        let display = format_device_display(&dev, &name, &extra);
        eprintln!("  [{i}] {display}");
    }

    eprintln!();
    eprintln!("Устройства вывода (loopback):");
    #[cfg(any(windows, target_os = "macos"))]
    eprintln!("  [default-output] системный звук (по умолчанию)");
    for (i, dev) in host.output_devices().unwrap().enumerate() {
        let name = dev.name().unwrap_or_else(|_| "?".into());
        let cfg = dev.default_output_config();
        let info = match cfg {
            Ok(c) => format!("rate {}, ch {}", c.sample_rate(), c.channels()),
            Err(_) => "no config".into(),
        };
        let is_default = host
            .default_output_device()
            .map(|d| d.name().unwrap_or_default() == name)
            .unwrap_or(false);
        let mark = if is_default { " (по умолчанию)" } else { "" };
        eprintln!("  [{i}] {name} ({info}){mark}");
    }

    eprintln!();
    #[cfg(any(windows, target_os = "macos"))]
    eprintln!("Для захвата системного звука: --loopback default-output (или имя/индекс устройства)");
    #[cfg(all(not(windows), not(target_os = "macos")))]
    eprintln!("Для захвата системного звука: --loopback <имя или индекс input-устройства>");
}

fn main() -> Result<()> {
    let cli = Cli::parse();


    if cli.list_devices {
        list_devices();
        return Ok(());
    }

    let cfg: ClientConfig = if let Some(ref path) = cli.config {
        eprintln!("Конфиг: {path}");
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Не удалось прочитать конфиг {path}"))?;
        serde_json::from_str(&text)
            .with_context(|| format!("Ошибка парсинга конфига {path}"))?
    } else {
        load_config()
    };

    if cli.configure {
        let (ui_tx, ui_rx) = mpsc::channel();
        let (ws_out_tx, _ws_out_rx) = mpsc::channel();
        let running = std::sync::Arc::new(AtomicBool::new(true));
        let pending_end_session = std::sync::Arc::new(AtomicBool::new(false));
        let output = cli.output.as_ref().or(cfg.output.as_ref()).cloned().unwrap_or_else(|| "transcript.txt".into());
        let _ = run_tui(ui_rx, ui_tx, ws_out_tx, running, pending_end_session, &output, false, &cfg, false, true)?;
        return Ok(());
    }

    let mut cfg = cfg;
    let mut recording = true;
    loop {
        let cfg_for_tui = cfg.clone();
        let server = cli.server.as_ref().or(cfg.server.as_ref()).cloned().unwrap_or_else(|| "ws://localhost:9745".into());
        let dev_query = cli.device.as_ref().or(cfg.device.as_ref()).cloned();
        let loopback_query = cli.loopback.as_ref().or(cfg.loopback.as_ref()).cloned().or_else(|| {
            cfg.device2.as_ref().or(cli.device2.as_ref()).and_then(|q| {
                q.strip_prefix("out:").map(|s| s.to_string())
            })
        });
        let dev2_query = if loopback_query.is_none() {
            cli.device2.as_ref().or(cfg.device2.as_ref()).cloned().filter(|q| !q.starts_with("out:"))
        } else {
            None
        };
        let output = cli.output.as_ref().or(cfg.output.as_ref()).cloned().unwrap_or_else(|| "transcript.txt".into());

        let device = match &dev_query {
            Some(q) => resolve_device(q)?,
            None => default_device()?,
        };
        let dev_name = device.name().unwrap_or_else(|_| "?".into());
        eprintln!("Микрофон: {dev_name}");

        if let Some(ref q) = loopback_query {
            eprintln!("Системный звук: «{q}»");
        }

        let input_dev2 = match &dev2_query {
            Some(q) => {
                let d = resolve_device(q)?;
                eprintln!("Микрофон 2: {}", d.name().unwrap_or_else(|_| "?".into()));
                Some(d)
            }
            None => None,
        };

        let has_source2 = loopback_query.is_some() || input_dev2.is_some();

        let url = Url::parse(&server).context("Некорректный URL сервера")?;
        let ws_url = url.as_str().to_string();
        eprintln!("Подключение к {url} ...");

        let running = std::sync::Arc::new(AtomicBool::new(true));
        let running_ctrlc = running.clone();
        let _ = ctrlc::set_handler(move || {
            running_ctrlc.store(false, Ordering::Relaxed);
        });

        let (ui_tx, ui_rx) = mpsc::channel::<UiEvent>();
        let (ws_out_tx, ws_out_rx) = mpsc::channel::<WsOutgoing>();

        let ui_tx_ws = ui_tx.clone();
        let running_ws = running.clone();
        let src_count = if has_source2 { 2 } else { 1 };
        let client_id = cfg.client_id.clone();
        let pending_end_session = std::sync::Arc::new(AtomicBool::new(false));
        let pending_end_session_ws = pending_end_session.clone();
        let ws_thread = thread::spawn(move || {
            ws_io_thread(ws_url, ws_out_rx, ui_tx_ws, running_ws, src_count, client_id, pending_end_session_ws, recording);
        });

        let ws_out_audio = ws_out_tx.clone();
        let ui_tx_audio = ui_tx.clone();
        let running_audio = running.clone();
        let audio_thread = thread::spawn(move || {
            if let Err(e) = audio_capture(device, 0, ws_out_audio, ui_tx_audio, running_audio) {
                eprintln!("Audio error: {e}");
            }
        });

        let audio2_thread = if let Some(lb_query) = loopback_query {
            let ws_out2 = ws_out_tx.clone();
            let ui_tx2 = ui_tx.clone();
            let running2 = running.clone();
            Some(thread::spawn(move || {
                if let Err(e) = loopback_capture(lb_query, ws_out2, ui_tx2, running2) {
                    eprintln!("Loopback error: {e}");
                }
            }))
        } else if let Some(dev2) = input_dev2 {
            let ws_out2 = ws_out_tx.clone();
            let ui_tx2 = ui_tx.clone();
            let running2 = running.clone();
            Some(thread::spawn(move || {
                if let Err(e) = audio_capture(dev2, 1, ws_out2, ui_tx2, running2) {
                    eprintln!("Audio2 error: {e}");
                }
            }))
        } else {
            None
        };

        let (need_restart, new_recording) = run_tui(ui_rx, ui_tx, ws_out_tx, running.clone(), pending_end_session.clone(), &output, has_source2, &cfg_for_tui, recording, false)?;
        recording = new_recording;

        running.store(false, Ordering::Relaxed);
        let _ = audio_thread.join();
        if let Some(t) = audio2_thread {
            let _ = t.join();
        }
        // ws_thread checks `running` between reconnect attempts;
        // give it a moment to notice and exit, but don't block forever.
        let ws_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if ws_thread.is_finished() {
                let _ = ws_thread.join();
                break;
            }
            if Instant::now() >= ws_deadline {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }

        if !need_restart {
            break;
        }

        cfg = load_config();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyCode;

    #[test]
    fn test_physical_key_cyrillic() {
        assert_eq!(physical_key('й'), 'q');
        assert_eq!(physical_key('Й'), 'q');
        assert_eq!(physical_key('ц'), 'w');
        assert_eq!(physical_key('ы'), 's');
        assert_eq!(physical_key('о'), 'j');
        assert_eq!(physical_key('л'), 'k');
    }

    #[test]
    fn test_physical_key_latin() {
        assert_eq!(physical_key('q'), 'q');
        assert_eq!(physical_key('Q'), 'q');
        assert_eq!(physical_key('s'), 's');
        assert_eq!(physical_key('r'), 'r');
    }

    #[test]
    fn test_key_matches() {
        assert!(key_matches(KeyCode::Char('q'), 'q'));
        assert!(key_matches(KeyCode::Char('й'), 'q'));
        assert!(key_matches(KeyCode::Char('й'), 'q'));
        assert!(key_matches(KeyCode::Char('s'), 's'));
        assert!(key_matches(KeyCode::Char('ы'), 's'));
        assert!(!key_matches(KeyCode::Char('a'), 'q'));
        assert!(!key_matches(KeyCode::Char('a'), 's'));
    }

    #[test]
    fn test_to_mono_stereo() {
        let stereo = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mono = to_mono(&stereo, 2);
        assert_eq!(mono, vec![1.5, 3.5, 5.5]);
    }

    #[test]
    fn test_to_mono_already_mono() {
        let data = vec![1.0, 2.0, 3.0];
        let mono = to_mono(&data, 1);
        assert_eq!(mono, data);
    }

    #[test]
    fn test_resample_same_rate() {
        let src = vec![1.0, 2.0, 3.0];
        let out = resample(&src, SAMPLE_RATE);
        assert_eq!(out, src);
    }

    #[test]
    fn test_resample_downsample() {
        let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = resample(&src, 48000);
        assert_eq!(out.len(), (100.0_f64 * 16000.0 / 48000.0).ceil() as usize);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_client_config_parse() {
        let json = r#"{"server":"ws://localhost:9745","device":"0","loopback":"Razer","output":"out.txt"}"#;
        let cfg: ClientConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.server.as_deref(), Some("ws://localhost:9745"));
        assert_eq!(cfg.device.as_deref(), Some("0"));
        assert_eq!(cfg.loopback.as_deref(), Some("Razer"));
        assert_eq!(cfg.output.as_deref(), Some("out.txt"));
    }

    #[test]
    fn test_client_config_device2_out_backward_compat() {
        let json = r#"{"device2":"out:Razer"}"#;
        let cfg: ClientConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.device2.as_deref(), Some("out:Razer"));
    }

    #[test]
    fn test_client_config_defaults() {
        let json = r#"{}"#;
        let cfg: ClientConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.server.is_none());
        assert!(cfg.device.is_none());
        assert!(cfg.device2.is_none());
        assert!(cfg.output.is_none());
    }

    #[test]
    fn test_server_message_transcript() {
        let json = r#"{"type":"transcript","text":"привет","source":0}"#;
        let msg: ServerMessage = serde_json::from_str(json).unwrap();
        match &msg {
            ServerMessage::Transcript { text, source, variants, .. } => {
                assert_eq!(text, "привет");
                assert_eq!(*source, Some(0));
                assert!(variants.is_none());
            }
            _ => panic!("expected Transcript"),
        }
    }

    #[test]
    fn test_server_message_debug() {
        let json = r#"{"type":"debug","text":"test"}"#;
        let msg: ServerMessage = serde_json::from_str(json).unwrap();
        match &msg {
            ServerMessage::Debug { text } => assert_eq!(text, "test"),
            _ => panic!("expected Debug"),
        }
    }

    #[test]
    fn test_server_message_error() {
        let json = r#"{"type":"error","text":"err"}"#;
        let msg: ServerMessage = serde_json::from_str(json).unwrap();
        match &msg {
            ServerMessage::Error { text } => assert_eq!(text, "err"),
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_export_trimmed_transcript() {
        let dir = std::env::temp_dir().join("localvox_export_test");
        let _ = std::fs::create_dir_all(&dir);
        let transcript = dir.join("transcript.txt");
        let export = dir.join("meeting_export.txt");
        let content = r#"=== SEGMENT src1_000001 | 0.0s–2.0s | sys | 2026-03-12 23:05:04 ===
  whisper: hello
  gigaam: hello
=== END SEGMENT src1_000001 ===

=== SEGMENT src1_000002 | 2.0s–4.0s | sys | 2026-03-12 23:05:06 ===
  whisper: world
=== END SEGMENT src1_000002 ===

=== SEGMENT src1_000003 | 4.0s–6.0s | sys | 2026-03-12 23:05:08 ===
  whisper: end
=== END SEGMENT src1_000003 ===
"#;
        std::fs::write(&transcript, content).unwrap();
        let n = export_trimmed_transcript(
            transcript.to_str().unwrap(),
            &export,
            &Some("src1_000001".into()),
            &Some("src1_000002".into()),
        ).unwrap();
        assert_eq!(n, 2);
        let out = std::fs::read_to_string(&export).unwrap();
        assert!(out.contains("hello"));
        assert!(out.contains("world"));
        assert!(!out.contains("end"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
