//! TUI: configure screen and main transcript view.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use chrono::Local;

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Terminal;

use crate::audio::{
    collect_input_devices, format_device_display, key_matches, list_output_device_names,
};
use crate::config::{get_config_save_path, ClientConfig};
use crate::summarize::spawn_summarize_ollama;
use crate::transcript::TranscriptStore;
use crate::types::{
    AudioSource, ServerMessage, SettingsState, StatusData, TuiMode, UiEvent, WsOutgoing,
    MAX_DEBUG_LINES,
};

/// Returns (need_restart, recording) — need_restart if settings were saved, recording state to preserve.
/// When configure_only is true, starts in Settings mode (for --configure CLI); no ws/audio threads.
#[allow(clippy::too_many_arguments)]
pub fn run_tui(
    ui_rx: mpsc::Receiver<UiEvent>,
    ui_tx: mpsc::Sender<UiEvent>,
    ws_tx: mpsc::Sender<WsOutgoing>,
    running: std::sync::Arc<AtomicBool>,
    pending_end_session: std::sync::Arc<AtomicBool>,
    output_path: &str,
    _has_source2: bool,
    cfg: &ClientConfig,
    initial_recording: bool,
    configure_only: bool,
) -> Result<(bool, bool)> {
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut store = TranscriptStore::load(output_path)?;
    let mut debug_lines: Vec<String> = Vec::new();
    let mut audio_level: f32 = 0.0;
    let mut audio_level2: f32 = 0.0;
    let mut _status = StatusData::default();
    let mut recording = initial_recording;
    let mut connected = false;
    let mut last_activity = Instant::now();
    let mut last_audio_dir_mb: Option<f64> = None;
    let mut last_llm_queue: Option<u32> = None;
    let mut last_disconnect_reason: Option<String> = None;

    let mut mode = if configure_only {
        TuiMode::Settings
    } else {
        TuiMode::Main
    };
    let mut settings_state: Option<SettingsState> = if configure_only {
        let input_devices: Vec<_> = collect_input_devices()
            .into_iter()
            .enumerate()
            .map(|(i, (dev, n))| (i, format_device_display(&dev, &n, "")))
            .collect();
        let output_devices = list_output_device_names();
        let input_selected = cfg
            .device
            .as_ref()
            .and_then(|d| d.parse::<usize>().ok())
            .filter(|&i| i < input_devices.len())
            .unwrap_or(0);
        let output_selected = cfg
            .loopback
            .as_deref()
            .or_else(|| cfg.device2.as_ref().and_then(|d| d.strip_prefix("out:")))
            .map_or(0, |s| {
                output_devices
                    .iter()
                    .position(|(_, n)| n.eq_ignore_ascii_case(s))
                    .map(|i| i + 1)
                    .unwrap_or(0)
            });
        let mut input_state = ListState::default();
        input_state.select(Some(
            input_selected.min(input_devices.len().saturating_sub(1)),
        ));
        let mut output_state = ListState::default();
        output_state.select(Some(output_selected.min(output_devices.len())));
        Some(SettingsState {
            input_devices,
            output_devices,
            input_state,
            output_state,
            focus: 0,
            saved_msg: None,
            saved_at: None,
        })
    } else {
        None
    };

    let mut focus: u8 = 0;
    let mut transcript_scroll: Option<u16> = None;
    let mut debug_scroll: Option<u16> = None;
    let mut need_restart = false;
    let mut restart_at: Option<Instant> = None;
    let mut export_msg: Option<(String, Instant)> = None;
    let mut _last_t_inner_w: usize = 80;
    let mut _last_t_scroll_y: u16 = 0;
    let mut _last_t_visible: u16 = 0;

    let mut last_draw = Instant::now();

    loop {
        if let Some((_, at)) = &export_msg {
            if at.elapsed() >= Duration::from_secs(3) {
                export_msg = None;
            }
        }
        if event::poll(Duration::from_millis(30))? {
            match event::read()? {
                Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
                    if matches!(mode, TuiMode::Settings) {
                        let st = settings_state.as_mut().unwrap();
                        match key.code {
                            KeyCode::Esc => {
                                if configure_only {
                                    running.store(false, Ordering::Relaxed);
                                    break;
                                }
                                mode = TuiMode::Main;
                            }
                            code if key_matches(code, 's') => {
                                let input_idx = st.input_state.selected().unwrap_or(0);
                                let device = st
                                    .input_devices
                                    .get(input_idx)
                                    .map(|(i, _)| format!("{i}"))
                                    .or_else(|| {
                                        st.input_devices.first().map(|(i, _)| format!("{i}"))
                                    })
                                    .unwrap_or_default();
                                let loopback = st.output_state.selected().and_then(|idx| {
                                    if idx == 0 {
                                        None
                                    } else {
                                        st.output_devices.get(idx - 1).map(|(_, n)| n.clone())
                                    }
                                });
                                let save_cfg = ClientConfig {
                                    server: cfg
                                        .server
                                        .clone()
                                        .or_else(|| Some("ws://localhost:9745".into())),
                                    device: if device.is_empty() {
                                        None
                                    } else {
                                        Some(device)
                                    },
                                    loopback,
                                    device2: None,
                                    output: cfg
                                        .output
                                        .clone()
                                        .or_else(|| Some("transcript.txt".into())),
                                    export_dir: cfg.export_dir.clone(),
                                    summarize_enabled: cfg.summarize_enabled,
                                    summarize_backend: cfg.summarize_backend.clone(),
                                    summarize_url: cfg.summarize_url.clone(),
                                    summarize_model: cfg.summarize_model.clone(),
                                    summarize_prompt: cfg.summarize_prompt.clone(),
                                    client_id: cfg.client_id.clone(),
                                };
                                let path = get_config_save_path();
                                if let Ok(text) = serde_json::to_string_pretty(&save_cfg) {
                                    st.saved_msg = Some(match std::fs::write(&path, text) {
                                        Ok(_) => {
                                            need_restart = true;
                                            restart_at = Some(Instant::now());
                                            "Сохранено. Применяется…".to_string()
                                        }
                                        Err(e) => format!("Ошибка: {e}"),
                                    });
                                    st.saved_at = Some(Instant::now());
                                }
                            }
                            KeyCode::Tab | KeyCode::BackTab => st.focus = 1 - st.focus,
                            code if code == KeyCode::Up || key_matches(code, 'k') => {
                                let state = if st.focus == 0 {
                                    &mut st.input_state
                                } else {
                                    &mut st.output_state
                                };
                                let len = if st.focus == 0 {
                                    st.input_devices.len()
                                } else {
                                    st.output_devices.len() + 1
                                };
                                let i = state.selected().unwrap_or(0).saturating_sub(1);
                                state.select(Some(i.min(len.saturating_sub(1))));
                            }
                            code if code == KeyCode::Down || key_matches(code, 'j') => {
                                let state = if st.focus == 0 {
                                    &mut st.input_state
                                } else {
                                    &mut st.output_state
                                };
                                let len = if st.focus == 0 {
                                    st.input_devices.len()
                                } else {
                                    st.output_devices.len() + 1
                                };
                                let i = state.selected().unwrap_or(0).saturating_add(1);
                                state.select(Some(i.min(len.saturating_sub(1))));
                            }
                            _ => {}
                        }
                    } else {
                        match key.code {
                            KeyCode::F(2) => {
                                if settings_state.is_none() {
                                    let input_devices: Vec<_> = collect_input_devices()
                                        .into_iter()
                                        .enumerate()
                                        .map(|(i, (dev, n))| {
                                            (i, format_device_display(&dev, &n, ""))
                                        })
                                        .collect();
                                    let output_devices = list_output_device_names();
                                    let input_selected = cfg
                                        .device
                                        .as_ref()
                                        .and_then(|d| d.parse::<usize>().ok())
                                        .filter(|&i| i < input_devices.len())
                                        .unwrap_or(0);
                                    let output_selected = cfg
                                        .loopback
                                        .as_deref()
                                        .or_else(|| {
                                            cfg.device2
                                                .as_ref()
                                                .and_then(|d| d.strip_prefix("out:"))
                                        })
                                        .map_or(0, |s| {
                                            output_devices
                                                .iter()
                                                .position(|(_, n)| n.eq_ignore_ascii_case(s))
                                                .map(|i| i + 1)
                                                .unwrap_or(0)
                                        });
                                    let mut input_state = ListState::default();
                                    input_state.select(Some(
                                        input_selected.min(input_devices.len().saturating_sub(1)),
                                    ));
                                    let mut output_state = ListState::default();
                                    output_state
                                        .select(Some(output_selected.min(output_devices.len())));
                                    settings_state = Some(SettingsState {
                                        input_devices,
                                        output_devices,
                                        input_state,
                                        output_state,
                                        focus: 0,
                                        saved_msg: None,
                                        saved_at: None,
                                    });
                                }
                                mode = TuiMode::Settings;
                            }
                            code if key_matches(code, 'q') || code == KeyCode::Esc => {
                                running.store(false, Ordering::Relaxed);
                                break;
                            }
                            code if key.modifiers.contains(KeyModifiers::CONTROL)
                                && key_matches(code, 'c') =>
                            {
                                running.store(false, Ordering::Relaxed);
                                break;
                            }
                            code if key_matches(code, 'r') => {
                                recording = !recording;
                                let msg =
                                    serde_json::json!({"type": "recording", "enabled": recording});
                                let _ = ws_tx.send(WsOutgoing::Text(msg.to_string()));
                            }
                            code if key_matches(code, 'x') => {
                                pending_end_session.store(true, Ordering::Relaxed);
                                let _ = ws_tx
                                    .send(WsOutgoing::Text(r#"{"type":"end_session"}"#.into()));
                                export_msg = Some(("Завершение сессии…".into(), Instant::now()));
                            }
                            KeyCode::Tab | KeyCode::BackTab => focus = 1 - focus,
                            code if code == KeyCode::Up || key_matches(code, 'k') => {
                                let scroll = if focus == 0 {
                                    &mut transcript_scroll
                                } else {
                                    &mut debug_scroll
                                };
                                *scroll = Some(scroll.unwrap_or(u16::MAX).saturating_sub(1));
                            }
                            code if code == KeyCode::Down || key_matches(code, 'j') => {
                                let scroll = if focus == 0 {
                                    &mut transcript_scroll
                                } else {
                                    &mut debug_scroll
                                };
                                *scroll = Some(scroll.unwrap_or(0).saturating_add(1));
                            }
                            KeyCode::End => {
                                if focus == 0 {
                                    transcript_scroll = None;
                                } else {
                                    debug_scroll = None;
                                }
                            }
                            KeyCode::Home => {
                                if focus == 0 {
                                    transcript_scroll = Some(0);
                                } else {
                                    debug_scroll = Some(0);
                                }
                            }
                            code if focus == 0 && key_matches(code, 'e') => {
                                let out_dir: std::path::PathBuf = cfg
                                    .export_dir
                                    .as_deref()
                                    .map(std::path::PathBuf::from)
                                    .unwrap_or_else(|| {
                                        std::path::Path::new(output_path)
                                            .parent()
                                            .unwrap_or(std::path::Path::new("."))
                                            .into()
                                    });
                                if let Err(e) = std::fs::create_dir_all(&out_dir) {
                                    export_msg = Some((
                                        format!("Ошибка папки экспорта: {}", e),
                                        Instant::now(),
                                    ));
                                } else {
                                    match store.export(&out_dir) {
                                        Ok((path, 0)) => {
                                            let _ = std::fs::remove_file(&path);
                                            export_msg = Some((
                                                "Нет данных для экспорта".into(),
                                                Instant::now(),
                                            ));
                                        }
                                        Ok((path, n)) => {
                                            let summarize_on = cfg.summarize_enabled == Some(true);
                                            let msg = if summarize_on {
                                                format!(
                                                    "Экспорт: {} строк → {}. Суммаризация…",
                                                    n,
                                                    path.display()
                                                )
                                            } else {
                                                format!("Экспорт: {} строк → {}", n, path.display())
                                            };
                                            export_msg = Some((msg, Instant::now()));
                                            if summarize_on {
                                                let url = cfg
                                                    .summarize_url
                                                    .as_deref()
                                                    .unwrap_or("http://localhost:11434");
                                                let model = cfg
                                                    .summarize_model
                                                    .as_deref()
                                                    .unwrap_or("qwen3.5:9b");
                                                let prompt = cfg.summarize_prompt.as_deref().unwrap_or(
                "Ты — суммаризатор встреч. Транскрипт ниже — запись разговора (mic = микрофон, sys = системный звук).\n\nСделай структурированную суммаризацию:\n\n1. Заголовок: «Суммаризация: [тема встречи]»\n2. УЧАСТНИКИ — «— Имя (mic/sys) — роль». Если имён нет — «— участник (mic)», «— участник (sys)»\n3. СТАТУСЫ ПО ЗАДАЧАМ — по участникам: что делает, вопросы. Пропусти «что сделал вчера», «блокеры», если нет информации. Не пиши «Не указано»\n4. РАЗБОР (если есть дискуссии) — Контекст, Суть, Решение\n5. ПРОЧЕЕ — организационные вопросы\n\nТолько русский. Только факты из транскрипта.\n\n---\n\n{text}");
                                                let backend = cfg
                                                    .summarize_backend
                                                    .as_deref()
                                                    .unwrap_or("ollama");
                                                if backend.eq_ignore_ascii_case("ollama") {
                                                    spawn_summarize_ollama(
                                                        path,
                                                        url,
                                                        model,
                                                        prompt,
                                                        ui_tx.clone(),
                                                    );
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            export_msg = Some((
                                                format!("Ошибка экспорта: {}", e),
                                                Instant::now(),
                                            ))
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    if matches!(mode, TuiMode::Settings) {
                        let st = settings_state.as_mut().unwrap();
                        if let MouseEventKind::ScrollUp = mouse.kind {
                            let state = if st.focus == 0 {
                                &mut st.input_state
                            } else {
                                &mut st.output_state
                            };
                            let len = if st.focus == 0 {
                                st.input_devices.len()
                            } else {
                                st.output_devices.len() + 1
                            };
                            let i = state.selected().unwrap_or(0).saturating_sub(1);
                            state.select(Some(i.min(len.saturating_sub(1))));
                        } else if let MouseEventKind::ScrollDown = mouse.kind {
                            let state = if st.focus == 0 {
                                &mut st.input_state
                            } else {
                                &mut st.output_state
                            };
                            let len = if st.focus == 0 {
                                st.input_devices.len()
                            } else {
                                st.output_devices.len() + 1
                            };
                            let i = state.selected().unwrap_or(0).saturating_add(1);
                            state.select(Some(i.min(len.saturating_sub(1))));
                        }
                    } else if let MouseEventKind::ScrollUp = mouse.kind {
                        let scroll = if focus == 0 {
                            &mut transcript_scroll
                        } else {
                            &mut debug_scroll
                        };
                        *scroll = Some(scroll.unwrap_or(u16::MAX).saturating_sub(1));
                    } else if let MouseEventKind::ScrollDown = mouse.kind {
                        let scroll = if focus == 0 {
                            &mut transcript_scroll
                        } else {
                            &mut debug_scroll
                        };
                        *scroll = Some(scroll.unwrap_or(0).saturating_add(1));
                    }
                }
                _ => {}
            }
        }

        while let Ok(ev) = ui_rx.try_recv() {
            match ev {
                UiEvent::Server(ServerMessage::Done) => {
                    match store.clear() {
                        Ok(_) => {
                            let ts = Local::now().format("%H:%M:%S");
                            debug_lines.push(format!("{ts} [session] transcript cleared"));
                        }
                        Err(e) => {
                            let ts = Local::now().format("%H:%M:%S");
                            debug_lines.push(format!("{ts} [session] clear error: {e}"));
                        }
                    }
                    transcript_scroll = None;
                    export_msg =
                        Some(("Сессия завершена. Переподключение…".into(), Instant::now()));
                }
                UiEvent::Server(ServerMessage::Transcript {
                    text,
                    source,
                    start_sec,
                    end_sec,
                    seg_id,
                    variants,
                    ..
                }) => {
                    let seg_key = seg_id.as_deref().unwrap_or("");
                    if store.contains_seg_id(seg_key) {
                    } else {
                        last_activity = Instant::now();
                        let write_to_file = recording;
                        if let Some(ref v) = variants {
                            store.append_variants(
                                &text,
                                source,
                                seg_id.as_deref(),
                                v,
                                start_sec,
                                end_sec,
                                write_to_file,
                            );
                        } else {
                            store.append_plain(&text, source, seg_id.as_deref(), write_to_file);
                        }
                        transcript_scroll = None;
                    }
                }
                UiEvent::Server(ServerMessage::Status(s)) => {
                    last_activity = Instant::now();
                    if let Some(r) = s.recording {
                        recording = r;
                    }
                    if let Some(mb) = s.audio_dir_size_mb {
                        last_audio_dir_mb = Some(mb);
                    }
                    if let Some(n) = s.llm_queue {
                        last_llm_queue = Some(n);
                    }
                    _status = s;
                }
                UiEvent::Server(ServerMessage::Debug { text }) => {
                    let ts = Local::now().format("%H:%M:%S");
                    debug_lines.push(format!("{ts} {text}"));
                    if debug_lines.len() > MAX_DEBUG_LINES {
                        debug_lines.drain(..debug_lines.len() - MAX_DEBUG_LINES);
                    }
                }
                UiEvent::Server(ServerMessage::Error { text }) => {
                    let ts = Local::now().format("%H:%M:%S");
                    debug_lines.push(format!("{ts} [ERROR] {text}"));
                }
                UiEvent::Server(ServerMessage::Session { .. }) => {
                    // Handled in ws.rs; logged as debug there
                }
                UiEvent::AudioLevel { source: 0, level } => audio_level = level,
                UiEvent::AudioLevel { source: _, level } => audio_level2 = level,
                UiEvent::Connected => {
                    connected = true;
                    let ts = Local::now().format("%H:%M:%S");
                    debug_lines.push(format!("{ts} [ws] подключено"));
                }
                UiEvent::Disconnected { reason } => {
                    connected = false;
                    last_disconnect_reason = Some(reason.clone());
                    let ts = Local::now().format("%H:%M:%S");
                    debug_lines.push(format!("{ts} [ws] отключено: {reason}"));
                }
                UiEvent::SummarizeDone { msg } => {
                    export_msg = Some((msg, Instant::now()));
                }
                UiEvent::Debug { text } => {
                    let ts = Local::now().format("%H:%M:%S");
                    debug_lines.push(format!("{ts} {text}"));
                    if debug_lines.len() > MAX_DEBUG_LINES {
                        debug_lines.drain(..debug_lines.len() - MAX_DEBUG_LINES);
                    }
                }
                UiEvent::Quit => {
                    running.store(false, Ordering::Relaxed);
                    break;
                }
            }
        }

        if !running.load(Ordering::Relaxed) {
            break;
        }

        if last_draw.elapsed() < Duration::from_millis(33) {
            continue;
        }
        last_draw = Instant::now();

        if let (TuiMode::Settings, Some(ref mut st)) = (&mode, &mut settings_state) {
            if let Some(saved_at) = st.saved_at {
                if saved_at.elapsed() >= Duration::from_secs(2) {
                    st.saved_msg = None;
                    st.saved_at = None;
                }
            }
        }
        if need_restart && restart_at.is_some_and(|t| t.elapsed() >= Duration::from_secs(2)) {
            break;
        }

        terminal.draw(|f| {
            if matches!(mode, TuiMode::Settings) {
                if let Some(ref mut st) = settings_state {
                    let area = f.area();
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Min(5), Constraint::Length(1)])
                        .split(area);
                    let main_chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(chunks[0]);

                    let input_items: Vec<ListItem> = st.input_devices.iter()
                        .map(|(i, n)| ListItem::new(format!("[{i}] {n}"))).collect();
                    let input_list = List::new(input_items)
                        .block(Block::default().borders(Borders::ALL)
                            .title(if st.focus == 0 { "Микрофон ◄" } else { "Микрофон" })
                            .border_style(Style::default().fg(if st.focus == 0 { Color::Yellow } else { Color::Reset })))
                        .highlight_style(Style::default().bg(Color::DarkGray))
                        .highlight_symbol(">> ");
                    f.render_stateful_widget(input_list, main_chunks[0], &mut st.input_state);

                    let output_items: Vec<ListItem> = std::iter::once(ListItem::new("— нет —"))
                        .chain(st.output_devices.iter().map(|(i, n)| ListItem::new(format!("[{i}] {n}"))))
                        .collect();
                    let output_list = List::new(output_items)
                        .block(Block::default().borders(Borders::ALL)
                            .title(if st.focus == 1 { "Loopback ◄" } else { "Loopback" })
                            .border_style(Style::default().fg(if st.focus == 1 { Color::Yellow } else { Color::Reset })))
                        .highlight_style(Style::default().bg(Color::DarkGray))
                        .highlight_symbol(">> ");
                    f.render_stateful_widget(output_list, main_chunks[1], &mut st.output_state);

                    f.render_widget(
                        Paragraph::new("Tab — переключить | ↑↓ / колёсико — выбор | S — сохранить | Esc — назад")
                            .style(Style::default().fg(Color::DarkGray)),
                        chunks[1],
                    );

                    if let (Some(ref msg), Some(saved_at)) = (&st.saved_msg, st.saved_at) {
                        if saved_at.elapsed() < Duration::from_secs(2) {
                            let a = f.area();
                            let popup = ratatui::layout::Rect {
                                x: a.width / 4, y: a.height - 2, width: a.width / 2, height: 1,
                            };
                            f.render_widget(
                                Paragraph::new(msg.as_str())
                                    .block(Block::default().borders(Borders::ALL).title("Сохранено"))
                                    .style(Style::default().fg(Color::Green)),
                                popup,
                            );
                        }
                    }
                }
                return;
            }

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(4),
                    Constraint::Min(5),
                    Constraint::Length(8),
                ])
                .split(f.area());

            let rec_icon = if recording { "● REC" } else { "○ STOP" };
            let rec_color = if recording { Color::Red } else { Color::DarkGray };

            let conn_icon = if connected { "●" } else { "○" };
            let conn_color = if connected { Color::Green } else { Color::Red };
            let conn_text = if connected {
                "подключен".to_string()
            } else {
                last_disconnect_reason
                    .as_ref()
                    .map(|r| format!("отключен ({})", r))
                    .unwrap_or_else(|| "отключен".to_string())
            };

            let pending_str = last_audio_dir_mb
                .map(|mb| format!("{:.1} MB", mb))
                .unwrap_or_else(|| "—".to_string());

            let llm_str = last_llm_queue
                .map(|n| format!("{}", n))
                .unwrap_or_else(|| "—".to_string());

            let level_bars = (audio_level * 8.0) as usize;
            let level_str: String =
                "▮".repeat(level_bars.min(8)) + &"▯".repeat(8_usize.saturating_sub(level_bars));
            let level2_bars = (audio_level2 * 8.0) as usize;
            let level2_str: String =
                "▮".repeat(level2_bars.min(8)) + &"▯".repeat(8_usize.saturating_sub(level2_bars));

            let last_sec = last_activity.elapsed().as_secs();
            let last_str = format!("{} с назад", last_sec);

            let line1 = Line::from(vec![
                Span::styled(rec_icon, Style::default().fg(rec_color)),
                Span::raw("  "),
                Span::styled(conn_icon, Style::default().fg(conn_color)),
                Span::raw(format!(" {}  ", conn_text)),
                Span::raw("очередь: "),
                Span::styled(&pending_str, Style::default().fg(Color::Yellow)),
                Span::raw("  llm: "),
                Span::styled(&llm_str, Style::default().fg(Color::Yellow)),
                Span::raw("  mic "),
                Span::styled(&level_str, Style::default().fg(Color::Green)),
                Span::raw("  sys "),
                Span::styled(&level2_str, Style::default().fg(Color::Cyan)),
                Span::raw("  последняя: "),
                Span::styled(&last_str, Style::default().fg(Color::Magenta)),
                Span::raw("  "),
                Span::styled("[r] [x] [F2] [q]", Style::default().fg(Color::DarkGray)),
                Span::raw("  "),
                Span::styled("[e] экспорт+суммаризация", Style::default().fg(Color::DarkGray)),
            ]);

            let mut status_lines = vec![line1];
            if let Some((ref msg, at)) = export_msg {
                if at.elapsed() < Duration::from_secs(3) {
                    status_lines.push(Line::from(Span::styled(msg.as_str(), Style::default().fg(Color::Green))));
                }
            }

            let status_widget = Paragraph::new(status_lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Статус")
                        .border_style(Style::default().fg(if connected { Color::Reset } else { Color::Red })),
                );
            f.render_widget(status_widget, chunks[0]);

            let t_inner_w = chunks[1].width.saturating_sub(2).max(1) as usize;
            _last_t_inner_w = t_inner_w;
            let t_total: u16 = store.lines
                .iter()
                .map(|l| {
                    let prefix_len = match l.source {
                        Some(AudioSource::Mic) | Some(AudioSource::Sys) => 4,
                        _ => 0,
                    };
                    let w = prefix_len + l.text.chars().count();
                    if w == 0 { 1u16 } else { w.div_ceil(t_inner_w) as u16 }
                })
                .sum();
            let t_visible = chunks[1].height.saturating_sub(2);
            _last_t_visible = t_visible;
            let t_max = t_total.saturating_sub(t_visible);
            let t_scroll_y = match transcript_scroll {
                None => t_max,
                Some(v) => v.min(t_max),
            };
            _last_t_scroll_y = t_scroll_y;
            if let Some(ref mut v) = transcript_scroll {
                *v = (*v).min(t_max);
            } else {
                transcript_scroll = Some(t_max);
            }

            let mut row = 0u16;
            let mut top_line_idx = 0;
            let mut bottom_line_idx = store.lines.len().saturating_sub(1);
            let bottom_row = t_scroll_y.saturating_add(t_visible).saturating_sub(1);
            let mut top_set = false;
            for (i, l) in store.lines.iter().enumerate() {
                let pl = match l.source { Some(AudioSource::Mic)|Some(AudioSource::Sys) => 4, _ => 0 };
                let w = pl + l.text.chars().count();
                let h = if w == 0 { 1 } else { w.div_ceil(t_inner_w) as u16 };
                if !top_set && row + h > t_scroll_y {
                    top_line_idx = i;
                    top_set = true;
                }
                if row + h > bottom_row {
                    bottom_line_idx = i;
                    break;
                }
                row += h;
                bottom_line_idx = i;
            }

            let t_title = "Транскрипция";
            let t_border = if focus == 0 { Color::Yellow } else { Color::Reset };
            let t_lines: Vec<Line> = store.lines
                .iter()
                .enumerate()
                .map(|(i, line)| {
                    let (prefix, color) = match line.source {
                        Some(AudioSource::Mic) => ("mic ", Color::Green),
                        Some(AudioSource::Sys) => ("sys ", Color::Cyan),
                        _ => ("", Color::White),
                    };
                    let mut spans = vec![];
                    let is_top = focus == 0 && i == top_line_idx;
                    let is_bottom = focus == 0 && i == bottom_line_idx;
                    if is_top {
                        spans.push(Span::styled("► ", Style::default().fg(Color::Yellow)));
                    } else if is_bottom {
                        spans.push(Span::styled("▼ ", Style::default().fg(Color::Yellow)));
                    }
                    if !prefix.is_empty() {
                        spans.push(Span::styled(prefix, Style::default().fg(color).add_modifier(Modifier::DIM)));
                    }
                    spans.push(Span::styled(line.text.as_str(), Style::default().fg(color)));
                    Line::from(spans)
                })
                .collect();
            let transcript_widget = Paragraph::new(t_lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(t_title)
                        .border_style(Style::default().fg(t_border)),
                )
                .wrap(Wrap { trim: false })
                .scroll((t_scroll_y, 0));
            f.render_widget(transcript_widget, chunks[1]);

            let d_inner_w = chunks[2].width.saturating_sub(2).max(1) as usize;
            let d_total: u16 = debug_lines
                .iter()
                .map(|l| {
                    let w = l.chars().count();
                    if w == 0 { 1u16 } else { w.div_ceil(d_inner_w) as u16 }
                })
                .sum();
            let d_visible = chunks[2].height.saturating_sub(2);
            let d_max = d_total.saturating_sub(d_visible);
            let d_scroll_y = match debug_scroll {
                None => d_max,
                Some(v) => {
                    let clamped = v.min(d_max);
                    if clamped >= d_max {
                        debug_scroll = None;
                    }
                    clamped
                }
            };

            let d_title = if focus == 1 { "Debug ◄" } else { "Debug" };
            let d_border = if focus == 1 { Color::Yellow } else { Color::Reset };
            let d_lines: Vec<Line> = debug_lines.iter().map(|l| Line::raw(l.as_str())).collect();
            let debug_widget = Paragraph::new(d_lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(d_title)
                        .border_style(Style::default().fg(d_border)),
                )
                .wrap(Wrap { trim: false })
                .scroll((d_scroll_y, 0));
            f.render_widget(debug_widget, chunks[2]);
        })?;
    }

    terminal::disable_raw_mode()?;
    execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen)?;
    Ok((need_restart, recording))
}
