use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseEventKind};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Terminal;
use serde::{Deserialize, Serialize};
use tungstenite::Message;
use url::Url;

const SAMPLE_RATE: u32 = 16000;
const CHUNK_FRAMES: usize = 512;

/// Физическая клавиша (не зависит от раскладки). Русская ЙЦУКЕН → Latin QWERTY.
fn physical_key(c: char) -> char {
    match c {
        'q' | 'Q' | 'й' | 'Й' => 'q',
        'w' | 'W' | 'ц' | 'Ц' => 'w',
        'e' | 'E' | 'у' | 'У' => 'e',
        'r' | 'R' | 'к' | 'К' => 'r',
        't' | 'T' | 'е' | 'Е' => 't',
        'y' | 'Y' | 'н' | 'Н' => 'y',
        'u' | 'U' | 'г' | 'Г' => 'u',
        'i' | 'I' | 'ш' | 'Ш' => 'i',
        'o' | 'O' | 'щ' | 'Щ' => 'o',
        'p' | 'P' | 'з' | 'З' => 'p',
        'a' | 'A' | 'ф' | 'Ф' => 'a',
        's' | 'S' | 'ы' | 'Ы' => 's',
        'd' | 'D' | 'в' | 'В' => 'd',
        'f' | 'F' | 'а' | 'А' => 'f',
        'g' | 'G' | 'п' | 'П' => 'g',
        'h' | 'H' | 'р' | 'Р' => 'h',
        'j' | 'J' | 'о' | 'О' => 'j',
        'k' | 'K' | 'л' | 'Л' => 'k',
        'l' | 'L' | 'д' | 'Д' => 'l',
        'z' | 'Z' | 'я' | 'Я' => 'z',
        'x' | 'X' | 'ч' | 'Ч' => 'x',
        'c' | 'C' | 'с' | 'С' => 'c',
        'v' | 'V' | 'м' | 'М' => 'v',
        'b' | 'B' | 'и' | 'И' => 'b',
        'n' | 'N' | 'т' | 'Т' => 'n',
        'm' | 'M' | 'ь' | 'Ь' => 'm',
        _ => c.to_ascii_lowercase(),
    }
}

fn key_matches(key: KeyCode, expected: char) -> bool {
    match key {
        KeyCode::Char(c) => physical_key(c) == expected,
        _ => false,
    }
}

// ── Config file ──────────────────────────────────────────────────────

#[derive(Deserialize, Serialize, Debug, Default, Clone)]
struct ClientConfig {
    #[serde(default)]
    server: Option<String>,
    #[serde(default)]
    device: Option<String>,
    /// Системный звук (loopback): устройство вывода по имени или индексу
    #[serde(default)]
    loopback: Option<String>,
    /// Второй микрофон (по имени или индексу). Игнорируется если задан loopback.
    #[serde(default)]
    device2: Option<String>,
    #[serde(default)]
    output: Option<String>,
    /// Stable client/session ID for server reconnection.
    #[serde(default)]
    client_id: Option<String>,
}

fn find_config_path() -> Option<PathBuf> {
    let name = "client-config.json";
    let mut candidates: Vec<Option<PathBuf>> = vec![
        std::env::current_dir().ok().map(|p| p.join(name)),
        std::env::current_dir().ok().and_then(|p| p.parent().map(|pp| pp.join(name))),
        std::env::current_exe().ok().and_then(|p| p.parent().map(|pp| pp.join(name))),
    ];
    candidates.push(
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .map(|base| base.join("live-transcribe").join(name)),
    );
    candidates.into_iter().flatten().find(|p| p.is_file())
}

fn load_config() -> ClientConfig {
    match find_config_path() {
        Some(path) => {
            eprintln!("Конфиг: {}", path.display());
            match std::fs::read_to_string(&path) {
                Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                    eprintln!("Ошибка парсинга конфига: {e}");
                    ClientConfig::default()
                }),
                Err(e) => {
                    eprintln!("Не удалось прочитать конфиг: {e}");
                    ClientConfig::default()
                }
            }
        }
        None => ClientConfig::default(),
    }
}

/// Путь для сохранения конфига. Предпочитает тот же файл, из которого загружаем.
fn get_config_save_path() -> PathBuf {
    if let Some(path) = find_config_path() {
        return path;
    }
    let name = "client-config.json";
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.is_dir() {
            return cwd.join(name);
        }
    }
    if let Some(base) = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    {
        let dir = base.join("live-transcribe");
        let _ = std::fs::create_dir_all(&dir);
        return dir.join(name);
    }
    PathBuf::from(name)
}

fn list_output_device_names() -> Vec<(usize, String)> {
    let enumerator = match wasapi::DeviceEnumerator::new() {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    let collection = match enumerator.get_device_collection(&wasapi::Direction::Render) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    collection
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| {
            let dev = r.ok()?;
            dev.get_friendlyname().ok().map(|n| (i, n))
        })
        .collect()
}

fn run_configure_tui(cfg: &ClientConfig) -> Result<()> {
    let host = cpal::default_host();
    let input_devices: Vec<(usize, String)> = host
        .input_devices()
        .unwrap_or_else(|_| panic!("input_devices"))
        .enumerate()
        .filter_map(|(i, dev)| dev.name().ok().map(|n| (i, n)))
        .collect();
    let output_devices = list_output_device_names();

    let input_selected = cfg
        .device
        .as_ref()
        .and_then(|d| d.parse::<usize>().ok())
        .filter(|&i| i < input_devices.len())
        .unwrap_or(0);
    let output_selected = cfg.loopback.as_ref().map(|s| s.as_str())
        .or_else(|| cfg.device2.as_ref().and_then(|d| d.strip_prefix("out:")))
        .map_or(0, |s| {
            output_devices
                .iter()
                .position(|(_, n)| n.eq_ignore_ascii_case(s))
                .map(|i| i + 1)
                .unwrap_or(0)
        });

    let mut input_state = ListState::default();
    input_state.select(Some(input_selected.min(input_devices.len().saturating_sub(1))));
    let mut output_state = ListState::default();
    output_state.select(Some(output_selected.min(output_devices.len()))); // +1 for "— нет —"

    let mut focus = 0u8; // 0=input, 1=output
    let mut saved_msg: Option<String> = None;

    terminal::enable_raw_mode()?;
    execute!(io::stdout(), EnableMouseCapture)?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    loop {
        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
                    match key.code {
                        key if key_matches(key, 'q') || key == KeyCode::Esc => break,
                        key if key_matches(key, 's') => {
                            let input_idx = input_state.selected().unwrap_or(0);
                            let device = input_devices
                                .get(input_idx)
                                .map(|(i, _)| format!("{i}"))
                                .or_else(|| input_devices.first().map(|(i, _)| format!("{i}")))
                                .unwrap_or_default();
                            let loopback = output_state
                                .selected()
                                .and_then(|idx| {
                                    if idx == 0 {
                                        None
                                    } else {
                                        output_devices.get(idx - 1).map(|(_, n)| n.clone())
                                    }
                                });

                            let save_cfg = ClientConfig {
                                server: cfg.server.clone().or_else(|| Some("ws://localhost:9745".into())),
                                device: if device.is_empty() { None } else { Some(device) },
                                loopback,
                                device2: None,
                                output: cfg.output.clone().or_else(|| Some("transcript.txt".into())),
                                client_id: cfg.client_id.clone(),
                            };
                            let path = get_config_save_path();
                            if let Ok(text) = serde_json::to_string_pretty(&save_cfg) {
                                if let Err(e) = std::fs::write(&path, text) {
                                    saved_msg = Some(format!("Ошибка: {e}"));
                                } else {
                                    saved_msg = Some(format!("Сохранено: {}", path.display()));
                                }
                            }
                        }
                        KeyCode::Tab | KeyCode::BackTab => focus = 1 - focus,
                        key if key == KeyCode::Up || key_matches(key, 'k') => {
                            let state = if focus == 0 { &mut input_state } else { &mut output_state };
                            let len = if focus == 0 { input_devices.len() } else { output_devices.len() + 1 };
                            let i = state.selected().unwrap_or(0).saturating_sub(1);
                            state.select(Some(i.min(len.saturating_sub(1))));
                        }
                        key if key == KeyCode::Down || key_matches(key, 'j') => {
                            let state = if focus == 0 { &mut input_state } else { &mut output_state };
                            let len = if focus == 0 { input_devices.len() } else { output_devices.len() + 1 };
                            let i = state.selected().unwrap_or(0).saturating_add(1);
                            state.select(Some(i.min(len.saturating_sub(1))));
                        }
                        _ => {}
                    }
                }
                Event::Mouse(mouse) => {
                    if let MouseEventKind::ScrollUp = mouse.kind {
                        let state = if focus == 0 { &mut input_state } else { &mut output_state };
                        let len = if focus == 0 { input_devices.len() } else { output_devices.len() + 1 };
                        let i = state.selected().unwrap_or(0).saturating_sub(1);
                        state.select(Some(i.min(len.saturating_sub(1))));
                    } else if let MouseEventKind::ScrollDown = mouse.kind {
                        let state = if focus == 0 { &mut input_state } else { &mut output_state };
                        let len = if focus == 0 { input_devices.len() } else { output_devices.len() + 1 };
                        let i = state.selected().unwrap_or(0).saturating_add(1);
                        state.select(Some(i.min(len.saturating_sub(1))));
                    }
                }
                _ => {}
            }
        }

        terminal.draw(|f| {
            let area = f.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(5), Constraint::Length(1)])
                .split(area);
            let main_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[0]);

            let input_items: Vec<ListItem> = input_devices
                .iter()
                .map(|(i, n)| ListItem::new(format!("[{i}] {n}")))
                .collect();
            let input_list = List::new(input_items.clone())
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(if focus == 0 { "Микрофон ◄" } else { "Микрофон" })
                        .border_style(Style::default().fg(if focus == 0 { Color::Yellow } else { Color::Reset })),
                )
                .highlight_style(Style::default().bg(Color::DarkGray))
                .highlight_symbol(">> ");
            f.render_stateful_widget(input_list, main_chunks[0], &mut input_state);

            let output_items: Vec<ListItem> = std::iter::once(ListItem::new("— нет —"))
                .chain(
                    output_devices
                        .iter()
                        .map(|(i, n)| ListItem::new(format!("[{i}] {n}"))),
                )
                .collect();
            let output_list = List::new(output_items.clone())
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(if focus == 1 { "Loopback (системный звук) ◄" } else { "Loopback" })
                        .border_style(Style::default().fg(if focus == 1 { Color::Yellow } else { Color::Reset })),
                )
                .highlight_style(Style::default().bg(Color::DarkGray))
                .highlight_symbol(">> ");
            f.render_stateful_widget(output_list, main_chunks[1], &mut output_state);

            f.render_widget(
                Paragraph::new("Tab — переключить | ↑↓ / колёсико — выбор | S — сохранить | Q — выход")
                    .style(Style::default().fg(Color::DarkGray)),
                chunks[1],
            );

            if let Some(ref msg) = saved_msg {
                let a = f.area();
                let popup = ratatui::layout::Rect {
                    x: a.width / 4,
                    y: a.height - 2,
                    width: a.width / 2,
                    height: 1,
                };
                f.render_widget(
                    Paragraph::new(msg.as_str())
                        .block(Block::default().borders(Borders::ALL).title("Сохранено"))
                        .style(Style::default().fg(Color::Green)),
                    popup,
                );
            }
        })?;
    }

    execute!(io::stdout(), DisableMouseCapture)?;
    terminal::disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    Ok(())
}

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

    /// Системный звук — устройство вывода для loopback (индекс или имя, напр. Razer)
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

// ── Audio device resolution ──────────────────────────────────────────

fn list_devices() {
    let host = cpal::default_host();

    eprintln!("Устройства ввода:");
    for (i, dev) in host.input_devices().unwrap().enumerate() {
        let name = dev.name().unwrap_or_else(|_| "?".into());
        let cfg = dev.default_input_config();
        let info = match cfg {
            Ok(c) => format!("rate {}, ch {}", c.sample_rate().0, c.channels()),
            Err(_) => "no config".into(),
        };
        let is_default = host
            .default_input_device()
            .map(|d| d.name().unwrap_or_default() == name)
            .unwrap_or(false);
        let mark = if is_default { " (по умолчанию)" } else { "" };
        eprintln!("  [{i}] {name} ({info}){mark}");
    }

    eprintln!();
    eprintln!("Устройства вывода:");
    for (i, dev) in host.output_devices().unwrap().enumerate() {
        let name = dev.name().unwrap_or_else(|_| "?".into());
        let cfg = dev.default_output_config();
        let info = match cfg {
            Ok(c) => format!("rate {}, ch {}", c.sample_rate().0, c.channels()),
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
    eprintln!("Для захвата системного звука: --loopback Razer или --loopback 0");
}

fn resolve_device(query: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();

    if let Ok(idx) = query.parse::<usize>() {
        return host
            .input_devices()?
            .nth(idx)
            .context(format!("Устройство с индексом {idx} не найдено"));
    }

    let needle = query.to_lowercase();
    let matches: Vec<(usize, cpal::Device, String)> = host
        .input_devices()?
        .enumerate()
        .filter_map(|(i, dev)| {
            let name = dev.name().unwrap_or_default();
            if name.to_lowercase().contains(&needle) {
                Some((i, dev, name))
            } else {
                None
            }
        })
        .collect();

    match matches.len() {
        0 => anyhow::bail!("Устройство «{query}» не найдено. Запустите --list-devices"),
        1 => Ok(matches.into_iter().next().unwrap().1),
        _ => {
            eprintln!("⚠ «{query}» совпало с несколькими устройствами, выбрано первое:");
            for (i, _, name) in &matches {
                eprintln!("  [{i}] {name}");
            }
            Ok(matches.into_iter().next().unwrap().1)
        }
    }
}

fn default_device() -> Result<cpal::Device> {
    cpal::default_host()
        .default_input_device()
        .context("Нет устройства ввода по умолчанию")
}

// ── Audio helpers ────────────────────────────────────────────────────

fn to_mono(data: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    let ch = channels as usize;
    data.chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn resample(src: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == SAMPLE_RATE {
        return src.to_vec();
    }
    let ratio = src_rate as f64 / SAMPLE_RATE as f64;
    let out_len = (src.len() as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = pos - idx as f64;
        let a = src[idx.min(src.len() - 1)];
        let b = src[(idx + 1).min(src.len() - 1)];
        out.push(a + (b - a) * frac as f32);
    }
    out
}

// ── Threads ──────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum ServerMessage {
    Transcript {
        text: String,
        #[serde(default)]
        source: Option<u8>,
    },
    Status(StatusData),
    Debug { text: String },
    Error { text: String },
}

#[derive(Deserialize, Debug, Clone, Default)]
struct StatusData {
    #[serde(default)]
    recording: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    running: bool,
    #[serde(default)]
    device: String,
    #[serde(default)]
    #[allow(dead_code)]
    task_queue_size: u32,
    #[serde(default)]
    #[allow(dead_code)]
    task_queue_2_size: u32,
    #[serde(default)]
    buffer_sec: f64,
    #[serde(default)]
    buffer_2_sec: f64,
    #[serde(default)]
    dropped_total: u64,
    #[serde(default)]
    dropped_2_total: u64,
    #[serde(default)]
    pending_1: u32,
    #[serde(default)]
    pending_2: u32,
    #[serde(default)]
    worker_busy: bool,
    #[serde(default)]
    worker_2_busy: bool,
    #[serde(default)]
    worker_last_proc_sec: f64,
    #[serde(default)]
    worker_2_last_proc_sec: f64,
    #[serde(default)]
    worker_last_audio_sec: f64,
    #[serde(default)]
    worker_2_last_audio_sec: f64,
    #[serde(default)]
    skipped_1: u64,
    #[serde(default)]
    skipped_2: u64,
    /// Отставание распознавания (с) — общее
    #[serde(default)]
    lag_sec: Option<f64>,
    /// Отставание по источнику 0 (mic)
    #[serde(default)]
    lag_sec_0: Option<f64>,
    /// Отставание по источнику 1 (sys)
    #[serde(default)]
    lag_sec_1: Option<f64>,
    /// Размер папки с аудио на сервере (MB)
    #[serde(default)]
    audio_dir_size_mb: Option<f64>,
    /// Очередь на LLM-коррекцию
    #[serde(default)]
    llm_queue: Option<u32>,
}

enum UiEvent {
    Server(ServerMessage),
    AudioLevel { source: u8, level: f32 },
    Connected,
    Disconnected { reason: String },
    Quit,
}

enum WsOutgoing {
    Binary(Vec<u8>),
    Text(String),
}

fn audio_capture(
    device: cpal::Device,
    source_id: u8,
    ws_tx: mpsc::Sender<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: std::sync::Arc<AtomicBool>,
) -> Result<()> {
    let supported = device.default_input_config()?;
    let native_rate = supported.sample_rate().0;
    let native_channels = supported.channels();

    let config = cpal::StreamConfig {
        channels: native_channels,
        sample_rate: cpal::SampleRate(native_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let ui_tx2 = ui_tx.clone();
    let running2 = running.clone();

    let mut pcm_buf: Vec<i16> = Vec::with_capacity(CHUNK_FRAMES * 4);

    let ui_tx_err = ui_tx.clone();
    let src_id = source_id;
    let err_fn = move |err: cpal::StreamError| {
        let _ = ui_tx_err.send(UiEvent::Server(ServerMessage::Error {
            text: format!("audio src{} stream error: {err}", src_id),
        }));
    };

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !running2.load(Ordering::Relaxed) {
                return;
            }
            let mono = to_mono(data, native_channels);
            let resampled = resample(&mono, native_rate);
            let samples: Vec<i16> = resampled
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            pcm_buf.extend_from_slice(&samples);

            while pcm_buf.len() >= CHUNK_FRAMES {
                let chunk: Vec<i16> = pcm_buf.drain(..CHUNK_FRAMES).collect();
                let rms: f64 =
                    (chunk.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / chunk.len() as f64)
                        .sqrt();
                let level = (rms / 32768.0 * 12.0).min(1.0) as f32;
                let _ = ui_tx2.send(UiEvent::AudioLevel { source: source_id, level });

                let mut bytes = Vec::with_capacity(1 + chunk.len() * 2);
                bytes.push(source_id);
                bytes.extend(chunk.iter().flat_map(|s| s.to_le_bytes()));
                let _ = ws_tx.send(WsOutgoing::Binary(bytes));
            }
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    while running.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(100));
    }

    drop(stream);
    Ok(())
}

fn resolve_output_device(query: &str) -> Result<wasapi::Device> {
    let enumerator = wasapi::DeviceEnumerator::new()
        .map_err(|e| anyhow::anyhow!("DeviceEnumerator: {e:?}"))?;

    if let Ok(idx) = query.parse::<usize>() {
        let collection = enumerator
            .get_device_collection(&wasapi::Direction::Render)
            .map_err(|e| anyhow::anyhow!("get_device_collection: {e:?}"))?;
        return collection
            .into_iter()
            .nth(idx)
            .context(format!("Выходное устройство с индексом {idx} не найдено"))?
            .map_err(|e| anyhow::anyhow!("device error: {e:?}"));
    }

    let needle = query.to_lowercase();
    let collection = enumerator
        .get_device_collection(&wasapi::Direction::Render)
        .map_err(|e| anyhow::anyhow!("get_device_collection: {e:?}"))?;
    for dev_result in collection.into_iter() {
        let dev = dev_result.map_err(|e| anyhow::anyhow!("device error: {e:?}"))?;
        let name = dev.get_friendlyname().unwrap_or_default();
        if name.to_lowercase().contains(&needle) {
            return Ok(dev);
        }
    }

    anyhow::bail!("Выходное устройство «{query}» не найдено. Запустите --list-devices")
}

fn loopback_capture(
    device_query: String,
    ws_tx: mpsc::Sender<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: std::sync::Arc<AtomicBool>,
) -> Result<()> {
    wasapi::initialize_mta()
        .ok()
        .context("COM init failed in loopback thread")?;

    let device = resolve_output_device(&device_query)?;
    let mut audio_client = device
        .get_iaudioclient()
        .map_err(|e| anyhow::anyhow!("get_iaudioclient: {e:?}"))?;

    let desired_format =
        wasapi::WaveFormat::new(16, 16, &wasapi::SampleType::Int, SAMPLE_RATE as usize, 1, None);

    let (_, min_time) = audio_client
        .get_device_period()
        .map_err(|e| anyhow::anyhow!("get_device_period: {e:?}"))?;

    let mode = wasapi::StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: min_time,
    };

    audio_client
        .initialize_client(&desired_format, &wasapi::Direction::Capture, &mode)
        .map_err(|e| anyhow::anyhow!("initialize_client loopback: {e:?}"))?;

    let h_event = audio_client
        .set_get_eventhandle()
        .map_err(|e| anyhow::anyhow!("set_get_eventhandle: {e:?}"))?;

    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|e| anyhow::anyhow!("get_audiocaptureclient: {e:?}"))?;

    audio_client
        .start_stream()
        .map_err(|e| anyhow::anyhow!("start_stream: {e:?}"))?;

    let blockalign = desired_format.get_blockalign() as usize;
    let chunk_bytes = CHUNK_FRAMES * blockalign;
    let mut sample_queue: VecDeque<u8> = VecDeque::with_capacity(chunk_bytes * 8);

    while running.load(Ordering::Relaxed) {
        capture_client
            .read_from_device_to_deque(&mut sample_queue)
            .map_err(|e| anyhow::anyhow!("read_from_device: {e:?}"))?;

        while sample_queue.len() >= chunk_bytes {
            let mut tagged = Vec::with_capacity(1 + chunk_bytes);
            tagged.push(1u8);
            let mut rms_sum: f64 = 0.0;
            for i in 0..chunk_bytes {
                let b = sample_queue.pop_front().unwrap();
                tagged.push(b);
                if i % 2 == 0 && i + 1 < chunk_bytes {
                    let lo = b as u8;
                    let hi = *sample_queue.front().unwrap_or(&0);
                    let sample = i16::from_le_bytes([lo, hi]);
                    rms_sum += (sample as f64).powi(2);
                }
            }
            let rms = (rms_sum / CHUNK_FRAMES as f64).sqrt();
            let level = (rms / 32768.0 * 12.0).min(1.0) as f32;
            let _ = ui_tx.send(UiEvent::AudioLevel { source: 1, level });
            let _ = ws_tx.send(WsOutgoing::Binary(tagged));
        }

        if h_event.wait_for_event(100).is_err() {}
    }

    audio_client
        .stop_stream()
        .map_err(|e| anyhow::anyhow!("stop_stream: {e:?}"))?;
    Ok(())
}

fn ws_io_thread(
    server_url: String,
    out_rx: mpsc::Receiver<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: std::sync::Arc<AtomicBool>,
    source_count: u8,
    client_id: Option<String>,
    initial_recording: bool,
) {
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
        let host = parsed_url.host_str().unwrap_or("127.0.0.1");
        let port = parsed_url.port().unwrap_or(9745);
        let addr = format!("{}:{}", host, port);

        let tcp = match std::net::TcpStream::connect_timeout(
            &addr.parse().unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], port))),
            Duration::from_secs(3),
        ) {
            Ok(s) => s,
            Err(e) => {
                let _ = ui_tx.send(UiEvent::Disconnected {
                    reason: format!("подключение: {e}"),
                });
                for _ in 0..20 {
                    if !running.load(Ordering::Relaxed) { return; }
                    thread::sleep(Duration::from_millis(100));
                }
                continue;
            }
        };
        let (mut ws, _) = match tungstenite::client::client_with_config(
            &server_url,
            tcp,
            None,
        ) {
            Ok(x) => x,
            Err(e) => {
                let _ = ui_tx.send(UiEvent::Disconnected {
                    reason: format!("ws handshake: {e}"),
                });
                for _ in 0..20 {
                    if !running.load(Ordering::Relaxed) { return; }
                    thread::sleep(Duration::from_millis(100));
                }
                continue;
            }
        };

        let mut config_msg = serde_json::json!({
            "type": "config",
            "source_count": source_count,
            "mode": "live",
        });
        if let Some(ref id) = client_id {
            config_msg["session_id"] = serde_json::json!(id);
        }
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
                        let _ = ui_tx.send(UiEvent::Server(srv_msg));
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
                if !running.load(Ordering::Relaxed) { return; }
                thread::sleep(Duration::from_millis(100));
            }
        }
    }

    running.store(false, Ordering::Relaxed);
    let _ = ui_tx.send(UiEvent::Quit);
}

enum TuiMode {
    Main,
    Settings,
}

struct SettingsState {
    input_devices: Vec<(usize, String)>,
    output_devices: Vec<(usize, String)>,
    input_state: ListState,
    output_state: ListState,
    focus: u8,
    saved_msg: Option<String>,
    saved_at: Option<Instant>,
}

/// Returns (need_restart, recording) — need_restart if settings were saved, recording state to preserve.
fn run_tui(
    ui_rx: mpsc::Receiver<UiEvent>,
    ws_tx: mpsc::Sender<WsOutgoing>,
    running: std::sync::Arc<AtomicBool>,
    output_path: &str,
    _has_source2: bool,
    cfg: &ClientConfig,
    initial_recording: bool,
) -> Result<(bool, bool)> {
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut transcript_lines: Vec<(String, Option<u8>)> = Vec::new();
    let mut debug_lines: Vec<String> = Vec::new();
    let mut audio_level: f32 = 0.0;
    let mut audio_level2: f32 = 0.0;
    let mut status = StatusData::default();
    let mut recording = initial_recording;
    let mut connected = false;
    let mut last_activity = Instant::now();
    let mut last_audio_dir_mb: Option<f64> = None;
    let mut last_llm_queue: Option<u32> = None;
    let mut last_disconnect_reason: Option<String> = None;

    let mut mode = TuiMode::Main;
    let mut settings_state: Option<SettingsState> = None;

    let mut focus: u8 = 0;
    let mut transcript_scroll: Option<u16> = None;
    let mut debug_scroll: Option<u16> = None;
    let mut need_restart = false;
    let mut restart_at: Option<Instant> = None;

    let mut out_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;

    let mut last_draw = Instant::now();

    loop {
        if event::poll(Duration::from_millis(30))? {
            match event::read()? {
                Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
                    if matches!(mode, TuiMode::Settings) {
                        let st = settings_state.as_mut().unwrap();
                        match key.code {
                            KeyCode::Esc => mode = TuiMode::Main,
                            code if key_matches(code, 's') => {
                                let input_idx = st.input_state.selected().unwrap_or(0);
                                let device = st.input_devices
                                    .get(input_idx)
                                    .map(|(i, _)| format!("{i}"))
                                    .or_else(|| st.input_devices.first().map(|(i, _)| format!("{i}")))
                                    .unwrap_or_default();
                                let loopback = st.output_state.selected().and_then(|idx| {
                                    if idx == 0 { None }
                                    else { st.output_devices.get(idx - 1).map(|(_, n)| n.clone()) }
                                });
                                let save_cfg = ClientConfig {
                                    server: cfg.server.clone().or_else(|| Some("ws://localhost:9745".into())),
                                    device: if device.is_empty() { None } else { Some(device) },
                                    loopback,
                                    device2: None,
                                    output: cfg.output.clone().or_else(|| Some("transcript.txt".into())),
                                    client_id: cfg.client_id.clone(),
                                };
                                let path = get_config_save_path();
                                if let Ok(text) = serde_json::to_string_pretty(&save_cfg) {
                                    st.saved_msg = Some(match std::fs::write(&path, text) {
                                        Ok(_) => {
                                            need_restart = true;
                                            restart_at = Some(Instant::now());
                                            format!("Сохранено. Применяется…")
                                        }
                                        Err(e) => format!("Ошибка: {e}"),
                                    });
                                    st.saved_at = Some(Instant::now());
                                }
                            }
                            KeyCode::Tab | KeyCode::BackTab => st.focus = 1 - st.focus,
                            code if code == KeyCode::Up || key_matches(code, 'k') => {
                                let state = if st.focus == 0 { &mut st.input_state } else { &mut st.output_state };
                                let len = if st.focus == 0 { st.input_devices.len() } else { st.output_devices.len() + 1 };
                                let i = state.selected().unwrap_or(0).saturating_sub(1);
                                state.select(Some(i.min(len.saturating_sub(1))));
                            }
                            code if code == KeyCode::Down || key_matches(code, 'j') => {
                                let state = if st.focus == 0 { &mut st.input_state } else { &mut st.output_state };
                                let len = if st.focus == 0 { st.input_devices.len() } else { st.output_devices.len() + 1 };
                                let i = state.selected().unwrap_or(0).saturating_add(1);
                                state.select(Some(i.min(len.saturating_sub(1))));
                            }
                            _ => {}
                        }
                    } else {
                        match key.code {
                            KeyCode::F(2) => {
                                if settings_state.is_none() {
                                    let host = cpal::default_host();
                                    let input_devices: Vec<_> = host.input_devices().unwrap_or_else(|_| panic!("input_devices"))
                                        .enumerate().filter_map(|(i, d)| d.name().ok().map(|n| (i, n))).collect();
                                    let output_devices = list_output_device_names();
                                    let input_selected = cfg.device.as_ref().and_then(|d| d.parse::<usize>().ok())
                                        .filter(|&i| i < input_devices.len()).unwrap_or(0);
                                    let output_selected = cfg.loopback.as_ref().map(|s| s.as_str())
                                        .or_else(|| cfg.device2.as_ref().and_then(|d| d.strip_prefix("out:")))
                                        .map_or(0, |s| {
                                        output_devices.iter().position(|(_, n)| n.eq_ignore_ascii_case(s)).map(|i| i + 1).unwrap_or(0)
                                    });
                                    let mut input_state = ListState::default();
                                    input_state.select(Some(input_selected.min(input_devices.len().saturating_sub(1))));
                                    let mut output_state = ListState::default();
                                    output_state.select(Some(output_selected.min(output_devices.len())));
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
                            code if key.modifiers.contains(KeyModifiers::CONTROL) && key_matches(code, 'c') => {
                                running.store(false, Ordering::Relaxed);
                                break;
                            }
                            code if key_matches(code, 'r') => {
                                recording = !recording;
                                let msg = serde_json::json!({"type": "recording", "enabled": recording});
                                let _ = ws_tx.send(WsOutgoing::Text(msg.to_string()));
                            }
                            KeyCode::Tab | KeyCode::BackTab => focus = 1 - focus,
                            code if code == KeyCode::Up || key_matches(code, 'k') => {
                                let scroll = if focus == 0 { &mut transcript_scroll } else { &mut debug_scroll };
                                *scroll = Some(scroll.unwrap_or(u16::MAX).saturating_sub(1));
                            }
                            code if code == KeyCode::Down || key_matches(code, 'j') => {
                                let scroll = if focus == 0 { &mut transcript_scroll } else { &mut debug_scroll };
                                *scroll = Some(scroll.unwrap_or(0).saturating_add(1));
                            }
                            KeyCode::End => {
                                if focus == 0 { transcript_scroll = None; } else { debug_scroll = None; }
                            }
                            KeyCode::Home => {
                                if focus == 0 { transcript_scroll = Some(0); } else { debug_scroll = Some(0); }
                            }
                            _ => {}
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    if matches!(mode, TuiMode::Settings) {
                        let st = settings_state.as_mut().unwrap();
                        if let MouseEventKind::ScrollUp = mouse.kind {
                            let state = if st.focus == 0 { &mut st.input_state } else { &mut st.output_state };
                            let len = if st.focus == 0 { st.input_devices.len() } else { st.output_devices.len() + 1 };
                            let i = state.selected().unwrap_or(0).saturating_sub(1);
                            state.select(Some(i.min(len.saturating_sub(1))));
                        } else if let MouseEventKind::ScrollDown = mouse.kind {
                            let state = if st.focus == 0 { &mut st.input_state } else { &mut st.output_state };
                            let len = if st.focus == 0 { st.input_devices.len() } else { st.output_devices.len() + 1 };
                            let i = state.selected().unwrap_or(0).saturating_add(1);
                            state.select(Some(i.min(len.saturating_sub(1))));
                        }
                    } else if let MouseEventKind::ScrollUp = mouse.kind {
                        let scroll = if focus == 0 { &mut transcript_scroll } else { &mut debug_scroll };
                        *scroll = Some(scroll.unwrap_or(u16::MAX).saturating_sub(1));
                    } else if let MouseEventKind::ScrollDown = mouse.kind {
                        let scroll = if focus == 0 { &mut transcript_scroll } else { &mut debug_scroll };
                        *scroll = Some(scroll.unwrap_or(0).saturating_add(1));
                    }
                }
                _ => {}
            }
        }

        while let Ok(ev) = ui_rx.try_recv() {
            match ev {
                UiEvent::Server(ServerMessage::Transcript { text, source }) => {
                    last_activity = Instant::now();
                    transcript_lines.push((text.clone(), source));
                    transcript_scroll = None;
                    if recording {
                        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
                        let prefix = match source {
                            Some(0) => "mic: ",
                            Some(1) => "sys: ",
                            _ => "",
                        };
                        let _ = writeln!(out_file, "[{ts}] {prefix}{text}");
                        let _ = out_file.flush();
                    }
                    if transcript_lines.len() > 500 {
                        transcript_lines.drain(..transcript_lines.len() - 500);
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
                    status = s;
                }
                UiEvent::Server(ServerMessage::Debug { text }) => {
                    debug_lines.push(text);
                    if debug_lines.len() > 200 {
                        debug_lines.drain(..debug_lines.len() - 200);
                    }
                }
                UiEvent::Server(ServerMessage::Error { text }) => {
                    debug_lines.push(format!("[ERROR] {text}"));
                }
                UiEvent::AudioLevel { source: 0, level } => audio_level = level,
                UiEvent::AudioLevel { source: _, level } => audio_level2 = level,
                UiEvent::Connected => {
                    connected = true;
                    debug_lines.push("[ws] подключено".into());
                }
                UiEvent::Disconnected { reason } => {
                    connected = false;
                    last_disconnect_reason = Some(reason.clone());
                    debug_lines.push(format!("[ws] отключено: {reason}"));
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
        if need_restart && restart_at.map_or(false, |t| t.elapsed() >= Duration::from_secs(2)) {
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
                Span::styled("[r] [F2] [q]", Style::default().fg(Color::DarkGray)),
            ]);

            let status_widget = Paragraph::new(vec![line1])
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Статус")
                        .border_style(Style::default().fg(if connected { Color::Reset } else { Color::Red })),
                );
            f.render_widget(status_widget, chunks[0]);

            let t_inner_w = chunks[1].width.saturating_sub(2).max(1) as usize;
            let t_total: u16 = transcript_lines
                .iter()
                .map(|(l, src)| {
                    let prefix_len = match *src {
                        Some(0) | Some(1) => 4, // "mic " / "sys "
                        _ => 0,
                    };
                    let w = prefix_len + l.chars().count();
                    if w == 0 { 1u16 } else { ((w + t_inner_w - 1) / t_inner_w) as u16 }
                })
                .sum();
            let t_visible = chunks[1].height.saturating_sub(2);
            let t_max = t_total.saturating_sub(t_visible);
            let t_scroll_y = match transcript_scroll {
                None => t_max,
                Some(v) => v.min(t_max),
            };
            if let Some(ref mut v) = transcript_scroll {
                *v = (*v).min(t_max);
            } else {
                transcript_scroll = Some(t_max);
            }

            let t_title = if focus == 0 { "Транскрипция ◄" } else { "Транскрипция" };
            let t_border = if focus == 0 { Color::Yellow } else { Color::Reset };
            let t_lines: Vec<Line> = transcript_lines
                .iter()
                .map(|(text, source)| {
                    let (prefix, color) = match *source {
                        Some(0) => ("mic ", Color::Green),
                        Some(1) => ("sys ", Color::Cyan),
                        _ => ("", Color::White),
                    };
                    let mut spans = vec![];
                    if !prefix.is_empty() {
                        spans.push(Span::styled(prefix, Style::default().fg(color).add_modifier(Modifier::DIM)));
                    }
                    spans.push(Span::styled(text.as_str(), Style::default().fg(color)));
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
                    if w == 0 { 1u16 } else { ((w + d_inner_w - 1) / d_inner_w) as u16 }
                })
                .sum();
            let d_visible = chunks[2].height.saturating_sub(2);
            let d_max = d_total.saturating_sub(d_visible);
            let d_scroll_y = match debug_scroll {
                None => d_max,
                Some(v) => v.min(d_max),
            };
            if let Some(ref mut v) = debug_scroll {
                *v = (*v).min(d_max);
            } else {
                debug_scroll = Some(d_max);
            }

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
        run_configure_tui(&cfg)?;
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
        let ws_thread = thread::spawn(move || {
            ws_io_thread(ws_url, ws_out_rx, ui_tx_ws, running_ws, src_count, client_id, recording);
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

        let (need_restart, new_recording) = run_tui(ui_rx, ws_out_tx, running.clone(), &output, has_source2, &cfg_for_tui, recording)?;
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
            ServerMessage::Transcript { text, source } => {
                assert_eq!(text, "привет");
                assert_eq!(*source, Some(0));
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
}
