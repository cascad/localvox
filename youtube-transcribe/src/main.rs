//! Очередь транскрипции YouTube: TUI, durability, перезапуск продолжает работу.

mod config;
mod overlap;
mod state;
mod worker;

use anyhow::Result;
use clap::Parser;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Terminal;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, MouseEventKind,
};

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
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};

use config::{
    load_settings, resolve_ffmpeg, resolve_ffmpeg_location_for_ytdlp, resolve_js_runtime,
    resolve_yt_dlp,
};
use state::{JobStatus, QueueState};
use worker::run_worker;

#[derive(Parser)]
#[command(name = "youtube-transcribe")]
#[command(about = "Очередь транскрипции YouTube через localvox, TUI, durability")]
struct Cli {
    /// URL видео для добавления в очередь (можно несколько)
    #[arg()]
    urls: Vec<String>,

    /// WebSocket сервер localvox
    #[arg(long, default_value = "ws://127.0.0.1:9745")]
    server: String,

    /// Папка для транскриптов (по умолчанию ./transcripts)
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Файл состояния очереди (по умолчанию queue.json рядом с exe)
    #[arg(long)]
    state_file: Option<PathBuf>,

    /// Путь к yt-dlp (или yt-dlp.exe рядом с exe)
    #[arg(long)]
    yt_dlp: Option<PathBuf>,

    /// Путь к ffmpeg
    #[arg(long)]
    ffmpeg: Option<PathBuf>,

    /// Сырой вывод: только сортировка по времени, без склейки (для отладки)
    #[arg(long)]
    raw: bool,

    /// API key for server auth (or LOCALVOX_API_KEY env)
    #[arg(long)]
    api_key: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let settings = load_settings();

    let output_dir = cli
        .output_dir
        .unwrap_or_else(|| PathBuf::from(&settings.output_dir));
    let _ = std::fs::create_dir_all(&output_dir);

    let state_path = cli
        .state_file
        .or_else(|| settings.state_file.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("queue.json")))
                .unwrap_or_else(|| PathBuf::from("queue.json"))
        });

    let server = if cli.server != "ws://127.0.0.1:9745" {
        cli.server.clone()
    } else {
        settings.server.clone()
    };

    let yt_dlp = resolve_yt_dlp(&settings, cli.yt_dlp.as_ref());
    let ffmpeg = resolve_ffmpeg(&settings, cli.ffmpeg.as_ref());
    let ffmpeg_location = resolve_ffmpeg_location_for_ytdlp(&ffmpeg);
    let js_runtime = resolve_js_runtime(&settings);

    let state = Arc::new(Mutex::new(QueueState::load(&state_path)?));
    let state_clone = state.clone();

    // При старте: in_progress → pending (retry)
    {
        let mut s = state.lock().unwrap();
        for j in s.jobs.iter_mut() {
            if matches!(j.status, JobStatus::InProgress) {
                j.status = JobStatus::Pending;
            }
        }
        for url in &cli.urls {
            if url::Url::parse(url).is_ok() {
                let out_name =
                    output_dir.join(format!("{}.txt", uuid::Uuid::new_v4().simple().to_string()));
                s.add_job(url.clone(), out_name);
            }
        }
        s.save(&state_path)?;
    }

    let api_key = cli
        .api_key
        .clone()
        .or(settings.api_key.clone())
        .or_else(|| std::env::var("LOCALVOX_API_KEY").ok());

    let worker_state = state.clone();
    let worker_path = state_path.clone();
    let worker_output_dir = output_dir.clone();
    let worker_server = server.clone();
    let raw_mode = cli.raw;
    let worker_api_key = api_key.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(run_worker(
            worker_state,
            worker_path,
            worker_output_dir,
            worker_server,
            worker_api_key,
            &yt_dlp,
            &ffmpeg,
            ffmpeg_location,
            js_runtime,
            raw_mode,
        ));
    });

    run_tui(state_clone, state_path, output_dir)?;

    Ok(())
}

fn open_file(path: &str) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::process::Command::new("cmd")
            .args(["/c", "start", "", path])
            .spawn()
            .map(|_| ())
    }
    #[cfg(not(windows))]
    {
        std::process::Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map(|_| ())
    }
}

fn run_tui(state: Arc<Mutex<QueueState>>, state_path: PathBuf, output_dir: PathBuf) -> Result<()> {
    terminal::enable_raw_mode()?;
    execute!(io::stdout(), EnableMouseCapture)?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut list_state = ListState::default();
    list_state.select(Some(0));
    let mut input_buf = String::new();
    let mut input_mode = false;
    let mut msg: Option<String> = None;
    let mut msg_until: Option<Instant> = None;
    let mut list_height: usize = 10;

    loop {
        {
            let s = state.lock().unwrap();
            let items: Vec<ListItem> = s
                .jobs
                .iter()
                .map(|j| {
                    let (status_display, style) = match &j.status {
                        JobStatus::Pending => ("ожидание".into(), Style::default()),
                        JobStatus::InProgress => {
                            let base = if let Some((sent, total)) = j.progress {
                                let s_mb = sent as f64 / 1024.0 / 1024.0;
                                let t_mb = total as f64 / 1024.0 / 1024.0;
                                format!("в работе {:.1}/{:.1} МБ", s_mb, t_mb)
                            } else {
                                "в работе".into()
                            };
                            let txt = if let Some(ref ss) = j.server_status {
                                format!(
                                    "{} | asr:{} llm:{} lag:{:.0}с",
                                    base, ss.task_queue_size, ss.llm_queue, ss.lag_sec
                                )
                            } else {
                                base
                            };
                            (txt, Style::default().fg(Color::Yellow))
                        }
                        JobStatus::Done => ("готово".into(), Style::default().fg(Color::Green)),
                        JobStatus::Failed(e) => (
                            format!("ошибка: {}", e.chars().take(20).collect::<String>()),
                            Style::default().fg(Color::Red),
                        ),
                    };
                    let short = j.url.chars().take(40).collect::<String>();
                    let path_hint = match &j.status {
                        JobStatus::Done => {
                            let name = std::path::Path::new(&j.output_path)
                                .file_name()
                                .map(|n| n.to_string_lossy())
                                .unwrap_or_default();
                            format!(" → {}", name)
                        }
                        _ => String::new(),
                    };
                    ListItem::from(Line::from(vec![
                        Span::raw(format!("{} | ", short)),
                        Span::styled(status_display, style),
                        Span::raw(path_hint),
                    ]))
                })
                .collect();

            let selected_path = list_state
                .selected()
                .and_then(|i| s.jobs.get(i))
                .and_then(|j| {
                    if matches!(j.status, JobStatus::Done) {
                        Some(j.output_path.clone())
                    } else {
                        None
                    }
                });
            let title = Paragraph::new("YouTube Transcribe Queue (a: добавить, o: открыть, j/k/↑↓: навигация, PgUp/PgDn: страница, колёсико: скролл, q: выход)")
                .block(Block::default().borders(Borders::ALL).title(" Очередь "));
            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Min(5),
                        Constraint::Length(3),
                    ])
                    .split(f.area());
                f.render_widget(title, chunks[0]);

                list_height = chunks[1].height.saturating_sub(2).max(1) as usize;
                let list = List::new(items)
                    .block(Block::default().borders(Borders::ALL))
                    .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
                f.render_stateful_widget(list, chunks[1], &mut list_state);

                let help = if msg_until.as_ref().map_or(false, |t| Instant::now() < *t) {
                    msg.as_ref().cloned().unwrap_or_default()
                } else if input_mode {
                    format!("URL: {}_", input_buf)
                } else if let Some(ref p) = selected_path {
                    format!("Файл: {} | State: {}", p, state_path.display())
                } else {
                    format!(
                        "State: {} | Output: {}",
                        state_path.display(),
                        output_dir.display()
                    )
                };
                let footer = Paragraph::new(help)
                    .block(Block::default().borders(Borders::ALL))
                    .wrap(Wrap { trim: true });
                f.render_widget(footer, chunks[2]);
            })?;
        }

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
                    if input_mode {
                        match key.code {
                            KeyCode::Enter => {
                                let url = input_buf.trim().to_string();
                                input_buf.clear();
                                input_mode = false;
                                if !url.is_empty() {
                                    if url::Url::parse(&url).is_ok() {
                                        let out_name = output_dir.join(format!(
                                            "{}.txt",
                                            uuid::Uuid::new_v4().simple().to_string()
                                        ));
                                        let mut s = state.lock().unwrap();
                                        s.add_job(url, out_name);
                                        s.save(&state_path)?;
                                        msg = Some("Добавлено в очередь".into());
                                        msg_until = Some(Instant::now() + Duration::from_secs(2));
                                    } else {
                                        msg = Some("Некорректный URL".into());
                                        msg_until = Some(Instant::now() + Duration::from_secs(2));
                                    }
                                }
                            }
                            KeyCode::Esc => {
                                input_mode = false;
                                input_buf.clear();
                            }
                            KeyCode::Backspace => {
                                input_buf.pop();
                            }
                            KeyCode::Char(c) => {
                                input_buf.push(c);
                            }
                            _ => {}
                        }
                    } else {
                        match key.code {
                            code if key_matches(code, 'q') || code == KeyCode::Esc => break,
                            code if key_matches(code, 'a') => {
                                input_mode = true;
                                msg = None;
                            }
                            code if key_matches(code, 'o') => {
                                if let Some(i) = list_state.selected() {
                                    let path = state.lock().unwrap().jobs.get(i).and_then(|j| {
                                        if matches!(j.status, JobStatus::Done)
                                            && std::path::Path::new(&j.output_path).is_file()
                                        {
                                            Some(j.output_path.clone())
                                        } else {
                                            None
                                        }
                                    });
                                    if let Some(p) = path {
                                        let _ = open_file(&p);
                                        msg = Some(format!("Открыто: {}", p));
                                        msg_until = Some(Instant::now() + Duration::from_secs(2));
                                    }
                                }
                            }
                            code if key_matches(code, 'd') => {
                                if let Some(i) = list_state.selected() {
                                    let mut s = state.lock().unwrap();
                                    if i < s.jobs.len() {
                                        let j = &s.jobs[i];
                                        if matches!(j.status, JobStatus::Pending) {
                                            s.jobs.remove(i);
                                            s.save(&state_path)?;
                                            msg = Some("Удалено из очереди".into());
                                            msg_until =
                                                Some(Instant::now() + Duration::from_secs(2));
                                        }
                                    }
                                }
                            }
                            code if code == KeyCode::Up || key_matches(code, 'k') => {
                                let len = state.lock().unwrap().jobs.len();
                                let i = list_state.selected().unwrap_or(0).saturating_sub(1);
                                list_state.select(if len > 0 {
                                    Some(i.min(len - 1))
                                } else {
                                    None
                                });
                            }
                            code if code == KeyCode::Down || key_matches(code, 'j') => {
                                let len = state.lock().unwrap().jobs.len();
                                let i = list_state.selected().unwrap_or(0) + 1;
                                list_state.select(if len > 0 {
                                    Some(i.min(len - 1))
                                } else {
                                    None
                                });
                            }
                            KeyCode::PageUp => {
                                let len = state.lock().unwrap().jobs.len();
                                let i = list_state
                                    .selected()
                                    .unwrap_or(0)
                                    .saturating_sub(list_height);
                                list_state.select(if len > 0 {
                                    Some(i.min(len - 1))
                                } else {
                                    None
                                });
                            }
                            KeyCode::PageDown => {
                                let len = state.lock().unwrap().jobs.len();
                                let i = list_state.selected().unwrap_or(0) + list_height;
                                list_state.select(if len > 0 {
                                    Some(i.min(len - 1))
                                } else {
                                    None
                                });
                            }
                            _ => {}
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    if !input_mode {
                        let len = state.lock().unwrap().jobs.len();
                        if let MouseEventKind::ScrollUp = mouse.kind {
                            let i = list_state.selected().unwrap_or(0).saturating_sub(1);
                            list_state.select(if len > 0 { Some(i.min(len - 1)) } else { None });
                        } else if let MouseEventKind::ScrollDown = mouse.kind {
                            let i = list_state.selected().unwrap_or(0) + 1;
                            list_state.select(if len > 0 { Some(i.min(len - 1)) } else { None });
                        }
                    }
                }
                _ => {}
            }
        }

        if msg_until.map_or(false, |t| Instant::now() >= t) {
            msg = None;
            msg_until = None;
        }
    }

    terminal::disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    execute!(io::stdout(), DisableMouseCapture)?;
    terminal.show_cursor()?;

    Ok(())
}
