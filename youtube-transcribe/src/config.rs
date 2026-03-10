//! Настройки из settings.json.

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize)]
pub struct Settings {
    /// Путь к yt-dlp (по умолчанию ищет в текущей папке, затем рядом с exe)
    #[serde(default)]
    pub yt_dlp_path: Option<String>,
    /// Путь к ffmpeg (по умолчанию "ffmpeg" из PATH)
    #[serde(default)]
    pub ffmpeg_path: Option<String>,
    /// JS runtime для yt-dlp: "node", "deno" или "node:C:/path/to/node.exe"
    #[serde(default)]
    pub js_runtime: Option<String>,
    /// Путь к node.exe (если задан, передаётся как node:path в yt-dlp)
    #[serde(default)]
    pub js_runtime_path: Option<String>,
    #[serde(default = "default_server")]
    pub server: String,
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
    #[serde(default)]
    pub state_file: Option<String>,
}

fn default_server() -> String {
    "ws://127.0.0.1:9745".to_string()
}
fn default_output_dir() -> String {
    "transcripts".to_string()
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            yt_dlp_path: None,
            ffmpeg_path: None,
            js_runtime: None,
            js_runtime_path: None,
            server: default_server(),
            output_dir: default_output_dir(),
            state_file: None,
        }
    }
}

fn settings_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("settings.json"));
            out.push(dir.join("youtube-transcribe-settings.json"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        out.push(cwd.join("youtube-transcribe").join("settings.json"));
        out.push(cwd.join("settings.json"));
        out.push(cwd.join("youtube-transcribe-settings.json"));
    }
    out.push(PathBuf::from("settings.json"));
    out
}

pub fn load_settings() -> Settings {
    for path in settings_candidates() {
        if path.is_file() {
            match std::fs::read_to_string(&path) {
                Ok(s) => {
                    if let Ok(cfg) = serde_json::from_str::<Settings>(&s) {
                        return cfg;
                    }
                }
                Err(_) => {}
            }
        }
    }
    Settings::default()
}

/// Разрешает путь к yt-dlp: settings → текущая папка → папка exe → "yt-dlp"
pub fn resolve_yt_dlp(settings: &Settings, cli_override: Option<&PathBuf>) -> String {
    if let Some(p) = cli_override {
        return p.to_string_lossy().to_string();
    }
    if let Some(ref s) = settings.yt_dlp_path {
        let p = PathBuf::from(s);
        let p = if p.is_absolute() {
            p
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.join(&p)
        } else {
            p
        };
        if p.is_file() {
            return p.to_string_lossy().to_string();
        }
    }
    // Дефолт: текущая папка
    if let Ok(cwd) = std::env::current_dir() {
        for name in ["yt-dlp.exe", "yt-dlp"] {
            let p = cwd.join(name);
            if p.is_file() {
                return p.to_string_lossy().to_string();
            }
        }
    }
    // Папка exe
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for name in ["yt-dlp.exe", "yt-dlp"] {
                let p = dir.join(name);
                if p.is_file() {
                    return p.to_string_lossy().to_string();
                }
            }
        }
    }
    "yt-dlp".to_string()
}

/// Путь к ffmpeg.exe для вызова. Дефолт: ищет в текущей папке, затем "ffmpeg" из PATH.
pub fn resolve_ffmpeg(settings: &Settings, cli_override: Option<&PathBuf>) -> String {
    if let Some(p) = cli_override {
        return p.to_string_lossy().to_string();
    }
    if let Some(ref s) = settings.ffmpeg_path {
        let p = PathBuf::from(s);
        let p = if p.is_absolute() {
            p
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.join(&p)
        } else {
            p
        };
        if p.is_file() {
            return p.to_string_lossy().to_string();
        }
        if p.is_dir() {
            for name in ["ffmpeg.exe", "ffmpeg"] {
                let exe = p.join(name);
                if exe.is_file() {
                    return exe.to_string_lossy().to_string();
                }
            }
        }
    }
    // Дефолт: текущая папка
    if let Ok(cwd) = std::env::current_dir() {
        for name in ["ffmpeg.exe", "ffmpeg"] {
            let p = cwd.join(name);
            if p.is_file() {
                return p.to_string_lossy().to_string();
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for name in ["ffmpeg.exe", "ffmpeg"] {
                let p = dir.join(name);
                if p.is_file() {
                    return p.to_string_lossy().to_string();
                }
            }
        }
    }
    "ffmpeg".to_string()
}

/// Значение для --js-runtime: "node", "node:C:/path" и т.д.
pub fn resolve_js_runtime(settings: &Settings) -> Option<String> {
    let runtime = settings.js_runtime.as_deref().unwrap_or("node");
    let path = settings.js_runtime_path.as_deref();

    if let Some(p) = path {
        let p = PathBuf::from(p);
        let p = if p.is_absolute() {
            p
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.join(&p)
        } else {
            p
        };
        if p.is_file() {
            let name = runtime.split(':').next().unwrap_or("node").trim();
            return Some(format!("{}:{}", if name.is_empty() { "node" } else { name }, p.to_string_lossy()));
        }
    }

    if runtime.contains(['\\', '/']) {
        let p = PathBuf::from(runtime);
        if p.is_file() {
            return Some(format!("node:{}", p.to_string_lossy()));
        }
    }

    if runtime.contains(':') {
        return Some(runtime.to_string());
    }

    if !runtime.is_empty() && runtime != "false" {
        Some(runtime.to_string())
    } else {
        None
    }
}

/// Директория с ffmpeg для yt-dlp --ffmpeg-location (там же ffprobe). None = не передавать.
pub fn resolve_ffmpeg_location_for_ytdlp(ffmpeg_path: &str) -> Option<String> {
    let p = PathBuf::from(ffmpeg_path);
    if p == PathBuf::from("ffmpeg") {
        return None;
    }
    let dir = if p.is_file() {
        p.parent()?.to_path_buf()
    } else if p.is_dir() {
        p
    } else {
        return None;
    };
    Some(dir.to_string_lossy().to_string())
}
