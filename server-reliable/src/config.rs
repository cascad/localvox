//! Load settings from settings.json.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default = "default_audio_dir")]
    pub audio_dir: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_max_chunk_sec")]
    pub max_chunk_duration_sec: f64,
    #[serde(default = "default_min_chunk_sec")]
    pub min_chunk_duration_sec: f64,
    #[serde(default = "default_vad_silence_sec")]
    pub vad_silence_sec: f64,
    #[serde(default = "default_overlap_sec")]
    pub overlap_sec: f64,
    /// Use GPU for Whisper (CUDA). Requires build with cuda feature. If false or GPU unavailable, uses CPU.
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
    /// Path to folder with CUDA DLLs (cublas64_12.dll etc.). Prepended to PATH before loading model. Required for GPU on Windows.
    #[serde(default)]
    pub cuda_path: Option<String>,
    /// Path to GigaAM model directory (model.int8.onnx + tokens.txt). If set and ensemble enabled, runs Whisper + GigaAM.
    #[serde(default)]
    pub gigaam_model_dir: Option<String>,
    /// Enable ensemble: merge Whisper + GigaAM results. Requires gigaam_model_dir.
    #[serde(default = "default_ensemble_enabled")]
    pub ensemble_enabled: bool,
    /// Enable async LLM correction via Ollama (arbiter + corrector in one pass).
    #[serde(default)]
    pub llm_correction_enabled: bool,
    /// Ollama API base URL.
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    /// Ollama model name for correction.
    #[serde(default = "default_llm_model")]
    pub llm_model: String,
    /// Удалять сессии старше N часов (0 = отключено).
    #[serde(default = "default_session_ttl_hours")]
    pub session_ttl_hours: f64,
    /// Макс. размер audio_dir в MB (0 = без лимита). При превышении удаляются старые сессии.
    #[serde(default = "default_audio_dir_max_mb")]
    pub audio_dir_max_mb: f64,
}

fn default_ensemble_enabled() -> bool {
    false
}
fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}
fn default_llm_model() -> String {
    "qwen2.5:7b-instruct".to_string()
}

fn default_use_gpu() -> bool {
    true
}

fn default_language() -> String {
    "ru".to_string()
}
fn default_audio_dir() -> String {
    "audio".to_string()
}
fn default_sample_rate() -> u32 {
    16000
}
fn default_max_chunk_sec() -> f64 {
    15.0
}
fn default_min_chunk_sec() -> f64 {
    2.0
}
fn default_vad_silence_sec() -> f64 {
    1.0
}
fn default_overlap_sec() -> f64 {
    2.5
}
fn default_session_ttl_hours() -> f64 {
    24.0
}
fn default_audio_dir_max_mb() -> f64 {
    0.0
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_path: None,
            language: default_language(),
            audio_dir: default_audio_dir(),
            sample_rate: default_sample_rate(),
            max_chunk_duration_sec: default_max_chunk_sec(),
            min_chunk_duration_sec: default_min_chunk_sec(),
            vad_silence_sec: default_vad_silence_sec(),
            overlap_sec: default_overlap_sec(),
            use_gpu: default_use_gpu(),
            cuda_path: None,
            gigaam_model_dir: None,
            ensemble_enabled: default_ensemble_enabled(),
            llm_correction_enabled: false,
            ollama_url: default_ollama_url(),
            llm_model: default_llm_model(),
            session_ttl_hours: default_session_ttl_hours(),
            audio_dir_max_mb: default_audio_dir_max_mb(),
        }
    }
}

fn settings_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("settings.json"));
            out.push(dir.join("..").join("server-reliable").join("settings.json"));
            out.push(dir.join("..").join("settings.json"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        out.push(cwd.join("server-reliable").join("settings.json"));
        out.push(cwd.join("settings.json"));
    }
    out.push(PathBuf::from("settings.json"));
    out.push(PathBuf::from("..").join("server-reliable").join("settings.json"));
    out.push(PathBuf::from("..").join("settings.json"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_default() {
        let s = Settings::default();
        assert_eq!(s.language, "ru");
        assert_eq!(s.sample_rate, 16000);
        assert_eq!(s.audio_dir, "audio");
        assert!(!s.ensemble_enabled);
        assert_eq!(s.ollama_url, "http://localhost:11434");
        assert_eq!(s.session_ttl_hours, 24.0);
        assert_eq!(s.audio_dir_max_mb, 0.0);
    }

    #[test]
    fn test_settings_parse_json() {
        let json = r#"{"model_path": "/path/model.bin", "language": "en", "ensemble_enabled": true}"#;
        let s: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(s.model_path.as_deref(), Some("/path/model.bin"));
        assert_eq!(s.language, "en");
        assert!(s.ensemble_enabled);
    }

    #[test]
    fn test_settings_parse_minimal() {
        let json = r#"{}"#;
        let s: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(s.language, "ru");
        assert_eq!(s.sample_rate, 16000);
    }
}

pub fn load_settings(model_path_override: Option<&str>) -> Result<Settings> {
    for path in settings_candidates() {
        if path.is_file() {
            let s = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            let mut settings: Settings = serde_json::from_str(&s)
                .with_context(|| format!("parse {}", path.display()))?;
            if let Some(p) = model_path_override {
                settings.model_path = Some(p.to_string());
            } else if settings.model_path.as_deref().unwrap_or("").trim().is_empty() {
                settings.model_path = None;
            }
            return Ok(settings);
        }
    }
    let mut settings = Settings::default();
    if let Some(p) = model_path_override {
        settings.model_path = Some(p.to_string());
    }
    Ok(settings)
}
