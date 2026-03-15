//! Load settings from settings.json.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

/// Single ASR model entry. Used in the new `models` array or synthesized from legacy config.
#[derive(Clone, Debug, Deserialize)]
pub struct ModelEntry {
    #[serde(rename = "type")]
    pub model_type: String, // "whisper" | "gigaam" | "silero"
    pub model_path: String,
    #[serde(default)]
    pub tokens_path: Option<String>,
    #[serde(default)]
    pub use_gpu: Option<bool>,
}

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
    /// LLM prompt template when only Whisper (no GigaAM). Placeholders: {context_str}, {merged_text}
    #[serde(default = "default_llm_prompt_single")]
    pub llm_prompt_single: String,
    /// LLM prompt template for ensemble (Whisper + GigaAM). Placeholders: {context_str}, {merged_text}, {whisper_text}, {gigaam_text}
    #[serde(default = "default_llm_prompt_ensemble")]
    pub llm_prompt_ensemble: String,
    /// LLM request timeout in seconds. Default 10.
    #[serde(default = "default_llm_timeout_sec")]
    pub llm_timeout_sec: u64,
    /// Number of parallel LLM correction workers (global). Default 2.
    #[serde(default = "default_llm_pool_size")]
    pub llm_pool_size: usize,
    /// Global ASR worker threads. Default 1.
    #[serde(default = "default_asr_workers")]
    pub asr_workers: usize,
    /// Live session reconnect timeout (seconds). Default 300.
    #[serde(default = "default_live_reconnect_timeout_sec")]
    pub live_reconnect_timeout_sec: u64,
    /// How long to keep completed sessions in registry (seconds). Default 3600.
    #[serde(default = "default_session_idle_timeout_sec")]
    pub session_idle_timeout_sec: u64,
    /// Удалять сессии старше N часов (0 = отключено).
    #[serde(default = "default_session_ttl_hours")]
    pub session_ttl_hours: f64,
    /// Макс. размер audio_dir в MB (0 = без лимита). При превышении удаляются старые сессии.
    #[serde(default = "default_audio_dir_max_mb")]
    pub audio_dir_max_mb: f64,
    /// New models array. If absent, synthesized from model_path + gigaam_model_dir + ensemble_enabled.
    #[serde(default)]
    pub models: Option<Vec<ModelEntry>>,
    /// Whisper no_speech_thold (0..1). Ниже = чувствительнее к тихому микрофону. Default 0.4.
    #[serde(default = "default_whisper_no_speech_thold")]
    pub whisper_no_speech_thold: f32,
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
fn default_llm_prompt_single() -> String {
    "Исправь ошибки распознавания речи в тексте.\n\
     Контекст предыдущих фраз: {context_str}\n\
     Текст: \"{merged_text}\"\n\
     Важно: отвечай ТОЛЬКО на русском языке, не переводи на другие языки.\n\
     Верни только исправленный текст, без кавычек и пояснений.".to_string()
}
fn default_llm_prompt_ensemble() -> String {
    "Даны варианты распознавания одного аудиофрагмента.\n\
     {model_variants}\n\
     Алгоритмический мерж: \"{merged_text}\"\n\
     Контекст предыдущих фраз: {context_str}\n\n\
     Выбери лучший вариант или объедини их. Исправь явные ошибки распознавания.\n\
     Важно: отвечай ТОЛЬКО на русском языке, не переводи на другие языки.\n\
     Верни только исправленный текст, без кавычек и пояснений.".to_string()
}

fn default_llm_timeout_sec() -> u64 {
    10
}
fn default_llm_pool_size() -> usize {
    2
}
fn default_asr_workers() -> usize {
    1
}
fn default_live_reconnect_timeout_sec() -> u64 {
    300
}
fn default_session_idle_timeout_sec() -> u64 {
    3600
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
fn default_whisper_no_speech_thold() -> f32 {
    0.4
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_path: None,
            models: None,
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
            llm_prompt_single: default_llm_prompt_single(),
            llm_prompt_ensemble: default_llm_prompt_ensemble(),
            llm_timeout_sec: default_llm_timeout_sec(),
            llm_pool_size: default_llm_pool_size(),
            asr_workers: default_asr_workers(),
            live_reconnect_timeout_sec: default_live_reconnect_timeout_sec(),
            session_idle_timeout_sec: default_session_idle_timeout_sec(),
            session_ttl_hours: default_session_ttl_hours(),
            audio_dir_max_mb: default_audio_dir_max_mb(),
            whisper_no_speech_thold: default_whisper_no_speech_thold(),
        }
    }
}

impl Settings {
    /// Returns the effective list of models to load. If `models` array is present, uses it.
    /// Otherwise synthesizes from legacy fields: model_path -> whisper, ensemble_enabled + gigaam_model_dir -> gigaam.
    pub fn resolved_models(&self) -> Vec<ModelEntry> {
        if let Some(ref models) = self.models {
            if !models.is_empty() {
                return models.clone();
            }
        }
        let mut out = Vec::new();
        if let Some(ref p) = self.model_path {
            if !p.trim().is_empty() {
                out.push(ModelEntry {
                    model_type: "whisper".to_string(),
                    model_path: p.clone(),
                    tokens_path: None,
                    use_gpu: Some(self.use_gpu),
                });
            }
        }
        if self.ensemble_enabled {
            if let Some(ref dir) = self.gigaam_model_dir {
                if !dir.trim().is_empty() {
                    out.push(ModelEntry {
                        model_type: "gigaam".to_string(),
                        model_path: dir.clone(),
                        tokens_path: None,
                        use_gpu: None,
                    });
                }
            }
        }
        out
    }
}

fn settings_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(path) = std::env::var_os("LOCALVOX_SETTINGS") {
        out.push(PathBuf::from(path));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("settings.json"));
            out.push(dir.join("settings.docker.json"));
            out.push(dir.join("..").join("server-reliable").join("settings.json"));
            out.push(dir.join("..").join("server-reliable").join("settings.docker.json"));
            out.push(dir.join("..").join("settings.json"));
            out.push(dir.join("..").join("settings.docker.json"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        out.push(cwd.join("server-reliable").join("settings.json"));
        out.push(cwd.join("server-reliable").join("settings.docker.json"));
        out.push(cwd.join("settings.json"));
        out.push(cwd.join("settings.docker.json"));
    }
    out.push(PathBuf::from("settings.json"));
    out.push(PathBuf::from("settings.docker.json"));
    out.push(PathBuf::from("..").join("server-reliable").join("settings.json"));
    out.push(PathBuf::from("..").join("server-reliable").join("settings.docker.json"));
    out.push(PathBuf::from("..").join("settings.json"));
    out.push(PathBuf::from("..").join("settings.docker.json"));
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

    #[test]
    fn test_resolved_models_legacy() {
        let s = Settings {
            model_path: Some("/path/whisper.bin".to_string()),
            ensemble_enabled: true,
            gigaam_model_dir: Some("/path/gigaam".to_string()),
            ..Default::default()
        };
        let models = s.resolved_models();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].model_type, "whisper");
        assert_eq!(models[0].model_path, "/path/whisper.bin");
        assert_eq!(models[1].model_type, "gigaam");
        assert_eq!(models[1].model_path, "/path/gigaam");
    }

    #[test]
    fn test_resolved_models_new() {
        let json = r#"{"models": [{"type": "silero", "model_path": "/models/silero.onnx", "tokens_path": "/models/tokens.txt"}]}"#;
        let s: Settings = serde_json::from_str(json).unwrap();
        let models = s.resolved_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model_type, "silero");
        assert_eq!(models[0].model_path, "/models/silero.onnx");
        assert_eq!(models[0].tokens_path.as_deref(), Some("/models/tokens.txt"));
    }
}

pub fn load_settings(model_path_override: Option<&str>) -> Result<Settings> {
    for path in settings_candidates() {
        if path.is_file() {
            let s = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            let llm_on = serde_json::from_str::<serde_json::Value>(&s)
                .ok()
                .and_then(|v| v.get("llm_correction_enabled").and_then(|b| b.as_bool()))
                .unwrap_or(false);
            tracing::info!("Config: {} (llm_correction={})", path.display(), llm_on);
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
