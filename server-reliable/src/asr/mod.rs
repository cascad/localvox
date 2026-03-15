//! Modular ASR adapter system. Trait-based pluggable models (Whisper, GigaAM, Silero, etc.)
//! selected at runtime via config. Ensemble = 2+ models.

mod gigaam;
mod parakeet;
mod silero;
mod whisper;

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

use crate::config::Settings;

pub use gigaam::GigaAmAdapter;
pub use parakeet::ParakeetAdapter;
pub use silero::SileroAdapter;
pub use whisper::WhisperAdapter;

/// Output from a single ASR model. Used in AsrResult and for ensemble merge.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelOutput {
    pub model_name: String,
    pub text: String,
}

/// Common interface for all ASR model adapters.
pub trait AsrModel: Send + Sync {
    fn name(&self) -> &str;
    /// Returns "CPU" or "GPU" depending on backend.
    fn backend(&self) -> &str;
    fn transcribe(
        &self,
        wav_path: &Path,
        samples: &[f32],
        language: &str,
    ) -> Result<String>;
    fn filter_hallucinations(&self, text: &str) -> String;
}

/// Loads all configured ASR models from settings. Returns empty vec if none configured.
pub fn load_models(settings: &Settings) -> Result<Vec<Arc<dyn AsrModel>>> {
    let entries = settings.resolved_models();
    let mut models: Vec<Arc<dyn AsrModel>> = Vec::new();

    for entry in entries {
        let adapter: Arc<dyn AsrModel> = match entry.model_type.to_lowercase().as_str() {
            "whisper" => {
                let use_gpu = entry.use_gpu.unwrap_or(settings.use_gpu);
                Arc::new(WhisperAdapter::new(
                    &entry.model_path,
                    use_gpu,
                    settings.whisper_no_speech_thold,
                )?)
            }
            "gigaam" => {
                let use_gpu = entry.use_gpu.unwrap_or(settings.use_gpu);
                Arc::new(GigaAmAdapter::new(std::path::Path::new(&entry.model_path), use_gpu)?)
            }
            "parakeet" => {
                let use_gpu = entry.use_gpu.unwrap_or(settings.use_gpu);
                Arc::new(parakeet::ParakeetAdapter::new(std::path::Path::new(&entry.model_path), use_gpu)?)
            }
            "silero" => Arc::new(silero::SileroAdapter::new(
                &entry.model_path,
                entry.tokens_path.as_deref().unwrap_or(""),
            )?),
            other => {
                tracing::warn!("Unknown ASR model type '{}', skipping", other);
                continue;
            }
        };
        tracing::info!("Loaded ASR model: {}", adapter.name());
        models.push(adapter);
    }

    Ok(models)
}
