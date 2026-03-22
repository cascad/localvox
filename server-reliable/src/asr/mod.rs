//! Modular ASR adapter system. Trait-based pluggable models (Whisper, GigaAM, Parakeet, etc.)
//! selected at runtime via config. Ensemble = 2+ models.

mod gigaam;
mod parakeet;
mod whisper;

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

use crate::config::Settings;

pub use gigaam::GigaAmAdapter;
#[allow(unused_imports)] // Kept for when Parakeet is configured
pub use parakeet::ParakeetAdapter;
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
    fn transcribe(&self, wav_path: &Path, samples: &[f32], language: &str) -> Result<String>;
    fn filter_hallucinations(&self, text: &str) -> String;
}

/// Run a single inference on silence to warm up CUDA / ONNX before real traffic.
///
/// Whisper (`whisper-rs`) warns and effectively skips work if audio is shorter than **1 s**; we use **1.2 s**
/// so the GPU path actually runs. With several models (ensemble), startup waits for **each** warmup
/// in sequence — that cost moves from “first segment” to “server boot”, it does not disappear.
pub fn warmup(models: &[Arc<dyn AsrModel>]) {
    use std::time::Instant;

    let sample_rate = 16000u32;
    // ≥1 s required for whisper_full; a bit extra avoids edge rounding to 999 ms in WAV.
    let duration_sec = 1.2f64;
    let duration_samples = ((sample_rate as f64) * duration_sec).round() as usize;
    let silence: Vec<f32> = vec![0.0; duration_samples];

    let tmp_dir = std::env::temp_dir().join("localvox_warmup");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let wav_path = tmp_dir.join("warmup.wav");

    // GigaAM/Parakeet read from file, so write a tiny WAV
    if let Ok(mut writer) = hound::WavWriter::create(
        &wav_path,
        hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    ) {
        for &s in &silence {
            let _ = writer.write_sample((s * 32767.0) as i16);
        }
        let _ = writer.finalize();
    }

    for model in models {
        let t0 = Instant::now();
        let name = model.name();
        tracing::info!("Warmup: {name} …");
        match model.transcribe(&wav_path, &silence, "ru") {
            Ok(_) => {
                tracing::info!("Warmup: {name} ready ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            Err(e) => {
                tracing::warn!("Warmup: {name} failed: {e}");
            }
        }
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
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
                Arc::new(GigaAmAdapter::new(
                    std::path::Path::new(&entry.model_path),
                    use_gpu,
                )?)
            }
            "parakeet" => {
                let use_gpu = entry.use_gpu.unwrap_or(settings.use_gpu);
                Arc::new(parakeet::ParakeetAdapter::new(
                    std::path::Path::new(&entry.model_path),
                    use_gpu,
                )?)
            }
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
