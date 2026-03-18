//! GigaAM v3 CTC ASR via sherpa-onnx (NeMo CTC).
//! Model: sherpa-onnx-nemo-ctc-giga-am-v3-russian.

use anyhow::{Context, Result};
use hound::WavReader;
use sherpa_rs::sherpa_rs_sys as sys;
use std::ffi::CString;
use std::path::Path;

use crate::hallucination;

/// GigaAM CTC recognizer (sherpa-onnx NeMo).
pub struct GigaAmAdapter {
    recognizer: *const sys::SherpaOnnxOfflineRecognizer,
    backend: &'static str,
}

unsafe impl Send for GigaAmAdapter {}
unsafe impl Sync for GigaAmAdapter {}

impl GigaAmAdapter {
    /// Loads model from directory (model.int8.onnx + tokens.txt).
    /// use_gpu: if true, uses "cuda" provider (requires sherpa-rs built with cuda feature).
    pub fn new(model_dir: &Path, use_gpu: bool) -> Result<Self> {
        let model_path = model_dir.join("model.int8.onnx");
        let tokens_path = model_dir.join("tokens.txt");
        if !model_path.is_file() {
            anyhow::bail!(
                "GigaAM model not found: {}. Run tools/download_gigaam.ps1",
                model_path.display()
            );
        }
        if !tokens_path.is_file() {
            anyhow::bail!("GigaAM tokens not found: {}", tokens_path.display());
        }

        let model_c =
            CString::new(model_path.to_string_lossy().as_bytes()).context("model path")?;
        let tokens_c =
            CString::new(tokens_path.to_string_lossy().as_bytes()).context("tokens path")?;
        let provider = if use_gpu { "cuda" } else { "cpu" };
        let provider_c = CString::new(provider).context("provider")?;
        let decoding_c = CString::new("greedy_search").context("decoding")?;

        let nemo_ctc = sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig {
            model: model_c.as_ptr(),
        };

        let mut model_config = unsafe { std::mem::zeroed::<sys::SherpaOnnxOfflineModelConfig>() };
        model_config.nemo_ctc = nemo_ctc;
        model_config.tokens = tokens_c.as_ptr();
        model_config.num_threads = 2;
        model_config.debug = 0;
        model_config.provider = provider_c.as_ptr();

        let feat_config = sys::SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 64,
        };

        let lm_config = unsafe { std::mem::zeroed::<sys::SherpaOnnxOfflineLMConfig>() };
        let recognizer_config = sys::SherpaOnnxOfflineRecognizerConfig {
            feat_config,
            model_config,
            lm_config,
            decoding_method: decoding_c.as_ptr(),
            max_active_paths: 4,
            hotwords_file: std::ptr::null(),
            hotwords_score: 1.5,
            rule_fsts: std::ptr::null(),
            rule_fars: std::ptr::null(),
            blank_penalty: 0.0,
            hr: unsafe { std::mem::zeroed() },
        };

        let recognizer = unsafe { sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            anyhow::bail!("SherpaOnnxCreateOfflineRecognizer failed");
        }

        let backend = if use_gpu { "GPU" } else { "CPU" };
        tracing::info!(
            "GigaAM loaded: {} (provider: {})",
            model_dir.display(),
            provider
        );
        Ok(Self {
            recognizer,
            backend,
        })
    }

    fn transcribe_inner(&self, wav_path: &Path) -> Result<String> {
        let (samples, sample_rate) = wav_to_f32(wav_path)?;
        if samples.is_empty() {
            return Ok(String::new());
        }

        let text = unsafe {
            let stream = sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw = result_ptr.read();
            let text = if raw.text.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(raw.text)
                    .to_string_lossy()
                    .into_owned()
            };
            sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sys::SherpaOnnxDestroyOfflineStream(stream);
            text
        };

        Ok(text.trim().to_string())
    }
}

impl Drop for GigaAmAdapter {
    fn drop(&mut self) {
        unsafe {
            sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}

impl super::AsrModel for GigaAmAdapter {
    fn name(&self) -> &str {
        "gigaam"
    }

    fn backend(&self) -> &str {
        self.backend
    }

    fn transcribe(&self, wav_path: &Path, _samples: &[f32], _language: &str) -> Result<String> {
        self.transcribe_inner(wav_path)
    }

    fn filter_hallucinations(&self, text: &str) -> String {
        hallucination::filter_gigaam(text)
    }
}

fn wav_to_f32(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        anyhow::bail!("expected 16-bit mono WAV");
    }
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s as f32 / 32768.0)
        .collect();
    Ok((samples, spec.sample_rate))
}
