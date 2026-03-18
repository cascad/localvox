//! Whisper ASR adapter via whisper-rs (GGML).

use anyhow::Result;
use std::path::Path;
use std::sync::Mutex;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::hallucination;

/// Whisper adapter with internal state pool for thread-safe parallel transcription.
pub struct WhisperAdapter {
    context: WhisperContext,
    state_pool: Mutex<Vec<WhisperState>>,
    use_gpu: bool,
    no_speech_thold: f32,
}

impl WhisperAdapter {
    pub fn new(model_path: &str, use_gpu: bool, no_speech_thold: f32) -> Result<Self> {
        let params = WhisperContextParameters {
            use_gpu,
            ..Default::default()
        };
        let backend = if use_gpu { "GPU (CUDA)" } else { "CPU" };
        tracing::info!("Whisper: provider {}", backend);
        let context = WhisperContext::new_with_params(model_path, params)
            .map_err(|e| anyhow::anyhow!("WhisperContext::new_with_params: {:?}", e))?;

        let state = context
            .create_state()
            .map_err(|e| anyhow::anyhow!("WhisperContext::create_state: {:?}", e))?;
        let state_pool = Mutex::new(vec![state]);

        Ok(Self {
            context,
            state_pool,
            use_gpu,
            no_speech_thold: no_speech_thold.clamp(0.0, 1.0),
        })
    }

    fn get_or_create_state(&self) -> Result<WhisperState> {
        let mut pool = self
            .state_pool
            .lock()
            .map_err(|_| anyhow::anyhow!("lock"))?;
        if let Some(state) = pool.pop() {
            return Ok(state);
        }
        drop(pool);
        let state = self
            .context
            .create_state()
            .map_err(|e| anyhow::anyhow!("WhisperContext::create_state: {:?}", e))?;
        Ok(state)
    }

    fn return_state(&self, state: WhisperState) {
        if let Ok(mut pool) = self.state_pool.lock() {
            pool.push(state);
        }
    }
}

impl super::AsrModel for WhisperAdapter {
    fn name(&self) -> &str {
        "whisper"
    }

    fn backend(&self) -> &str {
        if self.use_gpu {
            "GPU"
        } else {
            "CPU"
        }
    }

    fn transcribe(&self, _wav_path: &Path, samples: &[f32], language: &str) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut state = self.get_or_create_state()?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(true);
        params.set_no_speech_thold(self.no_speech_thold);
        params.set_suppress_non_speech_tokens(true);
        if !language.is_empty() && language != "auto" {
            params.set_language(Some(language));
        }

        state
            .full(params, samples)
            .map_err(|e| anyhow::anyhow!("whisper full: {:?}", e))?;
        let n = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("full_n_segments: {:?}", e))?;
        let mut out = String::new();
        for i in 0..n {
            if let Ok(seg) = state.full_get_segment_text(i) {
                out.push_str(&seg);
            }
        }

        self.return_state(state);
        Ok(out.trim().to_string())
    }

    fn filter_hallucinations(&self, text: &str) -> String {
        hallucination::filter_whisper(text)
    }
}
