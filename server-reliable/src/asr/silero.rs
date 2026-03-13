//! Silero STT ASR adapter via ONNX Runtime (ort crate).
//! Model: snakers4/silero-models STT, exported to ONNX.
//! Requires: model .onnx file + tokens.txt (one token per line, blank = "_").

use anyhow::{Context, Result};
use ndarray::{ArrayViewD, Axis};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

use crate::hallucination;

/// Silero STT adapter (ONNX + CTC greedy decoder).
pub struct SileroAdapter {
    session: Mutex<Session>,
    labels: Vec<String>,
    blank_idx: usize,
    space_idx: usize,
}

unsafe impl Send for SileroAdapter {}
unsafe impl Sync for SileroAdapter {}

impl SileroAdapter {
    pub fn new(model_path: &str, tokens_path: &str) -> Result<Self> {
        if model_path.is_empty() {
            anyhow::bail!("Silero model_path required");
        }
        let path = Path::new(model_path);
        if !path.is_file() {
            anyhow::bail!("Silero model not found: {}", model_path);
        }

        let session = Session::builder()
            .context("ort Session::builder")?
            .commit_from_file(path)
            .context("ort commit_from_file")?;

        let labels = if tokens_path.is_empty() {
            anyhow::bail!("Silero tokens_path required (tokens.txt with one token per line)");
        } else {
            let content = std::fs::read_to_string(tokens_path)
                .with_context(|| format!("read tokens {}", tokens_path))?;
            content
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        };

        let blank_idx = labels
            .iter()
            .position(|s| s == "_")
            .ok_or_else(|| anyhow::anyhow!("tokens must contain '_' (blank)"))?;
        let space_idx = labels
            .iter()
            .position(|s| s == " ")
            .ok_or_else(|| anyhow::anyhow!("tokens must contain ' ' (space)"))?;

        tracing::info!("Silero loaded: {} ({} tokens)", model_path, labels.len());
        Ok(Self {
            session: Mutex::new(session),
            labels,
            blank_idx,
            space_idx,
        })
    }

    fn ctc_greedy_decode(&self, probs: ArrayViewD<'_, f32>) -> String {
        let mut result = Vec::new();
        let mut prev_idx = self.blank_idx;

        for row in probs.outer_iter() {
            let idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(self.blank_idx);

            if idx == self.blank_idx {
                prev_idx = idx;
                continue;
            }
            if idx == prev_idx && idx != self.space_idx {
                prev_idx = idx;
                continue;
            }
            if self.labels[idx] == "2" {
                if let Some(&last) = result.last() {
                    result.push(last);
                }
                prev_idx = idx;
                continue;
            }
            result.push(idx);
            prev_idx = idx;
        }

        result
            .into_iter()
            .map(|i| self.labels[i].as_str())
            .collect::<Vec<_>>()
            .join("")
            .replace("  ", " ")
            .trim()
            .to_string()
    }

    fn transcribe_inner(&self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let seq_len = samples.len();
        let tensor_input = Tensor::from_array(([1, seq_len], samples.to_vec()))
            .context("Tensor::from_array")?;

        let mut session = self.session.lock().map_err(|_| anyhow::anyhow!("lock"))?;
        let outputs = session
            .run(ort::inputs!["input" => tensor_input])
            .context("ort run")?;

        let output = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .context("no output")?;
        let arr = output
            .try_extract_array::<f32>()
            .context("extract tensor")?;

        let ndim = arr.ndim();
        let probs: ndarray::ArrayD<f32> = if ndim == 2 {
            arr.to_owned()
        } else if ndim == 3 {
            arr.index_axis(Axis(0), 0).to_owned().into_dyn()
        } else {
            anyhow::bail!("unexpected output shape: {:?}", arr.shape());
        };

        let text = self.ctc_greedy_decode(probs.view());
        Ok(text)
    }
}

impl super::AsrModel for SileroAdapter {
    fn name(&self) -> &str {
        "silero"
    }

    fn backend(&self) -> &str {
        "CPU"
    }

    fn transcribe(
        &self,
        wav_path: &Path,
        samples: &[f32],
        _language: &str,
    ) -> Result<String> {
        let samples: Vec<f32> = if samples.is_empty() {
            crate::processor::wav_to_f32(wav_path)?
        } else {
            samples.to_vec()
        };
        self.transcribe_inner(&samples)
    }

    fn filter_hallucinations(&self, text: &str) -> String {
        hallucination::filter_silero(text)
    }
}
