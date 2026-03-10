//! Shared types and utilities for transcription pipeline.

use crossbeam_channel::Sender;
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use whisper_rs::{WhisperContext, WhisperContextParameters};

/// Messages sent to the client.
#[derive(Clone)]
pub enum ClientMessage {
    Transcript {
        text: String,
        source: u8,
        start_sec: f64,
        end_sec: f64,
    },
    Status(serde_json::Value),
    /// Signals that all queued segments have been processed (for batch/YouTube clients).
    Done,
}

/// Checkpoint after Whisper+GigaAM (Stage 0 -> 1).
#[derive(Serialize, Deserialize, Clone)]
pub struct AsrResult {
    pub whisper_text: String,
    pub gigaam_text: String,
    pub merged_text: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub source_id: u8,
    pub seg_id: String,
}

/// Task sent to LLM pool. output_sink allows global dispatcher to route to correct session.
/// llm_inflight: when LLM completes, fetch_sub(1) for wait_until_empty.
pub struct LlmTask {
    pub asr: AsrResult,
    pub context_snapshot: String,
    pub asr_path: PathBuf,
    pub wav_path: PathBuf,
    pub meta_path: PathBuf,
    pub output_sink: Arc<OutputSink>,
    pub llm_inflight: Arc<AtomicUsize>,
}

/// Dual-write: transcript.jsonl (always) + WebSocket (if client connected).
pub struct OutputSink {
    transcript_path: PathBuf,
    client_tx: Arc<Mutex<Option<Sender<ClientMessage>>>>,
}

impl OutputSink {
    pub fn new(transcript_path: PathBuf, client_tx: Option<Sender<ClientMessage>>) -> Self {
        Self {
            transcript_path,
            client_tx: Arc::new(Mutex::new(client_tx)),
        }
    }

    /// For Transcript/Done: append to transcript.jsonl, then forward to client.
    pub fn send(&self, msg: &ClientMessage) {
        match msg {
            ClientMessage::Transcript { text, source, start_sec, end_sec } => {
                let line = serde_json::json!({
                    "text": text,
                    "source": source,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                });
                if let Ok(mut f) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.transcript_path)
                {
                    let _ = writeln!(f, "{}", line);
                    let _ = f.flush();
                }
            }
            ClientMessage::Done => {
                let line = r#"{"type":"done"}"#;
                if let Ok(mut f) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.transcript_path)
                {
                    let _ = writeln!(f, "{}", line);
                    let _ = f.flush();
                }
            }
            ClientMessage::Status(_) => {}
        }
        if let Ok(guard) = self.client_tx.lock() {
            if let Some(ref tx) = *guard {
                let _ = tx.send(msg.clone());
            }
        }
    }

    /// Status messages: only forward to client (ephemeral).
    pub fn send_status(&self, status: serde_json::Value) {
        if let Ok(guard) = self.client_tx.lock() {
            if let Some(ref tx) = *guard {
                let _ = tx.send(ClientMessage::Status(status));
            }
        }
    }

    pub fn clear_client(&self) {
        if let Ok(mut guard) = self.client_tx.lock() {
            *guard = None;
        }
    }

    /// Replay transcript.jsonl to client, then set new sender.
    pub fn replay_to_client(&self, tx: Sender<ClientMessage>) {
        let mut guard = self.client_tx.lock().unwrap();
        if let Ok(content) = std::fs::read_to_string(&self.transcript_path) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if v.get("type").and_then(|t| t.as_str()) == Some("done") {
                        let _ = tx.send(ClientMessage::Done);
                    } else if let (Some(text), Some(source), Some(start), Some(end)) = (
                        v.get("text").and_then(|x| x.as_str()),
                        v.get("source").and_then(|x| x.as_u64()),
                        v.get("start_sec").and_then(|x| x.as_f64()),
                        v.get("end_sec").and_then(|x| x.as_f64()),
                    ) {
                        let _ = tx.send(ClientMessage::Transcript {
                            text: text.to_string(),
                            source: source as u8,
                            start_sec: start,
                            end_sec: end,
                        });
                    }
                }
            }
        }
        *guard = Some(tx);
    }
}

pub struct WorkerState {
    pub prev_tail: [String; 2],
    pub last_proc_sec: [f64; 2],
    pub last_audio_sec: [f64; 2],
    pub queue_len: [usize; 2],
    pub worker_busy: [bool; 2],
    pub context_lines: [VecDeque<String>; 2],
}

pub struct TranscribedEnd(Arc<TranscribedEndInner>);

struct TranscribedEndInner {
    sec_0: std::sync::atomic::AtomicU64,
    sec_1: std::sync::atomic::AtomicU64,
}

impl Default for TranscribedEndInner {
    fn default() -> Self {
        Self {
            sec_0: std::sync::atomic::AtomicU64::new(0),
            sec_1: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl Default for TranscribedEnd {
    fn default() -> Self {
        Self(Arc::new(TranscribedEndInner::default()))
    }
}

impl TranscribedEnd {
    pub fn get(&self, source: u8) -> f64 {
        let bits = if source == 0 {
            self.0.sec_0.load(Ordering::Relaxed)
        } else {
            self.0.sec_1.load(Ordering::Relaxed)
        };
        f64::from_bits(bits)
    }
    pub fn set(&self, source: u8, sec: f64) {
        let bits = sec.to_bits();
        if source == 0 {
            self.0.sec_0.store(bits, Ordering::Relaxed);
        } else {
            self.0.sec_1.store(bits, Ordering::Relaxed);
        }
    }
}

impl Clone for TranscribedEnd {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

pub fn load_whisper(model_path: &str, use_gpu: bool) -> anyhow::Result<WhisperContext> {
    let mut params = WhisperContextParameters::default();
    params.use_gpu = use_gpu;
    let backend = if use_gpu { "GPU (CUDA)" } else { "CPU" };
    tracing::info!("Whisper backend: {}", backend);
    let context = WhisperContext::new_with_params(model_path, params)
        .map_err(|e| anyhow::anyhow!("WhisperContext::new_with_params: {:?}", e))?;
    tracing::info!("Whisper loaded (requested: {})", backend);
    Ok(context)
}

pub fn wav_to_f32(path: &Path) -> anyhow::Result<Vec<f32>> {
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
    Ok(samples)
}

pub fn emit_status(state_shared: &Arc<Mutex<WorkerState>>, output_sink: &OutputSink, device: &str) {
    let st = state_shared.lock().unwrap();
    let status = serde_json::json!({
        "type": "status",
        "recording": true,
        "device": device,
        "task_queue_size": st.queue_len[0],
        "task_queue_2_size": st.queue_len[1],
        "pending_1": st.queue_len[0] + if st.worker_busy[0] { 1 } else { 0 },
        "pending_2": st.queue_len[1] + if st.worker_busy[1] { 1 } else { 0 },
        "worker_busy": st.worker_busy[0],
        "worker_2_busy": st.worker_busy[1],
        "worker_last_proc_sec": st.last_proc_sec[0],
        "worker_2_last_proc_sec": st.last_proc_sec[1],
        "worker_last_audio_sec": st.last_audio_sec[0],
        "worker_2_last_audio_sec": st.last_audio_sec[1],
    });
    output_sink.send_status(status);
}

