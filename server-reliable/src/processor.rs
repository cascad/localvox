//! Shared types and utilities for transcription pipeline.

use crossbeam_channel::Sender;
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
/// One model's output in a transcript batch (when correction disabled + N models).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TranscriptVariant {
    pub model: String,
    pub text: String,
}

/// Messages sent to the client.
#[derive(Clone)]
pub enum ClientMessage {
    Transcript {
        text: String,
        source: u8,
        start_sec: f64,
        end_sec: f64,
        /// Segment ID for batch grouping (e.g. for later LLM processing).
        seg_id: String,
        /// When correction disabled + N models: all variants labeled by model.
        #[allow(clippy::type_complexity)]
        variants: Option<Vec<TranscriptVariant>>,
        /// Monotonic sequence number assigned by OutputSink.
        seq: u64,
    },
    Status(serde_json::Value),
    /// Signals that all queued segments have been processed (for batch/YouTube clients).
    Done,
}

/// Output from a single ASR model. Re-exported from asr for use in processor.
pub use crate::asr::ModelOutput;

/// Checkpoint after ASR (Stage 0 -> 1). Supports 1+ models; ensemble = 2+.
#[derive(Serialize, Deserialize, Clone)]
pub struct AsrResult {
    pub model_outputs: Vec<ModelOutput>,
    pub merged_text: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub source_id: u8,
    pub seg_id: String,
}

/// Task sent to LLM pool. output_sink allows global dispatcher to route to correct session.
/// prev_tail: tail of last SENT segment (for algorithmic merge_overlap after LLM).
/// worker_state: to update prev_tail after we emit.
pub struct LlmTask {
    pub asr: AsrResult,
    pub context_snapshot: String,
    pub prev_tail: String,
    pub worker_state: Arc<Mutex<WorkerState>>,
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
    next_seq: AtomicU64,
}

impl OutputSink {
    pub fn new(transcript_path: PathBuf, client_tx: Option<Sender<ClientMessage>>) -> Self {
        Self {
            transcript_path,
            client_tx: Arc::new(Mutex::new(client_tx)),
            next_seq: AtomicU64::new(0),
        }
    }

    /// Restore seq counter (e.g. after reading existing transcript.jsonl on startup).
    pub fn set_next_seq(&self, val: u64) {
        self.next_seq.store(val, Ordering::Relaxed);
    }

    /// For Transcript/Done: append to transcript.jsonl, then forward to client.
    pub fn send(&self, msg: &ClientMessage) {
        match msg {
            ClientMessage::Transcript { text, source, start_sec, end_sec, seg_id, variants, .. } => {
                let seq = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
                let mut line = serde_json::json!({
                    "seq": seq,
                    "text": text,
                    "source": source,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "seg_id": seg_id,
                });
                if let Some(ref v) = variants {
                    line["variants"] = serde_json::to_value(v).unwrap_or_default();
                }
                if let Ok(mut f) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.transcript_path)
                {
                    let _ = writeln!(f, "{}", line);
                    let _ = f.flush();
                }
                let msg_with_seq = ClientMessage::Transcript {
                    text: text.clone(),
                    source: *source,
                    start_sec: *start_sec,
                    end_sec: *end_sec,
                    seg_id: seg_id.clone(),
                    variants: variants.clone(),
                    seq,
                };
                if let Ok(guard) = self.client_tx.lock() {
                    if let Some(ref tx) = *guard {
                        let _ = tx.send(msg_with_seq);
                    }
                }
                return;
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

    /// Replay transcript.jsonl to client, then set new sender (for live resume).
    /// Skips entries with seq ≤ skip_to_seq (0 = replay all).
    pub fn replay_to_client(&self, tx: Sender<ClientMessage>, skip_to_seq: u64) {
        self.replay_to_client_inner(tx, true, skip_to_seq);
    }

    /// Replay transcript.jsonl to client only; do not register sender (for completed session).
    pub fn replay_to_client_no_register(&self, tx: Sender<ClientMessage>, skip_to_seq: u64) {
        self.replay_to_client_inner(tx, false, skip_to_seq);
    }

    fn replay_to_client_inner(&self, tx: Sender<ClientMessage>, register: bool, skip_to_seq: u64) {
        let mut max_seq: u64 = 0;
        let mut replayed: u64 = 0;
        let mut skipped: u64 = 0;
        if let Ok(content) = std::fs::read_to_string(&self.transcript_path) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if v.get("type").and_then(|t| t.as_str()) == Some("done") {
                        let _ = tx.send(ClientMessage::Done);
                        continue;
                    }
                    let seq = v.get("seq").and_then(|x| x.as_u64()).unwrap_or(0);
                    if seq > max_seq {
                        max_seq = seq;
                    }
                    if seq > 0 && seq <= skip_to_seq {
                        skipped += 1;
                        continue;
                    }
                    if let (Some(text), Some(source), Some(start), Some(end)) = (
                        v.get("text").and_then(|x| x.as_str()),
                        v.get("source").and_then(|x| x.as_u64()),
                        v.get("start_sec").and_then(|x| x.as_f64()),
                        v.get("end_sec").and_then(|x| x.as_f64()),
                    ) {
                        let variants = v.get("variants")
                            .and_then(|a| a.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|o| {
                                        Some(TranscriptVariant {
                                            model: o.get("model")?.as_str()?.to_string(),
                                            text: o.get("text")?.as_str()?.to_string(),
                                        })
                                    })
                                    .collect::<Vec<_>>()
                            });
                        let seg_id = v.get("seg_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
                        let _ = tx.send(ClientMessage::Transcript {
                            text: text.to_string(),
                            source: source as u8,
                            start_sec: start,
                            end_sec: end,
                            seg_id,
                            variants,
                            seq,
                        });
                        replayed += 1;
                    }
                }
            }
        }
        if max_seq > 0 {
            self.next_seq.store(max_seq, Ordering::Relaxed);
        }
        if skip_to_seq > 0 {
            tracing::info!(
                "Replay: skipped {} (seq ≤ {}), sent {} new entries",
                skipped, skip_to_seq, replayed
            );
        }
        if register {
            if let Ok(mut guard) = self.client_tx.lock() {
                *guard = Some(tx);
            }
        }
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

