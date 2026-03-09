//! File processor: queue WAV paths, transcribe with Whisper (+ GigaAM ensemble), merge overlap, send results.

use crate::gigaam::GigaAM;
use crate::hallucination;
use crate::llm_corrector::LlmCorrector;
use crate::transcript_postprocess::{ensemble_merge, process_segment};
use crossbeam_channel::{Receiver, Sender};
use hound::WavReader;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tracing::info;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

/// Messages sent to the client.
#[derive(Clone)]
pub enum ClientMessage {
    Transcript { text: String, source: u8 },
    Status(serde_json::Value),
}

struct WorkerState {
    prev_tail: [String; 2],
    last_proc_sec: [f64; 2],
    last_audio_sec: [f64; 2],
    queue_len: [usize; 2],
    worker_busy: [bool; 2],
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
    fn set(&self, source: u8, sec: f64) {
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

fn wav_to_f32(path: &Path) -> anyhow::Result<Vec<f32>> {
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

fn run_worker(
    rx: Receiver<(std::path::PathBuf, u8)>,
    state: Arc<Mutex<WhisperState>>,
    gigaam: Option<Arc<GigaAM>>,
    llm: Option<Arc<LlmCorrector>>,
    tx_out: Sender<ClientMessage>,
    state_shared: Arc<Mutex<WorkerState>>,
    transcribed_end: TranscribedEnd,
    language: String,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::Relaxed) {
        let (path, source_id) = match rx.recv() {
            Ok(p) => p,
            Err(_) => break,
        };

        {
            let mut st = state_shared.lock().unwrap();
            st.worker_busy[source_id as usize] = true;
            st.queue_len[source_id as usize] = st.queue_len[source_id as usize].saturating_sub(1);
        }
        emit_status(&state_shared, &tx_out, "ggml-whisper");

        let wav_path = path.clone();
        let meta_path = path.with_extension("meta.json");
        info!(
            "processing src{}: {}",
            source_id,
            wav_path.file_name().unwrap_or_default().to_string_lossy()
        );
        let result = (|| -> anyhow::Result<()> {
            if !wav_path.is_file() {
                return Ok(());
            }
            let samples = wav_to_f32(&wav_path)?;
            if samples.is_empty() {
                return Ok(());
            }
            let duration_sec = samples.len() as f64 / 16000.0;
            let t0 = Instant::now();

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_single_segment(true);
            params.set_no_speech_thold(0.6);
            params.set_suppress_non_speech_tokens(true);
            if !language.is_empty() && language != "auto" {
                params.set_language(Some(language.as_str()));
            }

            let prev_tail = {
                let st = state_shared.lock().unwrap();
                st.prev_tail[source_id as usize].clone()
            };

            let (whisper_text, gigaam_text) = if let Some(ref g) = gigaam {
                let state_clone = Arc::clone(&state);
                let samples_clone = samples.clone();
                let wav_clone = wav_path.clone();
                let lang = language.clone();
                let h_whisper = thread::spawn(move || -> anyhow::Result<String> {
                    let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                    p.set_print_progress(false);
                    p.set_print_realtime(false);
                    p.set_print_timestamps(false);
                    p.set_single_segment(true);
                    p.set_no_speech_thold(0.6);
                    p.set_suppress_non_speech_tokens(true);
                    if !lang.is_empty() && lang != "auto" {
                        p.set_language(Some(lang.as_str()));
                    }
                    let mut st = state_clone.lock().map_err(|_| anyhow::anyhow!("lock"))?;
                    st.full(p, &samples_clone)
                        .map_err(|e| anyhow::anyhow!("whisper full: {:?}", e))?;
                    let n = st.full_n_segments().map_err(|e| anyhow::anyhow!("full_n_segments: {:?}", e))?;
                    let mut out = String::new();
                    for i in 0..n {
                        if let Ok(seg) = st.full_get_segment_text(i) {
                            out.push_str(&seg);
                        }
                    }
                    Ok(out)
                });
                let gigaam_clone = Arc::clone(g);
                let h_gigaam = thread::spawn(move || gigaam_clone.transcribe(&wav_clone));
                let w = h_whisper.join().map_err(|_| anyhow::anyhow!("whisper panic"))??;
                let gg = h_gigaam.join().map_err(|_| anyhow::anyhow!("gigaam panic")).unwrap_or(Ok(String::new()))?;
                (w, gg)
            } else {
                let mut st = state.lock().map_err(|_| anyhow::anyhow!("lock"))?;
                st.full(params, &samples)
                    .map_err(|e| anyhow::anyhow!("whisper full: {:?}", e))?;
                let n = st.full_n_segments().map_err(|e| anyhow::anyhow!("full_n_segments: {:?}", e))?;
                let mut out = String::new();
                for i in 0..n {
                    if let Ok(seg) = st.full_get_segment_text(i) {
                        out.push_str(&seg);
                    }
                }
                (out, String::new())
            };

            let whisper_text = whisper_text.trim().to_string();
            let whisper_text = hallucination::filter_whisper(&whisper_text);

            let elapsed = t0.elapsed().as_secs_f64();
            {
                let mut st = state_shared.lock().unwrap();
                st.last_proc_sec[source_id as usize] = elapsed;
            }

            let gigaam_filtered = hallucination::filter_gigaam(&gigaam_text.trim().to_string());
            let text = if gigaam.is_some() && !gigaam_filtered.is_empty() {
                ensemble_merge(&whisper_text, &gigaam_filtered)
            } else {
                whisper_text.clone()
            };
            if text.is_empty() {
                info!("transcribe src{}: {:.1}s -> (no speech)", source_id, duration_sec);
            } else {
                let (merged, new_tail) = process_segment(&prev_tail, &text);
                {
                    let mut st = state_shared.lock().unwrap();
                    st.prev_tail[source_id as usize] = new_tail;
                    st.last_audio_sec[source_id as usize] = duration_sec;
                }
                if !merged.is_empty() {
                    let final_text = if let Some(ref llm_ref) = llm {
                        if let Some(corrected) = llm_ref.correct(&whisper_text, &gigaam_filtered, &merged) {
                            llm_ref.push_context(&corrected);
                            corrected
                        } else {
                            llm_ref.push_context(&merged);
                            merged
                        }
                    } else {
                        merged
                    };
                    let preview: String = final_text.chars().take(80).collect();
                    info!(
                        "transcribe src{}: {:.1}s -> {}",
                        source_id,
                        duration_sec,
                        preview,
                    );
                    let _ = tx_out.send(ClientMessage::Transcript {
                        text: final_text,
                        source: source_id,
                    });
                }
            }

            if let Ok(meta_bytes) = std::fs::read(&meta_path) {
                if let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&meta_bytes) {
                    if let Some(end_sec) = meta.get("end_time_sec").and_then(|v| v.as_f64()) {
                        transcribed_end.set(source_id, end_sec);
                    }
                }
            }
            let _ = std::fs::remove_file(&wav_path);
            let _ = std::fs::remove_file(&meta_path);
            Ok(())
        })();

        if let Err(e) = result {
            let _ = tx_out.send(ClientMessage::Transcript {
                text: format!("[error: {}]", e),
                source: source_id,
            });
            let _ = std::fs::remove_file(&wav_path);
            let _ = std::fs::remove_file(&meta_path);
        }

        {
            let mut st = state_shared.lock().unwrap();
            st.worker_busy[source_id as usize] = false;
        }
        emit_status(&state_shared, &tx_out, "ggml-whisper");
    }
}

fn emit_status(state_shared: &Arc<Mutex<WorkerState>>, tx_out: &Sender<ClientMessage>, device: &str) {
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
    let _ = tx_out.send(ClientMessage::Status(status));
}

pub struct FileProcessor {
    queue_tx: Sender<(std::path::PathBuf, u8)>,
    state: Arc<Mutex<WorkerState>>,
    pub transcribed_end: TranscribedEnd,
    running: Arc<AtomicBool>,
    _worker_handles: Vec<thread::JoinHandle<()>>,
}

impl FileProcessor {
    pub fn new(
        tx_out: Sender<ClientMessage>,
        whisper_state: Arc<Mutex<WhisperState>>,
        gigaam: Option<Arc<GigaAM>>,
        llm: Option<Arc<LlmCorrector>>,
        language: String,
    ) -> Self {
        let (queue_tx, queue_rx) = crossbeam_channel::unbounded();
        let state = Arc::new(Mutex::new(WorkerState {
            prev_tail: [String::new(), String::new()],
            last_proc_sec: [0.0, 0.0],
            last_audio_sec: [0.0, 0.0],
            queue_len: [0, 0],
            worker_busy: [false, false],
        }));
        let transcribed_end = TranscribedEnd::default();
        let running = Arc::new(AtomicBool::new(true));

        let state_worker = Arc::clone(&state);
        let transcribed_end_worker = transcribed_end.clone();
        let running_worker = Arc::clone(&running);
        let h0 = thread::spawn(move || {
            run_worker(
                queue_rx,
                whisper_state,
                gigaam,
                llm,
                tx_out,
                state_worker,
                transcribed_end_worker,
                language,
                running_worker,
            )
        });

        Self {
            queue_tx,
            state,
            transcribed_end,
            running,
            _worker_handles: vec![h0],
        }
    }

    pub fn enqueue(&self, path: std::path::PathBuf, source_id: u8) {
        {
            let mut st = self.state.lock().unwrap();
            st.queue_len[source_id as usize] += 1;
        }
        let _ = self.queue_tx.send((path, source_id));
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}
