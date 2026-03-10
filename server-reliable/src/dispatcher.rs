//! Global ASR and LLM dispatchers with priority scheduling (live > youtube).
//!
//! ASR inference runs in parallel across workers. Post-processing (merge_overlap,
//! prev_tail update, send to client) is serialized per session+source via a
//! per-session finalize queue keyed by monotonic sequence number.

use crate::gigaam::GigaAM;
use crate::hallucination;
use crate::llm_corrector::LlmCorrector;
use crate::processor::{
    emit_status, wav_to_f32, ClientMessage, OutputSink, TranscribedEnd, WorkerState,
};
use crate::transcript_postprocess::{ensemble_merge, process_segment};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tracing::info;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

const CONTEXT_LINES: usize = 5;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Priority {
    High,   // live
    Normal, // youtube/batch
}

impl Priority {
    fn ord_key(&self) -> u8 {
        match self {
            Priority::High => 0,
            Priority::Normal => 1,
        }
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ord_key().cmp(&other.ord_key()).reverse()
    }
}

struct AsrTask {
    wav_path: PathBuf,
    source_id: u8,
    session_id: String,
    priority: Priority,
    submitted_at: Instant,
    seq_id: u64,
}

impl PartialEq for AsrTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
            && self.submitted_at == other.submitted_at
    }
}

impl Eq for AsrTask {}

impl PartialOrd for AsrTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AsrTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.submitted_at.cmp(&self.submitted_at))
    }
}

/// Raw ASR output (before merge_overlap), queued for ordered finalization.
struct AsrOutput {
    text: String,
    whisper_text: String,
    gigaam_text: String,
    duration_sec: f64,
    start_sec: f64,
    end_sec: f64,
    wav_path: PathBuf,
    meta_path: PathBuf,
    seg_id: String,
}

/// Entry in the per-session finalize queue. `None` = no speech / error (skip).
struct PendingFinalize {
    output: Option<AsrOutput>,
}

/// Per-session ordered queue: ASR results wait here until all prior chunks are done.
struct FinalizeState {
    queue: [BTreeMap<u64, PendingFinalize>; 2],
    next_seq: [u64; 2],
}

struct LlmTaskWithPriority {
    task: crate::processor::LlmTask,
    priority: Priority,
    submitted_at: Instant,
}

impl PartialEq for LlmTaskWithPriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.submitted_at == other.submitted_at
    }
}

impl Eq for LlmTaskWithPriority {}

impl PartialOrd for LlmTaskWithPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LlmTaskWithPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.submitted_at.cmp(&self.submitted_at))
    }
}

pub struct SessionHandle {
    pub id: String,
    pub worker_state: Arc<Mutex<WorkerState>>,
    pub output_sink: Arc<OutputSink>,
    pub session_dir: PathBuf,
    pub language: String,
    pub priority: Priority,
    pub transcribed_end: TranscribedEnd,
    pub pending_count: AtomicUsize,
    pub llm_inflight: Arc<AtomicUsize>,
    enqueue_seq: [AtomicU64; 2],
    finalize: Mutex<FinalizeState>,
}

pub struct AsrDispatcher {
    inner: Arc<Mutex<AsrInner>>,
    condvar: Arc<std::sync::Condvar>,
    workers: Vec<thread::JoinHandle<()>>,
}

struct AsrInner {
    queue: BinaryHeap<AsrTask>,
    sessions: HashMap<String, Arc<SessionHandle>>,
    running: bool,
}

impl AsrDispatcher {
    pub fn new(
        context: Arc<WhisperContext>,
        gigaam: Option<Arc<GigaAM>>,
        llm_dispatcher: Option<Arc<LlmDispatcher>>,
        num_workers: usize,
    ) -> Self {
        let inner = Arc::new(Mutex::new(AsrInner {
            queue: BinaryHeap::new(),
            sessions: HashMap::new(),
            running: true,
        }));
        let condvar = Arc::new(std::sync::Condvar::new());

        let workers = (0..num_workers.max(1))
            .map(|_| {
                let context = Arc::clone(&context);
                let gigaam = gigaam.clone();
                let llm_dispatcher = llm_dispatcher.clone();
                let inner = Arc::clone(&inner);
                let condvar = Arc::clone(&condvar);
                let state = context
                    .create_state()
                    .expect("create WhisperState");
                let state = Arc::new(Mutex::new(state));
                thread::spawn(move || {
                    Self::worker_loop(
                        state,
                        context,
                        gigaam,
                        llm_dispatcher,
                        &inner,
                        &condvar,
                    );
                })
            })
            .collect();

        Self {
            inner,
            condvar,
            workers,
        }
    }

    fn worker_loop(
        whisper_state: Arc<Mutex<WhisperState>>,
        _context: Arc<WhisperContext>,
        gigaam: Option<Arc<GigaAM>>,
        llm_dispatcher: Option<Arc<LlmDispatcher>>,
        inner: &Arc<Mutex<AsrInner>>,
        condvar: &Arc<std::sync::Condvar>,
    ) {
        loop {
            let (task, handle) = {
                let mut g = inner.lock().unwrap();
                loop {
                    if let Some(t) = g.queue.pop() {
                        if let Some(h) = g.sessions.get(&t.session_id).cloned() {
                            break (t, h);
                        }
                        tracing::debug!(
                            "dropping orphan ASR task for unregistered session {}",
                            t.session_id
                        );
                        continue;
                    }
                    if !g.running {
                        return;
                    }
                    g = condvar.wait(g).unwrap();
                }
            };

            let wav_path = task.wav_path.clone();
            let source_id = task.source_id;
            let seq_id = task.seq_id;
            let meta_path = wav_path.with_extension("meta.json");

            {
                let mut st = handle.worker_state.lock().unwrap();
                st.worker_busy[source_id as usize] = true;
                st.queue_len[source_id as usize] =
                    st.queue_len[source_id as usize].saturating_sub(1);
            }
            emit_status(&handle.worker_state, &handle.output_sink, "ggml-whisper");

            info!(
                "processing src{} seq{}: {}",
                source_id,
                seq_id,
                wav_path.file_name().unwrap_or_default().to_string_lossy()
            );

            // --- Stage 1: ASR inference (parallel, the slow part) ---
            let asr_output = (|| -> anyhow::Result<Option<AsrOutput>> {
                if !wav_path.is_file() {
                    return Ok(None);
                }
                let samples = wav_to_f32(&wav_path)?;
                if samples.is_empty() {
                    return Ok(None);
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
                if !handle.language.is_empty() && handle.language != "auto" {
                    params.set_language(Some(handle.language.as_str()));
                }

                let (whisper_text, gigaam_text) = if let Some(ref g) = gigaam {
                    let state_clone = Arc::clone(&whisper_state);
                    let samples_clone = samples.clone();
                    let wav_clone = wav_path.clone();
                    let lang = handle.language.clone();
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
                        let n = st
                            .full_n_segments()
                            .map_err(|e| anyhow::anyhow!("full_n_segments: {:?}", e))?;
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
                    let w = h_whisper
                        .join()
                        .map_err(|_| anyhow::anyhow!("whisper panic"))??;
                    let gg = h_gigaam
                        .join()
                        .map_err(|_| anyhow::anyhow!("gigaam panic"))
                        .unwrap_or(Ok(String::new()))?;
                    (w, gg)
                } else {
                    let mut st = whisper_state.lock().map_err(|_| anyhow::anyhow!("lock"))?;
                    st.full(params, &samples)
                        .map_err(|e| anyhow::anyhow!("whisper full: {:?}", e))?;
                    let n = st
                        .full_n_segments()
                        .map_err(|e| anyhow::anyhow!("full_n_segments: {:?}", e))?;
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
                    let mut st = handle.worker_state.lock().unwrap();
                    st.last_proc_sec[source_id as usize] = elapsed;
                }

                let gigaam_filtered =
                    hallucination::filter_gigaam(&gigaam_text.trim().to_string());
                let text = if gigaam.is_some() && !gigaam_filtered.is_empty() {
                    ensemble_merge(&whisper_text, &gigaam_filtered)
                } else {
                    whisper_text.clone()
                };

                if text.is_empty() {
                    info!(
                        "transcribe src{}: {:.1}s -> (no speech)",
                        source_id, duration_sec
                    );
                    let _ = std::fs::remove_file(&wav_path);
                    let _ = std::fs::remove_file(&meta_path);
                    return Ok(None);
                }

                let (start_sec, end_sec) = if let Ok(meta_bytes) = std::fs::read(&meta_path) {
                    if let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&meta_bytes) {
                        let start = meta
                            .get("start_time_sec")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        let end = meta
                            .get("end_time_sec")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        (start, end)
                    } else {
                        (0.0, 0.0)
                    }
                } else {
                    (0.0, 0.0)
                };

                let seg_id = wav_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();

                Ok(Some(AsrOutput {
                    text,
                    whisper_text,
                    gigaam_text: gigaam_filtered,
                    duration_sec,
                    start_sec,
                    end_sec,
                    wav_path: wav_path.clone(),
                    meta_path: meta_path.clone(),
                    seg_id,
                }))
            })();

            // ASR done — worker is free
            {
                let mut st = handle.worker_state.lock().unwrap();
                st.worker_busy[source_id as usize] = false;
            }
            emit_status(&handle.worker_state, &handle.output_sink, "ggml-whisper");

            // --- Stage 2: enqueue to finalize queue, then drain in order ---
            let pending = match asr_output {
                Ok(output) => PendingFinalize { output },
                Err(e) => {
                    tracing::debug!("transcription error src{} seq{}: {}", source_id, seq_id, e);
                    let _ = std::fs::remove_file(&wav_path);
                    let _ = std::fs::remove_file(&meta_path);
                    PendingFinalize { output: None }
                }
            };

            {
                let mut fin = handle.finalize.lock().unwrap();
                fin.queue[source_id as usize].insert(seq_id, pending);
            }

            Self::drain_finalize(&handle, source_id, &llm_dispatcher);
        }
    }

    /// Drain the finalize queue in order: while head == next_seq, pop and process.
    /// Called by any worker after inserting a result; only one will make progress
    /// at a time because we hold the finalize lock during pop and advance next_seq.
    fn drain_finalize(
        handle: &SessionHandle,
        source_id: u8,
        llm_dispatcher: &Option<Arc<LlmDispatcher>>,
    ) {
        let si = source_id as usize;
        loop {
            let pending = {
                let mut fin = handle.finalize.lock().unwrap();
                let next = fin.next_seq[si];
                if let Some(p) = fin.queue[si].remove(&next) {
                    fin.next_seq[si] = next + 1;
                    p
                } else {
                    break;
                }
            };

            let Some(out) = pending.output else {
                // No speech / error: just advance sequence
                handle.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
                continue;
            };

            // merge_overlap with prev_tail (sequential, no race)
            let prev_tail = {
                let st = handle.worker_state.lock().unwrap();
                st.prev_tail[si].clone()
            };

            let (merged, new_tail) = process_segment(&prev_tail, &out.text);
            {
                let mut st = handle.worker_state.lock().unwrap();
                st.prev_tail[si] = new_tail;
                st.last_audio_sec[si] = out.duration_sec;
            }

            if merged.is_empty() {
                handle.transcribed_end.set(source_id, out.end_sec);
                let _ = std::fs::remove_file(&out.wav_path);
                let _ = std::fs::remove_file(&out.meta_path);
                handle.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
                continue;
            }

            let context_snapshot = {
                let st = handle.worker_state.lock().unwrap();
                let ctx = &st.context_lines[si];
                if ctx.is_empty() {
                    "(нет)".to_string()
                } else {
                    ctx.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" | ")
                }
            };

            {
                let mut st = handle.worker_state.lock().unwrap();
                let ctx = &mut st.context_lines[si];
                ctx.push_back(merged.clone());
                while ctx.len() > CONTEXT_LINES {
                    ctx.pop_front();
                }
            }

            let asr = crate::processor::AsrResult {
                whisper_text: out.whisper_text.clone(),
                gigaam_text: out.gigaam_text.clone(),
                merged_text: merged.clone(),
                start_sec: out.start_sec,
                end_sec: out.end_sec,
                source_id,
                seg_id: out.seg_id.clone(),
            };

            let asr_path = out.wav_path.with_extension("asr.json");
            if let Ok(js) = serde_json::to_string_pretty(&asr) {
                let _ = std::fs::write(&asr_path, js);
            }

            handle.transcribed_end.set(source_id, out.end_sec);

            if let Some(ref disp) = llm_dispatcher {
                handle.llm_inflight.fetch_add(1, AtomicOrdering::Relaxed);
                let task = crate::processor::LlmTask {
                    asr,
                    context_snapshot,
                    asr_path: asr_path.clone(),
                    wav_path: out.wav_path.clone(),
                    meta_path: out.meta_path.clone(),
                    output_sink: Arc::clone(&handle.output_sink),
                    llm_inflight: Arc::clone(&handle.llm_inflight),
                };
                disp.enqueue(task, handle.priority);
            } else {
                let preview: String = merged.chars().take(80).collect();
                info!(
                    "transcribe src{}: {:.1}s -> {}",
                    source_id, out.duration_sec, preview,
                );
                handle.output_sink.send(&ClientMessage::Transcript {
                    text: merged,
                    source: source_id,
                    start_sec: out.start_sec,
                    end_sec: out.end_sec,
                });
                let _ = std::fs::remove_file(&out.wav_path);
                let _ = std::fs::remove_file(&out.meta_path);
                let _ = std::fs::remove_file(&asr_path);
            }

            handle.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
        }
    }

    pub fn register_session(
        &self,
        session_id: String,
        priority: Priority,
        output_sink: Arc<OutputSink>,
        session_dir: PathBuf,
        language: String,
    ) -> Arc<SessionHandle> {
        let transcribed_end = TranscribedEnd::default();
        let handle = Arc::new(SessionHandle {
            id: session_id.clone(),
            worker_state: Arc::new(Mutex::new(WorkerState {
                prev_tail: [String::new(), String::new()],
                last_proc_sec: [0.0, 0.0],
                last_audio_sec: [0.0, 0.0],
                queue_len: [0, 0],
                worker_busy: [false, false],
                context_lines: [VecDeque::new(), VecDeque::new()],
            })),
            output_sink,
            session_dir,
            language,
            priority,
            transcribed_end: transcribed_end.clone(),
            pending_count: AtomicUsize::new(0),
            llm_inflight: Arc::new(AtomicUsize::new(0)),
            enqueue_seq: [AtomicU64::new(0), AtomicU64::new(0)],
            finalize: Mutex::new(FinalizeState {
                queue: [BTreeMap::new(), BTreeMap::new()],
                next_seq: [0, 0],
            }),
        });
        self.inner.lock().unwrap().sessions.insert(session_id, handle.clone());
        handle
    }

    pub fn enqueue(&self, session_id: &str, path: PathBuf, source_id: u8) {
        let g = self.inner.lock().unwrap();
        let (priority, seq_id) = if let Some(h) = g.sessions.get(session_id) {
            let p = h.priority;
            let seq = h.enqueue_seq[source_id as usize].fetch_add(1, AtomicOrdering::Relaxed);
            h.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
            {
                let mut st = h.worker_state.lock().unwrap();
                st.queue_len[source_id as usize] += 1;
            }
            (p, seq)
        } else {
            (Priority::Normal, 0)
        };
        drop(g);

        let task = AsrTask {
            wav_path: path,
            source_id,
            session_id: session_id.to_string(),
            priority,
            submitted_at: Instant::now(),
            seq_id,
        };
        self.inner.lock().unwrap().queue.push(task);
        self.condvar.notify_one();
    }

    pub fn wait_session_empty(&self, session_id: &str, timeout_sec: f64) {
        let start = std::time::Instant::now();
        while start.elapsed().as_secs_f64() < timeout_sec {
            let Some(h) = self.inner.lock().unwrap().sessions.get(session_id).cloned() else {
                return;
            };
            let pending = h.pending_count.load(AtomicOrdering::Relaxed);
            let llm = h.llm_inflight.load(AtomicOrdering::Relaxed);
            if pending == 0 && llm == 0 {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    pub fn llm_queue_len(&self, session_id: &str) -> usize {
        self.inner
            .lock()
            .unwrap()
            .sessions
            .get(session_id)
            .map(|h| h.llm_inflight.load(AtomicOrdering::Relaxed))
            .unwrap_or(0)
    }

    pub fn asr_task_queue_len(&self, session_id: &str) -> usize {
        self.inner
            .lock()
            .unwrap()
            .sessions
            .get(session_id)
            .map(|h| h.pending_count.load(AtomicOrdering::Relaxed))
            .unwrap_or(0)
    }

    pub fn unregister_session(&self, session_id: &str) {
        self.inner.lock().unwrap().sessions.remove(session_id);
    }
}

pub struct LlmDispatcher {
    inner: Arc<Mutex<LlmInner>>,
    condvar: Arc<std::sync::Condvar>,
    workers: Vec<thread::JoinHandle<()>>,
}

struct LlmInner {
    queue: BinaryHeap<LlmTaskWithPriority>,
    running: bool,
}

impl LlmDispatcher {
    pub fn new(llm: Arc<LlmCorrector>, num_workers: usize) -> Self {
        let inner = Arc::new(Mutex::new(LlmInner {
            queue: BinaryHeap::new(),
            running: true,
        }));
        let condvar = Arc::new(std::sync::Condvar::new());

        let workers = (0..num_workers.max(1))
            .map(|_| {
                let llm = Arc::clone(&llm);
                let inner = Arc::clone(&inner);
                let condvar = Arc::clone(&condvar);
                thread::spawn(move || Self::worker_loop(llm, &inner, &condvar))
            })
            .collect();

        Self {
            inner,
            condvar,
            workers,
        }
    }

    fn worker_loop(
        llm: Arc<LlmCorrector>,
        inner: &Arc<Mutex<LlmInner>>,
        condvar: &Arc<std::sync::Condvar>,
    ) {
        loop {
            let task = {
                let mut g = inner.lock().unwrap();
                loop {
                    if let Some(t) = g.queue.pop() {
                        break t;
                    }
                    if !g.running {
                        return;
                    }
                    g = condvar.wait(g).unwrap();
                }
            };

            let crate::processor::LlmTask {
                asr,
                context_snapshot,
                asr_path,
                wav_path,
                meta_path,
                output_sink,
                llm_inflight,
            } = task.task;

            let final_text = match llm.correct(
                &asr.whisper_text,
                &asr.gigaam_text,
                &asr.merged_text,
                &context_snapshot,
            ) {
                Some(corrected) => corrected,
                None => {
                    tracing::debug!(
                        "LLM correction failed for {}, using merged as fallback",
                        asr.seg_id
                    );
                    asr.merged_text.clone()
                }
            };

            output_sink.send(&ClientMessage::Transcript {
                text: final_text,
                source: asr.source_id,
                start_sec: asr.start_sec,
                end_sec: asr.end_sec,
            });

            let _ = std::fs::remove_file(&wav_path);
            let _ = std::fs::remove_file(&meta_path);
            let _ = std::fs::remove_file(&asr_path);

            llm_inflight.fetch_sub(1, AtomicOrdering::Relaxed);
        }
    }

    pub fn enqueue(&self, task: crate::processor::LlmTask, priority: Priority) {
        let wrapped = LlmTaskWithPriority {
            task,
            priority,
            submitted_at: Instant::now(),
        };
        self.inner.lock().unwrap().queue.push(wrapped);
        self.condvar.notify_one();
    }
}

/// Thin wrapper for session: enqueues to global dispatcher.
pub struct FileProcessor {
    session_id: String,
    dispatcher: Arc<AsrDispatcher>,
    pub transcribed_end: TranscribedEnd,
}

impl FileProcessor {
    pub fn new(
        session_id: String,
        dispatcher: Arc<AsrDispatcher>,
        handle: Arc<SessionHandle>,
    ) -> Self {
        Self {
            session_id: session_id.clone(),
            dispatcher,
            transcribed_end: handle.transcribed_end.clone(),
        }
    }

    pub fn enqueue(&self, path: PathBuf, source_id: u8) {
        self.dispatcher.enqueue(&self.session_id, path, source_id);
    }

    pub fn wait_until_empty(&self, timeout_sec: f64) {
        self.dispatcher.wait_session_empty(&self.session_id, timeout_sec);
    }

    pub fn llm_queue_len(&self) -> usize {
        self.dispatcher.llm_queue_len(&self.session_id)
    }

    pub fn asr_task_queue_len(&self) -> usize {
        self.dispatcher.asr_task_queue_len(&self.session_id)
    }
}
