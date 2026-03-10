//! Session registry for client reconnection.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::dispatcher::FileProcessor;
use crate::processor::{OutputSink, TranscribedEnd};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SessionMode {
    Live,
    Batch,
}

pub struct Session {
    pub id: String,
    pub dir: PathBuf,
    pub mode: SessionMode,
    pub processor: Arc<FileProcessor>,
    pub output_sink: Arc<OutputSink>,
    pub transcribed_end: TranscribedEnd,
    pub end_of_stream: AtomicBool,
    pub processing_complete: AtomicBool,
    pub created_at: Instant,
    /// When Some: live session has a timeout task; send to cancel on reconnect.
    pub timeout_cancel_tx: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

pub struct SessionRegistry {
    sessions: Mutex<HashMap<String, Arc<Session>>>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    pub fn get(&self, session_id: &str) -> Option<Arc<Session>> {
        self.sessions.lock().unwrap().get(session_id).cloned()
    }

    pub fn insert(&self, session_id: String, session: Arc<Session>) {
        self.sessions.lock().unwrap().insert(session_id, session);
    }

    /// Take and send to cancel the live reconnect timeout. No-op if none.
    pub fn cancel_live_timeout(&self, session_id: &str) {
        if let Some(session) = self.get(session_id) {
            if let Ok(mut guard) = session.timeout_cancel_tx.lock() {
                if let Some(tx) = guard.take() {
                    let _ = tx.send(());
                }
            }
        }
    }

    pub fn remove(&self, session_id: &str) {
        self.sessions.lock().unwrap().remove(session_id);
    }

    /// Remove sessions that are completed and older than `max_age`.
    pub fn evict_completed(&self, max_age: std::time::Duration) -> usize {
        let mut sessions = self.sessions.lock().unwrap();
        let now = Instant::now();
        let before = sessions.len();
        sessions.retain(|_, s| {
            if s.processing_complete.load(Ordering::Relaxed) {
                now.duration_since(s.created_at) < max_age
            } else {
                true
            }
        });
        before - sessions.len()
    }
}
