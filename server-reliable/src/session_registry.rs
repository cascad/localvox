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
    pub client_id: Option<String>,
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
    /// Maps client_id → active session_id.
    client_sessions: Mutex<HashMap<String, String>>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            client_sessions: Mutex::new(HashMap::new()),
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

    /// Set the active session for a client_id.
    pub fn set_client_session(&self, client_id: String, session_id: String) {
        self.client_sessions
            .lock()
            .unwrap()
            .insert(client_id, session_id);
    }

    /// Get the active session_id for a client_id.
    /// Returns None if no mapping or if the session is already complete.
    pub fn get_active_session_for_client(&self, client_id: &str) -> Option<String> {
        let session_id = {
            let map = self.client_sessions.lock().unwrap();
            map.get(client_id).cloned()
        };
        if let Some(ref sid) = session_id {
            if let Some(session) = self.get(sid) {
                if session.processing_complete.load(Ordering::Relaxed) {
                    self.remove_client_session(client_id);
                    return None;
                }
                return Some(sid.clone());
            }
            self.remove_client_session(client_id);
        }
        None
    }

    /// Remove the client_id → session_id mapping.
    pub fn remove_client_session(&self, client_id: &str) {
        self.client_sessions.lock().unwrap().remove(client_id);
    }

    /// Remove sessions that are completed and older than `max_age`.
    /// Also cleans up stale client_sessions entries.
    pub fn evict_completed(&self, max_age: std::time::Duration) -> usize {
        let mut sessions = self.sessions.lock().unwrap();
        let now = Instant::now();
        let before = sessions.len();
        let mut evicted_ids: Vec<String> = Vec::new();
        sessions.retain(|id, s| {
            if s.processing_complete.load(Ordering::Relaxed) {
                if now.duration_since(s.created_at) >= max_age {
                    evicted_ids.push(id.clone());
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
        drop(sessions);

        if !evicted_ids.is_empty() {
            let mut cs = self.client_sessions.lock().unwrap();
            cs.retain(|_, sid| !evicted_ids.contains(sid));
        }

        before - before.min(before - evicted_ids.len())
    }

    /// Snapshot of current client → session mappings for logging.
    pub fn client_sessions_snapshot(&self) -> Vec<(String, String)> {
        self.client_sessions
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}
