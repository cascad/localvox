//! Durable queue state.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Server status for InProgress jobs (ASR/LLM queues, lag).
#[derive(Debug, Clone, Default)]
pub struct ServerStatus {
    pub task_queue_size: usize,
    pub llm_queue: usize,
    pub lag_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum JobStatus {
    Pending,
    InProgress,
    Done,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub url: String,
    pub status: JobStatus,
    pub output_path: String,
    #[serde(default)]
    pub error: Option<String>,
    pub created_at: String,
    /// Server session ID for reconnection. Persisted for retries.
    #[serde(default)]
    pub session_id: String,
    /// (sent_bytes, total_bytes) — для InProgress, не сериализуется в queue.json
    #[serde(skip, default)]
    pub progress: Option<(usize, usize)>,
    /// Server status (ASR/LLM queues, lag) — для InProgress.
    #[serde(skip, default)]
    pub server_status: Option<ServerStatus>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct QueueState {
    pub jobs: Vec<Job>,
}

impl QueueState {
    pub fn load(path: &Path) -> Result<Self> {
        if path.is_file() {
            let s = std::fs::read_to_string(path).context("read state")?;
            serde_json::from_str(&s).context("parse state")
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let s = serde_json::to_string_pretty(self).context("serialize state")?;
        std::fs::write(path, s).context("write state")
    }

    pub fn add_job(&mut self, url: String, output_path: impl AsRef<std::path::Path>) {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let session_id = uuid::Uuid::new_v4().simple().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        self.jobs.push(Job {
            id: id.clone(),
            url,
            status: JobStatus::Pending,
            output_path: output_path.as_ref().to_string_lossy().to_string(),
            error: None,
            created_at,
            session_id,
            progress: None,
            server_status: None,
        });
    }

    pub fn next_pending(&mut self) -> Option<usize> {
        self.jobs
            .iter()
            .position(|j| matches!(j.status, JobStatus::Pending))
    }

    pub fn set_in_progress(&mut self, idx: usize) {
        if let Some(j) = self.jobs.get_mut(idx) {
            j.status = JobStatus::InProgress;
            j.progress = None;
            j.server_status = None;
        }
    }

    pub fn set_server_status(&mut self, idx: usize, status: ServerStatus) {
        if let Some(j) = self.jobs.get_mut(idx) {
            j.server_status = Some(status);
        }
    }

    pub fn set_progress(&mut self, idx: usize, sent_bytes: usize, total_bytes: usize) {
        if let Some(j) = self.jobs.get_mut(idx) {
            j.progress = Some((sent_bytes, total_bytes));
        }
    }

    pub fn set_done(&mut self, idx: usize) {
        if let Some(j) = self.jobs.get_mut(idx) {
            j.status = JobStatus::Done;
        }
    }

    pub fn set_failed(&mut self, idx: usize, err: String) {
        if let Some(j) = self.jobs.get_mut(idx) {
            j.status = JobStatus::Failed(err.clone());
            j.error = Some(err);
        }
    }
}
