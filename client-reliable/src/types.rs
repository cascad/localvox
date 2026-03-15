//! Shared types for the live-transcribe client.

use std::time::Instant;

use ratatui::widgets::ListState;
use serde::Deserialize;

/// Audio source: mic (0) or system sound (1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSource {
    Mic,
    Sys,
    Unknown,
}

impl AudioSource {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => AudioSource::Mic,
            1 => AudioSource::Sys,
            _ => AudioSource::Unknown,
        }
    }

    pub fn to_u8(self) -> Option<u8> {
        match self {
            AudioSource::Mic => Some(0),
            AudioSource::Sys => Some(1),
            AudioSource::Unknown => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            AudioSource::Mic => "mic",
            AudioSource::Sys => "sys",
            AudioSource::Unknown => "src",
        }
    }

    pub fn export_prefix(&self) -> &'static str {
        match self {
            AudioSource::Mic => "mic: ",
            AudioSource::Sys => "sys: ",
            AudioSource::Unknown => "",
        }
    }
}

impl From<Option<u8>> for AudioSource {
    fn from(v: Option<u8>) -> Self {
        v.map(AudioSource::from_u8).unwrap_or(AudioSource::Unknown)
    }
}

/// Single transcript line: text, source, optional segment ID for deduplication.
#[derive(Debug, Clone)]
pub struct TranscriptLine {
    pub text: String,
    pub source: Option<AudioSource>,
    pub seg_id: Option<String>,
}

impl TranscriptLine {
    pub fn new(text: String, source: Option<u8>, seg_id: Option<String>) -> Self {
        Self {
            text,
            source: source.map(AudioSource::from_u8),
            seg_id,
        }
    }

    /// Legacy tuple format for gradual migration.
    pub fn from_tuple(t: (String, Option<u8>, Option<String>)) -> Self {
        Self::new(t.0, t.1, t.2)
    }

    pub fn to_tuple(&self) -> (String, Option<u8>, Option<String>) {
        (
            self.text.clone(),
            self.source.and_then(AudioSource::to_u8),
            self.seg_id.clone(),
        )
    }
}

/// Server message variant (LLM correction).
#[derive(Deserialize, Debug, Clone)]
pub struct TranscriptVariant {
    pub model: String,
    pub text: String,
}

/// WebSocket message from server.
#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ServerMessage {
    Done,
    Session {
        #[serde(default)]
        state: String,
        #[serde(default)]
        session_id: Option<String>,
    },
    Transcript {
        text: String,
        #[serde(default)]
        source: Option<u8>,
        #[serde(default)]
        start_sec: Option<f64>,
        #[serde(default)]
        end_sec: Option<f64>,
        #[serde(default)]
        seg_id: Option<String>,
        #[serde(default)]
        variants: Option<Vec<TranscriptVariant>>,
        #[serde(default)]
        seq: u64,
    },
    Status(StatusData),
    Debug { text: String },
    Error { text: String },
}

/// Server status payload.
#[derive(Deserialize, Debug, Clone, Default)]
pub struct StatusData {
    #[serde(default)]
    pub recording: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    pub running: bool,
    #[serde(default)]
    #[allow(dead_code)]
    pub device: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub task_queue_size: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub task_queue_2_size: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub buffer_sec: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub buffer_2_sec: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub dropped_total: u64,
    #[serde(default)]
    #[allow(dead_code)]
    pub dropped_2_total: u64,
    #[serde(default)]
    #[allow(dead_code)]
    pub pending_1: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub pending_2: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub worker_busy: bool,
    #[serde(default)]
    #[allow(dead_code)]
    pub worker_2_busy: bool,
    #[serde(default)]
    #[allow(dead_code)]
    pub worker_last_proc_sec: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub worker_2_last_proc_sec: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub worker_last_audio_sec: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub worker_2_last_audio_sec: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub skipped_1: u64,
    #[serde(default)]
    #[allow(dead_code)]
    pub skipped_2: u64,
    #[serde(default)]
    #[allow(dead_code)]
    pub lag_sec: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub lag_sec_0: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub lag_sec_1: Option<f64>,
    #[serde(default)]
    pub audio_dir_size_mb: Option<f64>,
    #[serde(default)]
    pub llm_queue: Option<u32>,
}

/// UI event from threads to main loop.
pub enum UiEvent {
    Server(ServerMessage),
    AudioLevel { source: u8, level: f32 },
    Connected,
    Disconnected { reason: String },
    SummarizeDone { msg: String },
    Debug { text: String },
    Quit,
}

/// Outgoing WebSocket message.
pub enum WsOutgoing {
    Binary(Vec<u8>),
    Text(String),
}

/// TUI mode: main view or settings overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiMode {
    Main,
    Settings,
}

/// State for settings overlay (device selection).
pub struct SettingsState {
    pub input_devices: Vec<(usize, String)>,
    pub output_devices: Vec<(usize, String)>,
    pub input_state: ListState,
    pub output_state: ListState,
    pub focus: u8,
    pub saved_msg: Option<String>,
    pub saved_at: Option<Instant>,
}

/// Limits.
pub const MAX_TRANSCRIPT_LINES: usize = 500;
pub const MAX_DEBUG_LINES: usize = 200;
