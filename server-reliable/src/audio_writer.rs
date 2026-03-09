//! Writes PCM from WebSocket to rotating WAV files.

use crate::vad::VadDetector;
use chrono::Utc;
use hound::{WavSpec, WavWriter, SampleFormat};
use serde::Serialize;
use std::collections::VecDeque;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};

const WAV_HEADER_BYTES: u64 = 44;

#[derive(Serialize)]
struct MetaJson {
    start_time_sec: f64,
    end_time_sec: f64,
    source_id: u8,
    duration_sec: f64,
}

struct SourceState {
    source_id: u8,
    session_dir: PathBuf,
    sample_rate: u32,
    max_chunk_sec: f64,
    min_chunk_sec: f64,
    vad_silence_sec: f64,
    overlap_sec: f64,
    seq: u32,
    duration_sec: f64,
    part_path: Option<PathBuf>,
    writer: Option<WavWriter<File>>,
    segment_start_time: Option<f64>,
    overlap_chunks: VecDeque<Vec<u8>>,
    overlap_bytes: usize,
    vad: VadDetector,
}

impl SourceState {
    fn overlap_target_bytes(&self) -> usize {
        (self.overlap_sec * self.sample_rate as f64 * 2.0) as usize
    }

    fn next_path(&mut self) -> PathBuf {
        self.seq += 1;
        self.session_dir
            .join(format!("src{}_{:06}.part", self.source_id, self.seq))
    }

    fn open_new_file(&mut self, prepend_overlap: Option<&[u8]>) -> Result<(), std::io::Error> {
        let path = self.next_path();
        let file = File::create(&path)?;
        let spec = WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::new(file, spec).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        if let Some(data) = prepend_overlap {
            for chunk in data.chunks(2) {
                if chunk.len() == 2 {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    let _ = writer.write_sample(s);
                }
            }
        }
        self.part_path = Some(path);
        self.writer = Some(writer);
        self.segment_start_time = Some(Utc::now().timestamp_millis() as f64 / 1000.0);
        Ok(())
    }

    fn close_and_finalize(&mut self) -> Result<Option<PathBuf>, std::io::Error> {
        let (path, start_time, duration_sec) = match (
            self.part_path.take(),
            self.segment_start_time.take(),
            self.writer.take(),
        ) {
            (Some(p), st, Some(mut w)) => {
                w.flush().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                drop(w);
                let dur = self.duration_sec;
                (p, st, dur)
            }
            _ => return Ok(None),
        };

        let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if size <= WAV_HEADER_BYTES {
            let _ = fs::remove_file(&path);
            return Ok(None);
        }

        let final_path = path.with_extension("wav");
        fs::rename(&path, &final_path)?;

        if let Some(start) = start_time {
            if duration_sec > 0.0 {
                let end_time_sec = start + duration_sec;
                let meta = MetaJson {
                    start_time_sec: start,
                    end_time_sec,
                    source_id: self.source_id,
                    duration_sec,
                };
                let meta_path = final_path.with_extension("meta.json");
                if let Ok(s) = serde_json::to_string_pretty(&meta) {
                    let _ = fs::write(meta_path, s);
                }
            }
        }
        Ok(Some(final_path))
    }

    fn add_to_overlap(&mut self, data: &[u8]) {
        let target = self.overlap_target_bytes();
        if target == 0 {
            return;
        }
        self.overlap_chunks.push_back(data.to_vec());
        self.overlap_bytes += data.len();
        while self.overlap_bytes > target && !self.overlap_chunks.is_empty() {
            if let Some(old) = self.overlap_chunks.pop_front() {
                self.overlap_bytes -= old.len();
            }
        }
    }

    fn get_overlap_bytes(&self) -> Vec<u8> {
        self.overlap_chunks.iter().flat_map(|c| c.iter().copied()).collect()
    }

    fn feed(&mut self, pcm: &[u8]) -> Result<Vec<PathBuf>, std::io::Error> {
        let mut completed = Vec::new();
        let chunk_sec = pcm.len() as f64 / (2.0 * self.sample_rate as f64);
        self.duration_sec += chunk_sec;

        let (_is_speech, should_flush_vad) = self.vad.process_frame(pcm);
        let flush_vad = should_flush_vad
            && self.duration_sec >= self.min_chunk_sec
            && self.writer.is_some();

        let flush_time = self.duration_sec >= self.max_chunk_sec && self.writer.is_some();

        if flush_vad {
            // Важно: записать текущий чанк перед закрытием, иначе теряется конец фразы
            self.write_pcm(pcm)?;
            self.add_to_overlap(pcm);
            self.vad.reset_silence();
            if let Some(p) = self.close_and_finalize()? {
                completed.push(p);
            }
            self.duration_sec = 0.0;
            self.overlap_chunks.clear();
            self.overlap_bytes = 0;
            return Ok(completed);
        }

        if flush_time {
            let overlap = self.get_overlap_bytes();
            self.add_to_overlap(pcm);
            if let Some(p) = self.close_and_finalize()? {
                completed.push(p);
            }
            self.duration_sec = chunk_sec;
            let prepend = if overlap.is_empty() { None } else { Some(overlap.as_slice()) };
            self.open_new_file(prepend)?;
            self.write_pcm(pcm)?;
            return Ok(completed);
        }

        if self.writer.is_none() {
            self.open_new_file(None)?;
        }
        self.write_pcm(pcm)?;
        self.add_to_overlap(pcm);
        Ok(completed)
    }

    fn write_pcm(&mut self, pcm: &[u8]) -> Result<(), std::io::Error> {
        if let Some(ref mut w) = self.writer {
            for chunk in pcm.chunks(2) {
                if chunk.len() == 2 {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    let _ = w.write_sample(s);
                }
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<Option<PathBuf>, std::io::Error> {
        let p = self.close_and_finalize()?;
        self.duration_sec = 0.0;
        self.vad.reset_silence();
        self.overlap_chunks.clear();
        self.overlap_bytes = 0;
        Ok(p)
    }
}

pub struct AudioWriter {
    audio_dir: PathBuf,
    settings: crate::config::Settings,
    session_dir: Option<PathBuf>,
    sources: std::collections::HashMap<u8, SourceState>,
    source_count: u8,
    recording: bool,
}

impl AudioWriter {
    pub fn new(audio_dir: &Path, settings: &crate::config::Settings) -> Self {
        Self {
            audio_dir: audio_dir.to_path_buf(),
            settings: settings.clone(),
            session_dir: None,
            sources: std::collections::HashMap::new(),
            source_count: 2,
            recording: true,
        }
    }

    pub fn start_session(&mut self) -> Result<PathBuf, std::io::Error> {
        let ts = Utc::now().format("%Y%m%d_%H%M%S");
        let session_dir = self.audio_dir.join(format!("session_{}", ts));
        fs::create_dir_all(&session_dir)?;
        self.session_dir = Some(session_dir.clone());
        self.sources.clear();
        Ok(session_dir)
    }

    pub fn set_source_count(&mut self, n: u8) {
        self.source_count = n.clamp(1, 2);
    }

    pub fn set_recording(&mut self, enabled: bool) {
        if self.recording && !enabled {
            let _ = self.flush_all();
        }
        self.recording = enabled;
    }

    fn get_source(&mut self, source_id: u8) -> Option<&mut SourceState> {
        let session_dir = self.session_dir.as_ref()?;
        if source_id == 1 && self.source_count < 2 {
            return None;
        }
        if !self.sources.contains_key(&source_id) {
            let silence_frames =
                (self.settings.sample_rate as f64 * self.settings.vad_silence_sec / 320.0).ceil() as u32;
            self.sources.insert(
                source_id,
                SourceState {
                    source_id,
                    session_dir: session_dir.clone(),
                    sample_rate: self.settings.sample_rate,
                    max_chunk_sec: self.settings.max_chunk_duration_sec,
                    min_chunk_sec: self.settings.min_chunk_duration_sec,
                    vad_silence_sec: self.settings.vad_silence_sec,
                    overlap_sec: self.settings.overlap_sec,
                    seq: 0,
                    duration_sec: 0.0,
                    part_path: None,
                    writer: None,
                    segment_start_time: None,
                    overlap_chunks: VecDeque::new(),
                    overlap_bytes: 0,
                    vad: VadDetector::new(self.settings.sample_rate, silence_frames),
                },
            );
        }
        self.sources.get_mut(&source_id)
    }

    pub fn feed(&mut self, source_id: u8, pcm: &[u8]) -> Result<Vec<(PathBuf, u8)>, std::io::Error> {
        if !self.recording {
            return Ok(Vec::new());
        }
        let src = match self.get_source(source_id) {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let paths = src.feed(pcm)?;
        Ok(paths.into_iter().map(|p| (p, source_id)).collect())
    }

    pub fn flush_all(&mut self) -> Result<Vec<(PathBuf, u8)>, std::io::Error> {
        let mut out = Vec::new();
        for (sid, src) in self.sources.iter_mut() {
            if let Some(p) = src.flush()? {
                out.push((p, *sid));
            }
        }
        Ok(out)
    }

    pub fn close(&mut self) -> Result<Vec<(PathBuf, u8)>, std::io::Error> {
        self.flush_all()
    }

    pub fn session_dir(&self) -> Option<&Path> {
        self.session_dir.as_deref()
    }
}
