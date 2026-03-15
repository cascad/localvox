//! Transcript file I/O and in-memory store.

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::Local;

use crate::types::{AudioSource, TranscriptLine, TranscriptVariant, MAX_TRANSCRIPT_LINES};

/// In-memory transcript store with file persistence.
/// Handles load, append, clear, drain, export, and seg_id deduplication.
pub struct TranscriptStore {
    pub lines: Vec<TranscriptLine>,
    pub known_seg_ids: HashSet<String>,
    out_file: Option<File>,
    output_path: String,
}

impl TranscriptStore {
    /// Load from file and open for append.
    pub fn load(path: &str) -> Result<Self> {
        let lines = Self::parse_file(path);
        let known_seg_ids: HashSet<String> = lines
            .iter()
            .filter_map(|l| l.seg_id.clone())
            .collect();
        let out_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("Не удалось открыть {}", path))?;
        Ok(Self {
            lines,
            known_seg_ids,
            out_file: Some(out_file),
            output_path: path.to_string(),
        })
    }

    /// Parse transcript.txt into TranscriptLine vec.
    fn parse_file(path: &str) -> Vec<TranscriptLine> {
        let Ok(content) = std::fs::read_to_string(path) else { return vec![] };
        let mut out = Vec::new();
        let mut current_segment_id: Option<String> = None;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("=== SEGMENT ") {
                let seg_id = rest.split_whitespace().next().unwrap_or("").to_string();
                if !seg_id.is_empty() && seg_id != "===" {
                    current_segment_id = Some(seg_id);
                }
                continue;
            }
            if line.starts_with("=== END") {
                current_segment_id = None;
                continue;
            }
            if line.starts_with("  ") {
                if let (Some(seg_id), Some(colon_pos)) = (&current_segment_id, line.find(": ")) {
                    let text = line[colon_pos + 2..].to_string();
                    if !text.is_empty() {
                        out.push(TranscriptLine::new(text, None, Some(seg_id.clone())));
                    }
                }
                continue;
            }
            if let Some(pos) = line.find("] ") {
                let rest = line[pos + 2..].trim_start();
                let (source, seg_id, text) = if rest.starts_with("mic [") {
                    if let Some(end) = rest.find("]: ") {
                        let seg = rest[5..end].to_string();
                        (Some(0u8), Some(seg), rest[end + 3..].to_string())
                    } else {
                        continue;
                    }
                } else if rest.starts_with("mic: ") {
                    (Some(0u8), None, rest[5..].to_string())
                } else if rest.starts_with("sys [") {
                    if let Some(end) = rest.find("]: ") {
                        let seg = rest[5..end].to_string();
                        (Some(1u8), Some(seg), rest[end + 3..].to_string())
                    } else {
                        continue;
                    }
                } else if rest.starts_with("sys: ") {
                    (Some(1u8), None, rest[5..].to_string())
                } else {
                    continue;
                };
                if !text.is_empty() {
                    out.push(TranscriptLine::new(text, source, seg_id));
                }
            }
        }
        out
    }

    /// Append plain text segment.
    pub fn append_plain(
        &mut self,
        text: &str,
        source: Option<u8>,
        seg_id: Option<&str>,
        write_to_file: bool,
    ) {
        let line = TranscriptLine::new(text.to_string(), source, seg_id.map(String::from));
        if let Some(ref s) = line.seg_id {
            self.known_seg_ids.insert(s.clone());
        }
        self.lines.push(line.clone());
        if write_to_file {
            let _ = self.write_plain(&line);
        }
        self.drain_excess();
    }

    /// Append segment with variants (LLM correction).
    pub fn append_variants(
        &mut self,
        text: &str,
        source: Option<u8>,
        seg_id: Option<&str>,
        variants: &[TranscriptVariant],
        start_sec: Option<f64>,
        end_sec: Option<f64>,
        write_to_file: bool,
    ) {
        let seg_id_owned = seg_id.map(String::from);
        if let Some(ref s) = seg_id_owned {
            self.known_seg_ids.insert(s.clone());
        }
        if !text.is_empty() {
            self.lines.push(TranscriptLine::new(
                text.to_string(),
                source,
                seg_id_owned.clone(),
            ));
        }
        for v in variants {
            let line = format!("  [{}] {}", v.model, v.text);
            self.lines.push(TranscriptLine::new(
                line,
                source,
                seg_id_owned.clone(),
            ));
        }
        if write_to_file {
            let _ = self.write_variants(seg_id.as_deref(), source, variants, start_sec, end_sec);
        }
        self.drain_excess();
    }

    fn write_plain(&mut self, line: &TranscriptLine) -> Result<()> {
        if let Some(ref mut f) = self.out_file {
            let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
            let src_label = line
                .source
                .map(|s| s.label())
                .unwrap_or("src");
            let seg_part = line
                .seg_id
                .as_ref()
                .map(|s| format!(" [{}]", s))
                .unwrap_or_default();
            writeln!(f, "[{ts}] {src_label}{seg_part}: {}", line.text)?;
            f.flush()?;
        }
        Ok(())
    }

    fn write_variants(
        &mut self,
        seg_id: Option<&str>,
        source: Option<u8>,
        variants: &[TranscriptVariant],
        start_sec: Option<f64>,
        end_sec: Option<f64>,
    ) -> Result<()> {
        if let Some(ref mut f) = self.out_file {
            let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
            let src_label = source
                .map(AudioSource::from_u8)
                .map(|s| s.label())
                .unwrap_or("src");
            let time_range = match (start_sec, end_sec) {
                (Some(s), Some(e)) => format!("{:.1}s–{:.1}s", s, e),
                _ => "?".to_string(),
            };
            let sid = seg_id.unwrap_or("?");
            writeln!(f)?;
            writeln!(f, "=== SEGMENT {} | {} | {} | {} ===", sid, time_range, src_label, ts)?;
            for v in variants {
                writeln!(f, "  {}: {}", v.model, v.text)?;
            }
            writeln!(f, "=== END SEGMENT {} ===", sid)?;
            f.flush()?;
        }
        Ok(())
    }

    /// Clear all lines and truncate file.
    pub fn clear(&mut self) -> Result<()> {
        self.lines.clear();
        self.known_seg_ids.clear();
        drop(self.out_file.take());
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.output_path)
            .with_context(|| format!("Не удалось truncate {}", self.output_path))?;
        drop(f);
        self.out_file = Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.output_path)
                .with_context(|| format!("Не удалось открыть {}", self.output_path))?,
        );
        Ok(())
    }

    /// Trim to MAX_TRANSCRIPT_LINES and rebuild known_seg_ids.
    pub fn drain_excess(&mut self) {
        if self.lines.len() > MAX_TRANSCRIPT_LINES {
            self.lines.drain(..self.lines.len() - MAX_TRANSCRIPT_LINES);
            self.known_seg_ids = self
                .lines
                .iter()
                .filter_map(|l| l.seg_id.clone())
                .collect();
        }
    }

    pub fn contains_seg_id(&self, id: &str) -> bool {
        !id.is_empty() && self.known_seg_ids.contains(id)
    }

    /// Export to result_{datetime}.txt (clean format without seg_id).
    pub fn export(&self, export_dir: &Path) -> Result<(std::path::PathBuf, usize)> {
        let datetime = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let export_path = export_dir.join(format!("result_{}.txt", datetime));
        let mut out = File::create(&export_path)
            .with_context(|| format!("Не удалось создать {}", export_path.display()))?;
        let mut count = 0;

        if let Ok(content) = std::fs::read_to_string(&self.output_path) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with("===") || line.starts_with("  ") {
                    continue;
                }
                if let Some(pos) = line.find("] ") {
                    let rest = line[pos + 2..].trim_start();
                    let clean = if rest.starts_with("mic [") || rest.starts_with("sys [") {
                        let prefix = &rest[..3];
                        if let Some(end) = rest.find("]: ") {
                            Some(format!("{}: {}", prefix, &rest[end + 3..]))
                        } else {
                            None
                        }
                    } else if rest.starts_with("mic: ") || rest.starts_with("sys: ") {
                        Some(rest.to_string())
                    } else {
                        None
                    };
                    if let Some(text) = clean {
                        let _ = writeln!(out, "{}", text);
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            for line in &self.lines {
                if line.text.is_empty() {
                    continue;
                }
                let prefix = line.source.map(|s| s.export_prefix()).unwrap_or("");
                let _ = writeln!(out, "{}{}", prefix, line.text);
                count += 1;
            }
        }

        out.flush()?;
        Ok((export_path, count))
    }

    /// Legacy: lines as tuples for gradual migration.
    pub fn lines_as_tuples(&self) -> Vec<(String, Option<u8>, Option<String>)> {
        self.lines.iter().map(|l| l.to_tuple()).collect()
    }
}

/// Export transcript filtered by segment ID range (inclusive).
/// Returns the number of segments exported.
pub fn export_trimmed_transcript(
    transcript_path: &str,
    export_path: &Path,
    start_seg_id: &Option<String>,
    end_seg_id: &Option<String>,
) -> Result<usize> {
    let content = std::fs::read_to_string(transcript_path)
        .with_context(|| format!("Не удалось прочитать {}", transcript_path))?;
    let mut out = File::create(export_path)
        .with_context(|| format!("Не удалось создать {}", export_path.display()))?;
    let mut segment_count = 0;
    let mut in_range = false;

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("=== SEGMENT ") {
            let seg_id = rest.split_whitespace().next().unwrap_or("").to_string();
            if !seg_id.is_empty() && seg_id != "===" {
                in_range = match (start_seg_id.as_deref(), end_seg_id.as_deref()) {
                    (Some(s), Some(e)) => seg_id.as_str() >= s && seg_id.as_str() <= e,
                    (Some(s), None) => seg_id.as_str() >= s,
                    (None, Some(e)) => seg_id.as_str() <= e,
                    (None, None) => true,
                };
                if in_range {
                    segment_count += 1;
                }
            }
            continue;
        }
        if line.starts_with("=== END") {
            continue;
        }
        if line.starts_with("  ") && in_range {
            if let Some(colon_pos) = line.find(": ") {
                let text = line[colon_pos + 2..].trim();
                if !text.is_empty() {
                    let _ = writeln!(out, "{}", text);
                }
            }
            continue;
        }
        if in_range {
            if let Some(pos) = line.find("] ") {
                let rest = line[pos + 2..].trim_start();
                let clean = if rest.starts_with("mic [") || rest.starts_with("sys [") {
                    if let Some(end) = rest.find("]: ") {
                        Some(format!("{}: {}", &rest[..3], &rest[end + 3..]))
                    } else {
                        None
                    }
                } else if rest.starts_with("mic: ") || rest.starts_with("sys: ") {
                    Some(rest.to_string())
                } else {
                    None
                };
                if let Some(text) = clean {
                    let _ = writeln!(out, "{}", text);
                }
            }
        }
    }

    out.flush()?;
    Ok(segment_count)
}
