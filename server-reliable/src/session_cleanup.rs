//! Очистка старых сессий: по времени (TTL) и по размеру (max MB).
//! Sessions marked with `.done` are cleaned up with a short grace period (5 min).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::info;

/// TTL for sessions marked as .done (5 minutes).
const DONE_SESSION_TTL: Duration = Duration::from_secs(5 * 60);

/// Информация о сессии для сортировки.
struct SessionInfo {
    path: PathBuf,
    mtime: SystemTime,
    size_bytes: u64,
    is_done: bool,
}

/// Удаляет старые сессии по TTL и лимиту размера.
/// Sessions with `.done` marker are cleaned up after 5 minutes regardless of main TTL.
/// Не трогает сессии, изменённые в последние grace_minutes (защита активных, кроме .done).
pub fn run(
    audio_dir: &Path,
    session_ttl_hours: f64,
    audio_dir_max_mb: f64,
    grace_minutes: u64,
) {
    let Ok(entries) = std::fs::read_dir(audio_dir) else {
        return;
    };

    let grace = Duration::from_secs(grace_minutes * 60);
    let ttl = Duration::from_secs_f64(session_ttl_hours * 3600.0);
    let max_bytes = if audio_dir_max_mb > 0.0 {
        (audio_dir_max_mb * 1024.0 * 1024.0) as u64
    } else {
        u64::MAX
    };

    let now = SystemTime::now();
    let mut sessions: Vec<SessionInfo> = Vec::new();
    let mut total_bytes: u64 = 0;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.starts_with("session_") {
            continue;
        }

        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
        let size_bytes = dir_size(&path);
        let is_done = path.join(".done").exists();
        total_bytes += size_bytes;
        sessions.push(SessionInfo {
            path,
            mtime,
            size_bytes,
            is_done,
        });
    }

    sessions.sort_by(|a, b| a.mtime.cmp(&b.mtime));

    let mut deleted = 0u32;
    let mut freed_bytes: u64 = 0;

    for s in &sessions {
        let age = now.duration_since(s.mtime).unwrap_or(Duration::ZERO);

        if s.is_done {
            // .done sessions: short TTL, no grace protection
            if age >= DONE_SESSION_TTL {
                if let Err(e) = std::fs::remove_dir_all(&s.path) {
                    tracing::warn!("Failed to remove done session {}: {}", s.path.display(), e);
                } else {
                    deleted += 1;
                    freed_bytes += s.size_bytes;
                    total_bytes = total_bytes.saturating_sub(s.size_bytes);
                }
            }
            continue;
        }

        if session_ttl_hours <= 0.0 && audio_dir_max_mb <= 0.0 {
            continue;
        }

        if deleted > 0 && total_bytes <= max_bytes {
            break;
        }

        if age < grace {
            continue;
        }

        let by_ttl = session_ttl_hours > 0.0 && age >= ttl;
        let by_size = audio_dir_max_mb > 0.0 && total_bytes > max_bytes;

        if by_ttl || by_size {
            if let Err(e) = std::fs::remove_dir_all(&s.path) {
                tracing::warn!("Failed to remove session {}: {}", s.path.display(), e);
            } else {
                deleted += 1;
                freed_bytes += s.size_bytes;
                total_bytes = total_bytes.saturating_sub(s.size_bytes);
            }
        }
    }

    if deleted > 0 {
        info!(
            "Session cleanup: removed {} dirs, freed {:.1} MB",
            deleted,
            freed_bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

fn dir_size(path: &Path) -> u64 {
    let mut total: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += dir_size(&path);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}
