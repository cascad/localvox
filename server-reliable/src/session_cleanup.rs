//! Очистка старых сессий: по времени (TTL) и по размеру (max MB).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::info;

/// Информация о сессии для сортировки.
struct SessionInfo {
    path: PathBuf,
    mtime: SystemTime,
    size_bytes: u64,
}

/// Удаляет старые сессии по TTL и лимиту размера.
/// Не трогает сессии, изменённые в последние grace_minutes (защита активных).
pub fn run(
    audio_dir: &Path,
    session_ttl_hours: f64,
    audio_dir_max_mb: f64,
    grace_minutes: u64,
) {
    if session_ttl_hours <= 0.0 && audio_dir_max_mb <= 0.0 {
        return;
    }

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
        total_bytes += size_bytes;
        sessions.push(SessionInfo {
            path,
            mtime,
            size_bytes,
        });
    }

    // Сортируем по mtime (старые первыми)
    sessions.sort_by(|a, b| a.mtime.cmp(&b.mtime));

    let mut deleted = 0u32;
    let mut freed_bytes: u64 = 0;

    for s in &sessions {
        if deleted > 0 && total_bytes <= max_bytes {
            break;
        }

        let age = now.duration_since(s.mtime).unwrap_or(Duration::ZERO);
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
