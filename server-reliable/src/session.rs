//! Session metadata on disk: session.json, .done marker, startup scan.

use std::path::{Path, PathBuf};
use tracing::info;

/// Write session metadata (client_id etc.) to session_dir/session.json.
pub fn write_session_meta(session_dir: &Path, client_id: Option<&str>) {
    let meta = serde_json::json!({
        "client_id": client_id,
        "created_at": chrono::Utc::now().to_rfc3339(),
    });
    let path = session_dir.join("session.json");
    if let Err(e) = std::fs::write(&path, serde_json::to_string_pretty(&meta).unwrap_or_default()) {
        tracing::warn!("Failed to write session.json: {}", e);
    }
}

/// Mark a session directory as done by creating a .done marker file.
pub fn mark_session_done(session_dir: &Path) {
    let marker = session_dir.join(".done");
    if let Err(e) = std::fs::write(&marker, chrono::Utc::now().to_rfc3339()) {
        tracing::warn!("Failed to write .done marker in {}: {}", session_dir.display(), e);
    }
}

/// Info about a session dir found on disk during startup scan.
pub struct DiskSession {
    pub dir: PathBuf,
    pub session_id: String,
    pub client_id: Option<String>,
    pub is_done: bool,
}

trait ReadClientId {
    fn pipe_read_client_id(&self) -> Option<String>;
}

impl ReadClientId for PathBuf {
    fn pipe_read_client_id(&self) -> Option<String> {
        let data = std::fs::read_to_string(self).ok()?;
        let v: serde_json::Value = serde_json::from_str(&data).ok()?;
        v.get("client_id")?.as_str().map(String::from)
    }
}

/// Scan audio_dir for existing session directories. Returns list of DiskSessions.
pub fn scan_sessions_on_disk(audio_dir: &Path) -> Vec<DiskSession> {
    let Ok(entries) = std::fs::read_dir(audio_dir) else {
        return Vec::new();
    };
    let mut results = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let Some(session_id) = name.strip_prefix("session_") else {
            continue;
        };
        let session_id = session_id.to_string();
        let is_done = path.join(".done").exists();
        let client_id = path.join("session.json").pipe_read_client_id();
        results.push(DiskSession {
            dir: path,
            session_id,
            client_id,
            is_done,
        });
    }
    results
}

/// On startup: scan session dirs, deduplicate by client_id (keep newest by UUIDv7),
/// mark stale as .done, populate client_sessions in registry, log a table.
pub fn startup_session_scan(
    audio_dir: &Path,
    registry: &crate::session_registry::SessionRegistry,
) {
    let sessions = scan_sessions_on_disk(audio_dir);
    if sessions.is_empty() {
        info!("Startup scan: no existing sessions found in {}", audio_dir.display());
        return;
    }

    let active: Vec<&DiskSession> = sessions.iter().filter(|s| !s.is_done).collect();
    let done_count = sessions.len() - active.len();

    // Group active sessions by client_id
    let mut by_client: std::collections::HashMap<String, Vec<&DiskSession>> =
        std::collections::HashMap::new();
    let mut no_client: Vec<&DiskSession> = Vec::new();

    for s in &active {
        match &s.client_id {
            Some(cid) => by_client.entry(cid.clone()).or_default().push(s),
            None => no_client.push(s),
        }
    }

    for (client_id, mut group) in by_client {
        // UUIDv7 sorts lexicographically by time — newest is the largest
        group.sort_by(|a, b| b.session_id.cmp(&a.session_id));
        let newest = group[0];
        registry.set_client_session(client_id.clone(), newest.session_id.clone());

        // Mark the rest as done
        for stale in &group[1..] {
            mark_session_done(&stale.dir);
            info!(
                "Startup: marked stale session {} for client {} as done",
                stale.session_id, client_id
            );
        }

        info!(
            "Startup: client {} → active session {}",
            client_id, newest.session_id
        );
    }

    // Sessions without client_id: mark as done (orphaned)
    for s in &no_client {
        mark_session_done(&s.dir);
        info!(
            "Startup: marked orphan session {} (no client_id) as done",
            s.session_id
        );
    }

    // Log summary table
    let mappings = registry.client_sessions_snapshot();
    if mappings.is_empty() {
        info!("Startup scan: {} sessions on disk ({} done), no active client mappings", sessions.len(), done_count);
    } else {
        info!("Startup scan: {} sessions on disk ({} done), {} active:", sessions.len(), done_count, mappings.len());
        let cw = mappings.iter().map(|(c, _)| c.len()).max().unwrap_or(8).max(9);
        info!("  {:<cw$} | session_id", "client_id", cw = cw);
        info!("  {:-<cw$}-+-{:-<36}", "", "", cw = cw);
        for (cid, sid) in &mappings {
            info!("  {:<cw$} | {}", cid, sid, cw = cw);
        }
    }
}
