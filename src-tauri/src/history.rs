//! Persistent chat history.
//!
//! One JSON file per session under `<config>/tokamak/sessions/`. The
//! frontend owns the live session object and re-saves the whole thing after
//! every completed turn, so a crash loses at most the reply in flight. Files
//! are plain JSON on purpose: greppable, diffable, and portable — no
//! proprietary blob store, same philosophy as the model library.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Sampler settings in force when a user turn was sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerSnap {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub system: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTurn {
    pub role: String,
    #[serde(default)]
    pub kind: Option<String>, // "tool-result" for tool feedback turns
    #[serde(default)]
    pub tool_name: Option<String>,
    pub content: String,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub tokens: Option<u64>,
    #[serde(default)]
    pub decode_tok_s: Option<f64>,
    #[serde(default)]
    pub stopped: Option<bool>,
    #[serde(default)]
    pub error: Option<bool>,
    #[serde(default)]
    pub timestamp_ms: u64,
    #[serde(default)]
    pub sampler: Option<SamplerSnap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub kind: String, // "chat" | "code"
    pub title: String,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub binary_label: Option<String>,
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default)]
    pub workspace: Option<String>,
    pub created_ms: u64,
    pub updated_ms: u64,
    pub turns: Vec<StoredTurn>,
}

/// Lightweight listing row (the full file is read anyway — sessions are
/// small — but the frontend list stays snappy to render).
#[derive(Debug, Clone, Serialize)]
pub struct SessionMeta {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub model_name: Option<String>,
    pub n_gpu_layers: Option<u32>,
    pub ctx_size: Option<u32>,
    pub workspace: Option<String>,
    pub created_ms: u64,
    pub updated_ms: u64,
    pub turn_count: usize,
    pub total_tokens: u64,
    pub avg_decode_tok_s: f64,
}

fn sessions_dir() -> Result<PathBuf, String> {
    let dir = dirs::config_dir()
        .ok_or("no config dir on this platform")?
        .join("tokamak")
        .join("sessions");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Session ids come from the frontend and become file names — keep them on a
/// strict allowlist so they can never traverse anywhere.
fn validate_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > 64 {
        return Err("bad session id".into());
    }
    if !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
        return Err("bad session id".into());
    }
    Ok(())
}

pub fn save(session: &Session) -> Result<(), String> {
    validate_id(&session.id)?;
    let path = sessions_dir()?.join(format!("{}.json", session.id));
    let json = serde_json::to_string_pretty(session).map_err(|e| e.to_string())?;
    fs::write(&path, json).map_err(|e| e.to_string())
}

pub fn get(id: &str) -> Result<Session, String> {
    validate_id(id)?;
    let path = sessions_dir()?.join(format!("{id}.json"));
    let text = fs::read_to_string(&path).map_err(|e| format!("session not found: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("corrupt session file: {e}"))
}

pub fn delete(id: &str) -> Result<(), String> {
    validate_id(id)?;
    let path = sessions_dir()?.join(format!("{id}.json"));
    fs::remove_file(&path).map_err(|e| e.to_string())
}

pub fn list() -> Result<Vec<SessionMeta>, String> {
    let dir = sessions_dir()?;
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e != "json").unwrap_or(true) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(s) = serde_json::from_str::<Session>(&text) else {
            continue; // skip corrupt files rather than failing the whole list
        };
        let total_tokens: u64 = s.turns.iter().filter_map(|t| t.tokens).sum();
        let rates: Vec<f64> = s
            .turns
            .iter()
            .filter_map(|t| t.decode_tok_s)
            .filter(|r| *r > 0.0)
            .collect();
        let avg = if rates.is_empty() {
            0.0
        } else {
            rates.iter().sum::<f64>() / rates.len() as f64
        };
        out.push(SessionMeta {
            id: s.id,
            kind: s.kind,
            title: s.title,
            model_name: s.model_name,
            n_gpu_layers: s.n_gpu_layers,
            ctx_size: s.ctx_size,
            workspace: s.workspace,
            created_ms: s.created_ms,
            updated_ms: s.updated_ms,
            turn_count: s.turns.len(),
            total_tokens,
            avg_decode_tok_s: avg,
        });
    }
    out.sort_by(|a, b| b.updated_ms.cmp(&a.updated_ms));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_validation_blocks_traversal() {
        assert!(validate_id("20260718-093301-ab12").is_ok());
        assert!(validate_id("..\\evil").is_err());
        assert!(validate_id("../evil").is_err());
        assert!(validate_id("a/b").is_err());
        assert!(validate_id("").is_err());
        assert!(validate_id(&"x".repeat(80)).is_err());
    }

    /// Round-trips a session through the real sessions dir, then deletes it.
    #[test]
    #[ignore] // touches the machine's config dir; run with --ignored
    fn session_roundtrip() {
        let id = format!("test-{}", std::process::id());
        let s = Session {
            id: id.clone(),
            kind: "chat".into(),
            title: "roundtrip".into(),
            model_name: Some("TestModel".into()),
            model_path: None,
            binary_label: None,
            n_gpu_layers: Some(48),
            ctx_size: Some(16384),
            workspace: None,
            created_ms: 1,
            updated_ms: 2,
            turns: vec![StoredTurn {
                role: "user".into(),
                kind: None,
                tool_name: None,
                content: "hi".into(),
                thinking: None,
                tokens: Some(2),
                decode_tok_s: None,
                stopped: None,
                error: None,
                timestamp_ms: 1,
                sampler: None,
            }],
        };
        save(&s).unwrap();
        let back = get(&id).unwrap();
        assert_eq!(back.title, "roundtrip");
        assert_eq!(back.turns.len(), 1);
        assert!(list().unwrap().iter().any(|m| m.id == id));
        delete(&id).unwrap();
        assert!(get(&id).is_err());
    }
}
