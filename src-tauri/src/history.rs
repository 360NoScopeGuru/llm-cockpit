/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Persistent chat history.
//!
//! One JSON file per session under `<config>/tokamak/sessions/`. The
//! frontend owns the live session object and re-saves the whole thing after
//! every completed turn, so a crash loses at most the reply in flight. Files
//! are plain JSON on purpose: greppable, diffable, and portable — no
//! proprietary blob store, same philosophy as the model library.
//!
//! Since Rev G `turns` is a *node pool*, not a transcript: every turn carries
//! an id and the id of the turn it follows, so the array describes a tree.
//! A conversation that never forked is a tree with one leaf, which is what
//! every file written before Rev G is — [`migrate`] gives those files ids and
//! chains them in array order, describing the exact same conversation.

use std::collections::{HashMap, HashSet};
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
    /// Stable node id. `None` only in files written before Rev G, and only
    /// until [`migrate`] runs — nothing downstream of a load ever sees one.
    #[serde(default)]
    pub id: Option<String>,
    /// The turn this one follows. `None` marks a root.
    #[serde(default)]
    pub parent: Option<String>,
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
    /// The stream's `finish_reason` ("stop" / "length"). `None` means either an
    /// interrupted stream or a session saved before this was recorded — the UI
    /// must not treat absence as evidence of truncation.
    #[serde(default)]
    pub finish: Option<String>,
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
    /// Leaf of the branch last selected. `None` resolves to the last turn in
    /// the pool, which is what a linear session has always meant.
    #[serde(default)]
    pub head: Option<String>,
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
    /// Turns on the *active* branch. Pool size would inflate this with turns
    /// the reader cannot see from where they are.
    pub turn_count: usize,
    /// Leaves in the pool. 1 means the session never forked.
    pub branch_count: usize,
    /// Every token generated in this session, abandoned branches included.
    /// You paid for those; a tool that sells honest arithmetic should not
    /// quietly drop them from the total.
    pub total_tokens: u64,
    pub avg_decode_tok_s: f64,
}

/// Give a pre-Rev-G file the tree shape the rest of the code now expects.
///
/// Only a file with *no* ids at all is treated as legacy and chained in array
/// order — that is exactly the conversation it already described. A file that
/// has some ids is a Rev G file, and guessing at its parents would invent
/// structure rather than recover it, so this only backfills the ids and leaves
/// every edge alone.
fn migrate(session: &mut Session) {
    let legacy = !session.turns.is_empty() && session.turns.iter().all(|t| t.id.is_none());
    let mut prev: Option<String> = None;
    for (i, t) in session.turns.iter_mut().enumerate() {
        if t.id.is_none() {
            t.id = Some(format!("t{i}"));
        }
        if legacy {
            t.parent = prev;
        }
        prev = t.id.clone();
    }
}

/// Indices of the turns on the branch ending at `head`, in conversation order.
///
/// `head` of `None` resolves to the last turn in the pool. The walk carries a
/// visited set because a hand-edited file could contain a cycle, and a
/// history viewer that hangs on a malformed file is worse than one that
/// truncates.
fn path_indices(turns: &[StoredTurn], head: Option<&str>) -> Vec<usize> {
    if turns.is_empty() {
        return Vec::new();
    }
    // A pool with no ids has not been through `migrate` yet. It is a plain
    // transcript that is already in order, and walking it as a tree would
    // follow no edges and return only its last turn.
    if turns.iter().all(|t| t.id.is_none()) {
        return (0..turns.len()).collect();
    }
    let by_id: HashMap<&str, usize> = turns
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.id.as_deref().map(|id| (id, i)))
        .collect();
    let Some(start) = head
        .and_then(|h| by_id.get(h).copied())
        .or(Some(turns.len() - 1))
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut cur = Some(start);
    while let Some(i) = cur {
        if !seen.insert(i) {
            break;
        }
        out.push(i);
        cur = turns[i]
            .parent
            .as_deref()
            .and_then(|p| by_id.get(p).copied());
    }
    out.reverse();
    out
}

/// A leaf is a turn no other turn claims as its parent.
fn branch_count(turns: &[StoredTurn]) -> usize {
    let claimed: HashSet<&str> = turns.iter().filter_map(|t| t.parent.as_deref()).collect();
    turns
        .iter()
        .filter_map(|t| t.id.as_deref())
        .filter(|id| !claimed.contains(id))
        .count()
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

/// Read one session file and bring it to the current shape.
///
/// Deliberately does not write the migrated form back: a read stays a read,
/// and the next `save` persists it anyway. `list` goes through here too, or
/// legacy files would count their whole transcript as a one-turn branch.
fn load(path: &PathBuf) -> Result<Session, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("session not found: {e}"))?;
    let mut session: Session =
        serde_json::from_str(&text).map_err(|e| format!("corrupt session file: {e}"))?;
    migrate(&mut session);
    Ok(session)
}

pub fn get(id: &str) -> Result<Session, String> {
    validate_id(id)?;
    load(&sessions_dir()?.join(format!("{id}.json")))
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
        let Ok(s) = load(&path) else {
            continue; // skip unreadable or corrupt files rather than failing the whole list
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
        let turn_count = path_indices(&s.turns, s.head.as_deref()).len();
        let branches = branch_count(&s.turns);
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
            turn_count,
            branch_count: branches,
            total_tokens,
            avg_decode_tok_s: avg,
        });
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.updated_ms));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(id: Option<&str>, parent: Option<&str>, content: &str) -> StoredTurn {
        StoredTurn {
            id: id.map(Into::into),
            parent: parent.map(Into::into),
            role: "user".into(),
            kind: None,
            tool_name: None,
            content: content.into(),
            thinking: None,
            tokens: None,
            decode_tok_s: None,
            stopped: None,
            finish: None,
            error: None,
            timestamp_ms: 0,
            sampler: None,
        }
    }

    fn session(turns: Vec<StoredTurn>, head: Option<&str>) -> Session {
        Session {
            id: "s".into(),
            kind: "chat".into(),
            title: "t".into(),
            model_name: None,
            model_path: None,
            binary_label: None,
            n_gpu_layers: None,
            ctx_size: None,
            workspace: None,
            created_ms: 0,
            updated_ms: 0,
            head: head.map(Into::into),
            turns,
        }
    }

    fn contents(turns: &[StoredTurn], head: Option<&str>) -> Vec<String> {
        path_indices(turns, head)
            .into_iter()
            .map(|i| turns[i].content.clone())
            .collect()
    }

    /// The property the whole migration rests on: a file written before Rev G
    /// must describe exactly the same conversation afterwards, in the same
    /// order, with nothing added or dropped.
    #[test]
    fn migration_preserves_the_transcript() {
        let before: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let mut s = session(
            before.iter().map(|c| turn(None, None, c)).collect(),
            None,
        );
        migrate(&mut s);
        assert!(s.turns.iter().all(|t| t.id.is_some()));
        assert_eq!(s.turns[0].parent, None, "the first turn is the root");
        assert_eq!(contents(&s.turns, s.head.as_deref()), before);
    }

    /// Deserialising a genuine pre-Rev-G file: no `id`, no `parent`, no `head`
    /// anywhere in the JSON. Serde defaults have to carry it.
    #[test]
    fn legacy_json_parses_and_migrates() {
        let json = r#"{
            "id": "1784357022134-696nw", "kind": "chat", "title": "old",
            "created_ms": 1, "updated_ms": 2,
            "turns": [
                {"role": "user", "content": "hi", "timestamp_ms": 1},
                {"role": "assistant", "content": "hello", "timestamp_ms": 2}
            ]
        }"#;
        let mut s: Session = serde_json::from_str(json).unwrap();
        assert_eq!(s.head, None);
        migrate(&mut s);
        assert_eq!(contents(&s.turns, s.head.as_deref()), vec!["hi", "hello"]);
        assert_eq!(branch_count(&s.turns), 1, "a linear session has one leaf");
    }

    /// Migration must not invent edges in a file that already has them. A
    /// forked pool with one id missing gets that id filled and nothing else
    /// touched — chaining it in array order would silently graft a branch.
    #[test]
    fn migration_leaves_existing_edges_alone() {
        let mut s = session(
            vec![
                turn(Some("a"), None, "root"),
                turn(Some("b"), Some("a"), "left"),
                turn(None, Some("a"), "right"),
            ],
            Some("a"),
        );
        migrate(&mut s);
        assert_eq!(s.turns[2].parent.as_deref(), Some("a"), "edge preserved");
        assert!(s.turns[2].id.is_some(), "missing id backfilled");
        assert_eq!(branch_count(&s.turns), 2);
    }

    #[test]
    fn path_follows_the_selected_branch() {
        let turns = vec![
            turn(Some("a"), None, "root"),
            turn(Some("b"), Some("a"), "left"),
            turn(Some("c"), Some("a"), "right"),
            turn(Some("d"), Some("c"), "right-2"),
        ];
        assert_eq!(contents(&turns, Some("b")), vec!["root", "left"]);
        assert_eq!(contents(&turns, Some("d")), vec!["root", "right", "right-2"]);
        assert_eq!(branch_count(&turns), 2);
    }

    /// `head` of `None` means "the last turn", which is what every file
    /// written before Rev G relies on.
    #[test]
    fn absent_head_resolves_to_the_last_turn() {
        let turns = vec![
            turn(Some("a"), None, "root"),
            turn(Some("b"), Some("a"), "mid"),
            turn(Some("c"), Some("b"), "tip"),
        ];
        assert_eq!(contents(&turns, None), vec!["root", "mid", "tip"]);
    }

    /// A head naming a turn that is not in the pool must not silently render
    /// an empty transcript; fall back to the last turn.
    #[test]
    fn dangling_head_falls_back_rather_than_emptying() {
        let turns = vec![
            turn(Some("a"), None, "root"),
            turn(Some("b"), Some("a"), "tip"),
        ];
        assert_eq!(contents(&turns, Some("ghost")), vec!["root", "tip"]);
    }

    /// A hand-edited file could contain a cycle. Truncating is acceptable;
    /// hanging the history rail is not.
    #[test]
    fn cyclic_parents_terminate() {
        let turns = vec![
            turn(Some("a"), Some("b"), "a"),
            turn(Some("b"), Some("a"), "b"),
        ];
        let path = path_indices(&turns, Some("a"));
        assert!(path.len() <= turns.len());
    }

    /// Mirrors the same guard in `pathTo` on the frontend. Reachable only if
    /// something walks a pool before `migrate` has run, but the two
    /// implementations have to agree about what an id-less pool means.
    #[test]
    fn unmigrated_pool_reads_as_a_plain_transcript() {
        let turns = vec![
            turn(None, None, "one"),
            turn(None, None, "two"),
            turn(None, None, "three"),
        ];
        assert_eq!(contents(&turns, None), vec!["one", "two", "three"]);
    }

    #[test]
    fn empty_pool_has_no_path_and_no_branches() {
        assert!(path_indices(&[], None).is_empty());
        assert_eq!(branch_count(&[]), 0);
    }

    #[test]
    fn id_validation_blocks_traversal() {
        assert!(validate_id("20260718-093301-ab12").is_ok());
        assert!(validate_id("..\\evil").is_err());
        assert!(validate_id("../evil").is_err());
        assert!(validate_id("a/b").is_err());
        assert!(validate_id("").is_err());
        assert!(validate_id(&"x".repeat(80)).is_err());
    }

    /// The migration property, run against whatever is actually on this
    /// machine rather than a fixture: every real session file must come back
    /// describing the same transcript, in the same order, with the same
    /// number of turns.
    #[test]
    #[ignore] // reads the machine's own history; run with --ignored
    fn real_session_files_migrate_intact() {
        let dir = sessions_dir().unwrap();
        let mut checked = 0;
        for entry in fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().map(|e| e != "json").unwrap_or(true) {
                continue;
            }
            let text = fs::read_to_string(&path).unwrap();
            let Ok(raw) = serde_json::from_str::<Session>(&text) else {
                continue;
            };
            let before: Vec<String> = raw.turns.iter().map(|t| t.content.clone()).collect();
            let migrated = load(&path).unwrap();
            assert_eq!(
                contents(&migrated.turns, migrated.head.as_deref()),
                before,
                "{} did not survive migration",
                path.display()
            );
            assert_eq!(branch_count(&migrated.turns), if before.is_empty() { 0 } else { 1 });
            checked += 1;
        }
        assert!(checked > 0, "no session files found to check");
        eprintln!("migration verified against {checked} real session files");
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
            head: None,
            turns: vec![StoredTurn {
                id: Some("t0".into()),
                parent: None,
                role: "user".into(),
                kind: None,
                tool_name: None,
                content: "hi".into(),
                thinking: None,
                tokens: Some(2),
                decode_tok_s: None,
                stopped: None,
                finish: None,
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
