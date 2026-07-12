//! Model cache scanner.
//!
//! Discovers GGUF models the user *already has* on disk — the no-lock-in
//! adoption hook. Scans known cache locations (Hugging Face hub, LM Studio) plus
//! any user-added folders, reads each file's GGUF metadata, and flags continuation
//! shards of split models so the library shows one entry per model.

use std::path::{Path, PathBuf};

use serde::Serialize;
use walkdir::WalkDir;

use crate::gguf::{read_gguf_metadata, GgufMetadata};

/// A directory the scanner looks in, with a human label for its origin.
#[derive(Debug, Clone, Serialize)]
pub struct ScanRoot {
    pub path: String,
    pub source: String,
    pub exists: bool,
}

/// One discovered model file.
#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub path: String,
    pub file_name: String,
    pub size_bytes: u64,
    /// Origin: "huggingface" | "lm-studio" | "folder".
    pub source: String,
    /// True for shard 2..N of a split model — hidden from the primary list.
    pub is_shard_continuation: bool,
    pub shard_total: Option<u32>,
    pub metadata: Option<GgufMetadata>,
    pub parse_error: Option<String>,
}

/// Build the default set of roots to scan, honoring `HF_HOME` if set.
pub fn default_roots() -> Vec<(PathBuf, String)> {
    let mut roots: Vec<(PathBuf, String)> = Vec::new();

    // Hugging Face hub cache. Default is ~/.cache/huggingface/hub unless HF_HOME
    // (or the legacy HUGGINGFACE_HUB_CACHE) overrides it.
    if let Some(hf_home) = std::env::var_os("HF_HOME") {
        roots.push((PathBuf::from(hf_home).join("hub"), "huggingface".into()));
    }
    if let Some(cache) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        roots.push((PathBuf::from(cache), "huggingface".into()));
    }
    if let Some(home) = dirs::home_dir() {
        roots.push((
            home.join(".cache").join("huggingface").join("hub"),
            "huggingface".into(),
        ));
        // LM Studio: current (~/.lmstudio/models) and legacy (~/.cache/lm-studio/models).
        roots.push((home.join(".lmstudio").join("models"), "lm-studio".into()));
        roots.push((
            home.join(".cache").join("lm-studio").join("models"),
            "lm-studio".into(),
        ));
    }

    dedupe_roots(roots)
}

/// Canonicalize + de-duplicate root paths, keeping the first source label seen.
fn dedupe_roots(roots: Vec<(PathBuf, String)>) -> Vec<(PathBuf, String)> {
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut out: Vec<(PathBuf, String)> = Vec::new();
    for (path, source) in roots {
        let key = path.canonicalize().unwrap_or_else(|_| path.clone());
        if seen.iter().any(|p| p == &key) {
            continue;
        }
        seen.push(key);
        out.push((path, source));
    }
    out
}

/// Report the default roots and whether each currently exists (for the UI).
pub fn default_roots_info() -> Vec<ScanRoot> {
    default_roots()
        .into_iter()
        .map(|(path, source)| ScanRoot {
            exists: path.is_dir(),
            path: path.to_string_lossy().into_owned(),
            source,
        })
        .collect()
}

/// Scan the default roots plus any `extra_dirs` (labeled "folder") for GGUF models.
pub fn scan_models(extra_dirs: &[String]) -> Vec<ModelEntry> {
    let mut roots = default_roots();
    for d in extra_dirs {
        roots.push((PathBuf::from(d), "folder".into()));
    }
    let roots = dedupe_roots(roots);

    let mut entries: Vec<ModelEntry> = Vec::new();
    let mut seen_paths: Vec<PathBuf> = Vec::new();

    for (root, source) in &roots {
        if !root.is_dir() {
            continue;
        }
        for entry in WalkDir::new(root).follow_links(false).into_iter().flatten() {
            let path = entry.path();
            if !is_gguf(path) {
                continue;
            }
            // A model may live under overlapping roots; de-dup by canonical path.
            let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if seen_paths.iter().any(|p| p == &key) {
                continue;
            }
            seen_paths.push(key);

            entries.push(build_entry(path, source));
        }
    }

    entries.sort_by(|a, b| a.file_name.to_lowercase().cmp(&b.file_name.to_lowercase()));
    entries
}

fn build_entry(path: &Path, source: &str) -> ModelEntry {
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let size_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);
    let shard = shard_index(&file_name);

    let (metadata, parse_error) = match read_gguf_metadata(path) {
        Ok(m) => (Some(m), None),
        Err(e) => (None, Some(e.to_string())),
    };

    ModelEntry {
        path: path.to_string_lossy().into_owned(),
        file_name,
        size_bytes,
        source: source.to_string(),
        is_shard_continuation: shard.map(|(i, _)| i > 1).unwrap_or(false),
        shard_total: shard.map(|(_, t)| t),
        metadata,
        parse_error,
    }
}

fn is_gguf(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
}

/// Parse a `...-00001-of-00003.gguf` shard suffix into (index, total), 1-based.
fn shard_index(file_name: &str) -> Option<(u32, u32)> {
    let stem = file_name.strip_suffix(".gguf")?;
    let (left, total) = stem.rsplit_once("-of-")?;
    let total: u32 = total.parse().ok()?;
    let idx: u32 = left.rsplit('-').next()?.parse().ok()?;
    Some((idx, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_shard_suffix() {
        assert_eq!(shard_index("model-00001-of-00003.gguf"), Some((1, 3)));
        assert_eq!(shard_index("model-00002-of-00003.gguf"), Some((2, 3)));
        assert_eq!(shard_index("plain-model.gguf"), None);
        assert_eq!(shard_index("not-a-model.txt"), None);
    }

    /// Real end-to-end scan of whatever models are on this machine's caches.
    /// Ignored by default (machine-dependent); run with:
    ///   cargo test -- --ignored --nocapture scan_real_caches
    #[test]
    #[ignore]
    fn scan_real_caches() {
        let entries = scan_models(&[]);
        println!("\n--- scanned {} GGUF file(s) ---", entries.len());
        for e in &entries {
            let md = e.metadata.as_ref();
            println!(
                "{:<55} arch={:<10} quant={:<8} ctx={:<8} {} [{}]{}{}",
                e.file_name,
                md.and_then(|m| m.architecture.clone()).unwrap_or("?".into()),
                md.and_then(|m| m.quant_label.clone()).unwrap_or("?".into()),
                md.and_then(|m| m.context_length).unwrap_or(0),
                human_size(e.size_bytes),
                e.source,
                if e.is_shard_continuation { " (shard)" } else { "" },
                e.parse_error
                    .as_ref()
                    .map(|s| format!(" ERR: {s}"))
                    .unwrap_or_default(),
            );
        }
    }

    fn human_size(n: u64) -> String {
        let gb = n as f64 / 1024.0 / 1024.0 / 1024.0;
        format!("{gb:.2}GB")
    }
}
