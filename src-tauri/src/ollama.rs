/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Ollama model store reader.
//!
//! Ollama keeps weights as content-addressed blobs (`blobs/sha256-<hex>`, no
//! file extension) and records which blob belongs to which model in JSON
//! manifests. A plain extension-based scan therefore finds nothing, which is
//! why models pulled with `ollama pull` were invisible to the library.
//!
//! This reads the manifests and maps each one to its GGUF blob, so those models
//! appear alongside Hugging Face and LM Studio ones. The blobs are ordinary
//! GGUF files (verified magic `GGUF`), so llama-server loads them by path with
//! no conversion or copying — no lock-in, nothing duplicated on disk.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The layer mediaType that carries the actual weights; a manifest also lists
/// template/params/license layers we do not care about.
const MODEL_LAYER: &str = "application/vnd.ollama.image.model";

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    layers: Vec<Layer>,
}

#[derive(Debug, Deserialize)]
struct Layer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    #[serde(default)]
    size: u64,
}

/// A model registered in the local Ollama store.
#[derive(Debug, Clone)]
pub struct OllamaModel {
    /// Ollama-style reference, e.g. `deepseek-r1:14b` or `user/model:latest`.
    pub name: String,
    pub blob_path: PathBuf,
    pub size_bytes: u64,
}

/// Root of the Ollama store: `$OLLAMA_MODELS`, else `~/.ollama/models`.
pub fn models_root() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("OLLAMA_MODELS") {
        return Some(PathBuf::from(dir));
    }
    dirs::home_dir().map(|h| h.join(".ollama").join("models"))
}

/// Every model in the local store, sorted by name. Unreadable or malformed
/// manifests are skipped rather than failing the whole scan — a half-pulled
/// model should not hide the rest of the library.
pub fn discover() -> Vec<OllamaModel> {
    let Some(root) = models_root() else {
        return Vec::new();
    };
    let manifests = root.join("manifests");
    if !manifests.is_dir() {
        return Vec::new();
    }
    let blobs = root.join("blobs");

    let mut out: Vec<OllamaModel> = Vec::new();
    for entry in walkdir::WalkDir::new(&manifests)
        .follow_links(false)
        .into_iter()
        .flatten()
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(model) = read_manifest(path, &manifests, &blobs) else {
            continue;
        };
        out.push(model);
    }
    out.sort_by_cached_key(|m| m.name.to_lowercase());
    out.dedup_by(|a, b| a.blob_path == b.blob_path);
    out
}

/// Parse one manifest file into a model, if it names a weights blob that exists.
fn read_manifest(path: &Path, manifests_root: &Path, blobs: &Path) -> Option<OllamaModel> {
    let text = std::fs::read_to_string(path).ok()?;
    let manifest: Manifest = serde_json::from_str(&text).ok()?;
    let layer = manifest
        .layers
        .iter()
        .find(|l| l.media_type == MODEL_LAYER)?;
    let blob_path = blobs.join(digest_to_filename(&layer.digest)?);
    if !blob_path.is_file() {
        return None; // manifest present but blob missing (interrupted pull)
    }
    // Prefer the real file size; the manifest's is a fallback.
    let size_bytes = blob_path
        .metadata()
        .map(|m| m.len())
        .unwrap_or(layer.size);
    Some(OllamaModel {
        name: model_name(path, manifests_root)?,
        blob_path,
        size_bytes,
    })
}

/// `sha256:abc…` → `sha256-abc…`, the on-disk blob file name. Rejects anything
/// with path separators so a hostile manifest cannot escape the blobs dir.
fn digest_to_filename(digest: &str) -> Option<String> {
    if digest.contains('/') || digest.contains('\\') || digest.contains("..") {
        return None;
    }
    let (algo, hex) = digest.split_once(':')?;
    if algo.is_empty() || hex.is_empty() || !hex.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(format!("{algo}-{hex}"))
}

/// Rebuild the `ollama list` name from the manifest's path. Layout is
/// `manifests/<registry>/<namespace>/<name>/<tag>`; the registry is dropped and
/// the default `library` namespace is implicit, matching what Ollama displays.
fn model_name(path: &Path, manifests_root: &Path) -> Option<String> {
    let rel = path.strip_prefix(manifests_root).ok()?;
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if parts.len() < 3 {
        return None;
    }
    let tag = parts.last()?;
    let name = parts.get(parts.len() - 2)?;
    // Everything between the registry and the model name is the namespace.
    let namespace = &parts[1..parts.len() - 2];
    if namespace.is_empty() || namespace == ["library"] {
        Some(format!("{name}:{tag}"))
    } else {
        Some(format!("{}/{name}:{tag}", namespace.join("/")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_maps_to_blob_filename() {
        assert_eq!(
            digest_to_filename("sha256:abc123"),
            Some("sha256-abc123".to_string())
        );
        // Traversal attempts and malformed digests are rejected.
        assert_eq!(digest_to_filename("sha256:../../etc/passwd"), None);
        assert_eq!(digest_to_filename("sha256:a/b"), None);
        assert_eq!(digest_to_filename("no-colon"), None);
        assert_eq!(digest_to_filename("sha256:"), None);
    }

    #[test]
    fn builds_ollama_style_names() {
        let root = Path::new("/m");
        // library models drop the implicit namespace
        assert_eq!(
            model_name(Path::new("/m/registry.ollama.ai/library/qwen3/14b"), root),
            Some("qwen3:14b".to_string())
        );
        // third-party models keep theirs
        assert_eq!(
            model_name(
                Path::new("/m/registry.ollama.ai/sweaterdog/andy-4/latest"),
                root
            ),
            Some("sweaterdog/andy-4:latest".to_string())
        );
        // too shallow to be a manifest
        assert_eq!(model_name(Path::new("/m/stray-file"), root), None);
    }

    /// Reports whatever is in this machine's Ollama store. Ignored by default;
    /// run with: cargo test -- --ignored --nocapture lists_real_ollama_models
    #[test]
    #[ignore]
    fn lists_real_ollama_models() {
        for m in discover() {
            println!(
                "{:<45} {:>8.2} GB  {}",
                m.name,
                m.size_bytes as f64 / 1e9,
                m.blob_path.display()
            );
        }
    }
}
