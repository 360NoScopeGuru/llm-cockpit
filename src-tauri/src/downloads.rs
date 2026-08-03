//! In-app model downloads.
//!
//! Three sources, all landing as plain GGUF files in a folder the scanner
//! already watches — no proprietary store, no re-download to use them
//! elsewhere:
//!   * `huggingface` — search the hub, pick a quant, stream it down.
//!   * `url`         — paste a direct link to a .gguf.
//!   * `ollama`      — shell out to `ollama pull` (only offered when Ollama is
//!                     installed; the blobs it writes are already scanned).
//!
//! Downloads are resumable: bytes go to `<name>.part` and a restart continues
//! with a Range request, so a dropped connection 15 GiB in is not fatal.
//!
//! Everything here is user-initiated. Tokamak makes no network request unless
//! someone clicks something — the offline-by-default promise is the point.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::Emitter;

const HF_API: &str = "https://huggingface.co/api";
const HF_HOST: &str = "https://huggingface.co";
/// Emit progress at most this often — a 15 GiB file would otherwise flood the
/// webview with events and cost more than the download.
const PROGRESS_EVERY: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Serialize)]
pub struct HfModel {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub gated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfFile {
    pub name: String,
    pub size_bytes: u64,
    /// Quant label parsed out of the file name (Q4_K_M, IQ4_XS, F16, …), used
    /// to line the file up against the quant advisor's verdict.
    pub quant: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Progress {
    id: u64,
    received: u64,
    total: u64,
    bytes_per_sec: f64,
    done: bool,
    cancelled: bool,
    path: Option<String>,
    error: Option<String>,
}

/// One cancel flag per in-flight download, so several can run and be cancelled
/// independently.
#[derive(Default)]
pub struct DownloadState {
    flags: Mutex<Vec<(u64, Arc<AtomicBool>)>>,
}

impl DownloadState {
    fn arm(&self, id: u64) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        let mut g = self.flags.lock().unwrap();
        g.retain(|(_, f)| !f.load(Ordering::Relaxed));
        g.push((id, flag.clone()));
        flag
    }

    pub fn cancel(&self, id: u64) {
        if let Some((_, f)) = self.flags.lock().unwrap().iter().find(|(i, _)| *i == id) {
            f.store(true, Ordering::Relaxed);
        }
    }
}

/// Where downloads land. Registered as a scan root so they appear in the
/// library immediately, and readable by any other tool.
pub fn models_dir() -> Result<PathBuf, String> {
    let dir = dirs::config_dir()
        .ok_or("no config dir on this platform")?
        .join("tokamak")
        .join("models");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Strip a path segment down to something safe to join onto the models dir.
/// Repo and file names come from the internet, so they are treated as hostile:
/// separators, `..` and control characters are removed rather than escaped.
fn safe_segment(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    let trimmed = cleaned.trim().trim_matches('.').to_string();
    if trimmed.is_empty() {
        "unnamed".into()
    } else {
        trimmed.chars().take(120).collect()
    }
}

/// Resolve the destination for a download and prove it stays inside the models
/// directory even if `safe_segment` were ever bypassed.
fn dest_path(parts: &[&str], file: &str) -> Result<PathBuf, String> {
    let root = models_dir()?;
    let mut p = root.clone();
    for part in parts {
        p.push(safe_segment(part));
    }
    p.push(safe_segment(file));
    let canon_root = fs::canonicalize(&root).unwrap_or(root);
    // The file does not exist yet, so check the deepest existing ancestor.
    let mut probe = p.clone();
    while !probe.exists() {
        if !probe.pop() {
            return Err("could not resolve download path".into());
        }
    }
    let canon_probe = fs::canonicalize(&probe).map_err(|e| e.to_string())?;
    if !canon_probe.starts_with(&canon_root) {
        return Err("download path escapes the models directory".into());
    }
    Ok(p)
}

/// Parse a GGUF quant label out of a file name, e.g. `…-Q4_K_M.gguf`.
fn quant_from_name(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    stem.rsplit(['-', '.'])
        .find(|part| {
            let u = part.to_ascii_uppercase();
            (u.starts_with('Q') || u.starts_with("IQ") || u == "F16" || u == "F32" || u == "BF16")
                && u.chars().any(|c| c.is_ascii_digit())
        })
        .map(|s| s.to_ascii_uppercase())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(60))
        .build()
}

fn with_token(req: ureq::Request, token: Option<&str>) -> ureq::Request {
    match token {
        Some(t) if !t.trim().is_empty() => req.set("Authorization", &format!("Bearer {}", t.trim())),
        _ => req,
    }
}

/// Search the hub for GGUF repos, most-downloaded first.
pub fn search(query: &str, token: Option<&str>) -> Result<Vec<HfModel>, String> {
    let url = format!(
        "{HF_API}/models?search={}&filter=gguf&sort=downloads&direction=-1&limit=30",
        urlencode(query)
    );
    let body = with_token(agent().get(&url), token)
        .call()
        .map_err(|e| format!("search failed: {e}"))?
        .into_string()
        .map_err(|e| e.to_string())?;
    let raw: Vec<serde_json::Value> = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    Ok(raw
        .into_iter()
        .filter_map(|m| {
            Some(HfModel {
                id: m.get("modelId")?.as_str()?.to_string(),
                downloads: m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                likes: m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                // `gated` is false, "auto" or "manual" — anything truthy needs a token.
                gated: m
                    .get("gated")
                    .map(|g| !matches!(g, serde_json::Value::Bool(false)))
                    .unwrap_or(false),
            })
        })
        .collect())
}

/// List the GGUF files in a repo, with sizes so the caller can judge fit.
pub fn list_files(repo: &str, token: Option<&str>) -> Result<Vec<HfFile>, String> {
    let url = format!("{HF_API}/models/{repo}?blobs=true");
    let resp = with_token(agent().get(&url), token).call().map_err(|e| {
        if e.to_string().contains("401") || e.to_string().contains("403") {
            format!("{repo} is gated — add a Hugging Face token in settings")
        } else {
            format!("could not read {repo}: {e}")
        }
    })?;
    let v: serde_json::Value = serde_json::from_str(
        &resp.into_string().map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut out: Vec<HfFile> = v
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|f| {
                    let name = f.get("rfilename")?.as_str()?.to_string();
                    if !name.to_ascii_lowercase().ends_with(".gguf") {
                        return None;
                    }
                    Some(HfFile {
                        size_bytes: f.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                        quant: quant_from_name(&name),
                        name,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by_key(|f| f.size_bytes);
    Ok(out)
}

/// Download a file from a Hugging Face repo.
pub fn start_hf(
    window: tauri::Window,
    state: &DownloadState,
    id: u64,
    repo: String,
    file: String,
    token: Option<String>,
) -> Result<(), String> {
    let dest = dest_path(&repo.split('/').collect::<Vec<_>>(), &file)?;
    let url = format!("{HF_HOST}/{repo}/resolve/main/{file}");
    spawn_stream(window, state, id, url, dest, token);
    Ok(())
}

/// Download a direct .gguf URL.
pub fn start_url(
    window: tauri::Window,
    state: &DownloadState,
    id: u64,
    url: String,
) -> Result<(), String> {
    if !url.starts_with("https://") {
        return Err("only https:// URLs are accepted".into());
    }
    let name = url
        .split('?')
        .next()
        .unwrap_or(&url)
        .rsplit('/')
        .next()
        .filter(|n| n.to_ascii_lowercase().ends_with(".gguf"))
        .ok_or("that URL does not point at a .gguf file")?
        .to_string();
    let dest = dest_path(&["direct"], &name)?;
    spawn_stream(window, state, id, url, dest, None);
    Ok(())
}

/// Stream a URL to disk on a worker thread, resuming a partial file if present.
fn spawn_stream(
    window: tauri::Window,
    state: &DownloadState,
    id: u64,
    url: String,
    dest: PathBuf,
    token: Option<String>,
) {
    let cancel = state.arm(id);
    std::thread::spawn(move || {
        let w = window.clone();
        let emit = move |received: u64, total: u64, rate: f64| {
            let _ = w.emit(
                "download-progress",
                Progress {
                    id,
                    received,
                    total,
                    bytes_per_sec: rate,
                    done: false,
                    cancelled: false,
                    path: None,
                    error: None,
                },
            );
        };
        let result = stream_to_disk(&emit, &url, &dest, token.as_deref(), &cancel);
        let done = match result {
            Ok(Some(path)) => Progress {
                id,
                received: 0,
                total: 0,
                bytes_per_sec: 0.0,
                done: true,
                cancelled: false,
                path: Some(path),
                error: None,
            },
            Ok(None) => Progress {
                id,
                received: 0,
                total: 0,
                bytes_per_sec: 0.0,
                done: true,
                cancelled: true,
                path: None,
                error: None,
            },
            Err(e) => Progress {
                id,
                received: 0,
                total: 0,
                bytes_per_sec: 0.0,
                done: true,
                cancelled: false,
                path: None,
                error: Some(e),
            },
        };
        let _ = window.emit("download-progress", done);
    });
}

/// Core download loop. Takes a progress callback rather than a `Window` so it
/// can be exercised in tests without a running Tauri app.
fn stream_to_disk(
    on_progress: &dyn Fn(u64, u64, f64),
    url: &str,
    dest: &Path,
    token: Option<&str>,
    cancel: &AtomicBool,
) -> Result<Option<String>, String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    if dest.exists() {
        // Already downloaded — nothing to do, and re-fetching 15 GiB would be rude.
        return Ok(Some(dest.to_string_lossy().into_owned()));
    }
    let part = dest.with_extension("part");
    let already = part.metadata().map(|m| m.len()).unwrap_or(0);

    let mut req = agent().get(url);
    req = with_token(req, token);
    if already > 0 {
        req = req.set("Range", &format!("bytes={already}-"));
    }
    let resp = req.call().map_err(|e| format!("download failed: {e}"))?;

    let status = resp.status();
    // 206 = server honoured the resume; 200 = it ignored it and is sending the
    // whole file, so the partial data must be discarded rather than appended to.
    let resuming = status == 206 && already > 0;
    let remaining: u64 = resp
        .header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let total = if resuming { already + remaining } else { remaining };

    let mut file = if resuming {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .open(&part)
            .map_err(|e| e.to_string())?;
        f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        f
    } else {
        fs::File::create(&part).map_err(|e| e.to_string())?
    };

    let mut received = if resuming { already } else { 0 };
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20]; // 1 MiB
    let started = Instant::now();
    let start_bytes = received;
    let mut last_emit = Instant::now();

    loop {
        if cancel.load(Ordering::Relaxed) {
            // Keep the .part file so the user can resume later.
            let _ = file.flush();
            return Ok(None);
        }
        let n = reader.read(&mut buf).map_err(|e| format!("read failed: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| {
            format!("write failed (out of disk?): {e}")
        })?;
        received += n as u64;

        if last_emit.elapsed() >= PROGRESS_EVERY {
            let secs = started.elapsed().as_secs_f64().max(0.001);
            on_progress(received, total, (received - start_bytes) as f64 / secs);
            last_emit = Instant::now();
        }
    }

    file.flush().map_err(|e| e.to_string())?;
    drop(file);
    if total > 0 && received < total {
        return Err(format!(
            "connection closed early: got {received} of {total} bytes — retry to resume"
        ));
    }
    fs::rename(&part, dest).map_err(|e| format!("could not finalise download: {e}"))?;
    Ok(Some(dest.to_string_lossy().into_owned()))
}

/// Is the `ollama` CLI available? Drives whether the Ollama source is offered
/// at all — a source that always errors is worse than one that is absent.
pub fn ollama_available() -> bool {
    which_ollama().is_some()
}

fn which_ollama() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "ollama.exe" } else { "ollama" };
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(exe))
            .find(|p| p.is_file())
    })
}

/// Run `ollama pull <model>`, streaming its progress lines to the UI. The blobs
/// it writes are already picked up by the scanner, so nothing else is needed.
pub fn start_ollama(
    window: tauri::Window,
    state: &DownloadState,
    id: u64,
    model: String,
) -> Result<(), String> {
    let exe = which_ollama().ok_or("ollama is not installed or not on PATH")?;
    if model.trim().is_empty() || model.contains(['&', '|', ';', '\n']) {
        return Err("invalid model name".into());
    }
    let cancel = state.arm(id);
    std::thread::spawn(move || {
        use std::process::{Command, Stdio};
        let mut cmd = Command::new(exe);
        cmd.arg("pull")
            .arg(model.trim())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = window.emit(
                    "download-progress",
                    Progress {
                        id,
                        received: 0,
                        total: 0,
                        bytes_per_sec: 0.0,
                        done: true,
                        cancelled: false,
                        path: None,
                        error: Some(format!("could not run ollama: {e}")),
                    },
                );
                return;
            }
        };
        loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = window.emit(
                    "download-progress",
                    Progress {
                        id,
                        received: 0,
                        total: 0,
                        bytes_per_sec: 0.0,
                        done: true,
                        cancelled: true,
                        path: None,
                        error: None,
                    },
                );
                return;
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    let ok = status.success();
                    let _ = window.emit(
                        "download-progress",
                        Progress {
                            id,
                            received: 0,
                            total: 0,
                            bytes_per_sec: 0.0,
                            done: true,
                            cancelled: false,
                            path: None,
                            error: (!ok).then(|| format!("ollama pull failed ({status})")),
                        },
                    );
                    return;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(300)),
                Err(e) => {
                    let _ = window.emit(
                        "download-progress",
                        Progress {
                            id,
                            received: 0,
                            total: 0,
                            bytes_per_sec: 0.0,
                            done: true,
                            cancelled: false,
                            path: None,
                            error: Some(e.to_string()),
                        },
                    );
                    return;
                }
            }
        }
    });
    Ok(())
}

/// Minimal percent-encoding for a query string — avoids pulling in a crate for
/// one call site.
fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[derive(Debug, Deserialize)]
pub struct DownloadRequest {
    pub id: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_cannot_escape() {
        assert_eq!(safe_segment("../../etc"), "_.._etc");
        assert_eq!(safe_segment("a/b"), "a_b");
        assert_eq!(safe_segment("a\\b"), "a_b");
        assert_eq!(safe_segment("C:evil"), "C_evil");
        assert_eq!(safe_segment("  "), "unnamed");
        assert_eq!(safe_segment(""), "unnamed");
        // Trailing dots are a classic Windows trick.
        assert_eq!(safe_segment("nul."), "nul");
        assert!(safe_segment(&"x".repeat(500)).chars().count() <= 120);
    }

    #[test]
    fn reads_quant_from_file_name() {
        assert_eq!(quant_from_name("Qwen3-14B-Q4_K_M.gguf"), Some("Q4_K_M".into()));
        assert_eq!(quant_from_name("model-IQ4_XS.gguf"), Some("IQ4_XS".into()));
        assert_eq!(quant_from_name("mmproj-F16.gguf"), Some("F16".into()));
        assert_eq!(quant_from_name("plain-model.gguf"), None);
    }

    #[test]
    fn encodes_query_strings() {
        assert_eq!(urlencode("qwen3 coder"), "qwen3+coder");
        assert_eq!(urlencode("a/b&c"), "a%2Fb%26c");
        assert_eq!(urlencode("plain"), "plain");
    }

    /// Hits the real hub. Ignored by default:
    /// cargo test -- --ignored --nocapture hf_search_and_list
    #[test]
    #[ignore]
    fn hf_search_and_list() {
        let hits = search("qwen3", None).expect("search");
        println!("\n{} results", hits.len());
        for m in hits.iter().take(3) {
            println!("  {} dl={} gated={}", m.id, m.downloads, m.gated);
        }
        let files = list_files("lmstudio-community/Qwen3-14B-GGUF", None).expect("list");
        for f in &files {
            println!(
                "  {:<44} {:>7.2} GiB  quant={:?}",
                f.name,
                f.size_bytes as f64 / 1024.0_f64.powi(3),
                f.quant
            );
        }
        assert!(!files.is_empty());
    }
}

#[cfg(test)]
mod live_download {
    use super::*;

    /// A genuinely small GGUF so the test is quick but still exercises the real
    /// CDN, Range headers and the .part rename.
    const SMALL: &str =
        "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf";

    /// Downloads for real, interrupts, resumes, and checks the result is a
    /// valid GGUF of the right length.
    /// cargo test -- --ignored --nocapture resumes_after_interruption
    #[test]
    #[ignore]
    fn resumes_after_interruption() {
        let dir = std::env::temp_dir().join(format!("tokamak-dl-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let dest = dir.join("stories15M-q4_0.gguf");
        let part = dest.with_extension("part");
        let _ = fs::remove_file(&dest);
        let _ = fs::remove_file(&part);

        // Cancel as soon as anything has landed.
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        let seen = Arc::new(Mutex::new(0u64));
        let s2 = seen.clone();
        let on_progress = move |received: u64, _total: u64, _rate: f64| {
            *s2.lock().unwrap() = received;
            c2.store(true, Ordering::Relaxed);
        };
        let first = stream_to_disk(&on_progress, SMALL, &dest, None, &cancel).expect("first leg");
        assert!(first.is_none(), "should have reported cancellation");
        let partial = part.metadata().map(|m| m.len()).unwrap_or(0);
        println!("interrupted with {partial} bytes in .part");
        assert!(partial > 0, "expected a partial file to resume from");
        assert!(!dest.exists(), "must not publish an incomplete file");

        // Resume: no cancellation this time.
        let quiet = |_: u64, _: u64, _: f64| {};
        let never = AtomicBool::new(false);
        let done = stream_to_disk(&quiet, SMALL, &dest, None, &never)
            .expect("resume leg")
            .expect("should complete");
        println!("finished: {done}");

        let bytes = fs::read(&dest).expect("read result");
        assert_eq!(&bytes[..4], b"GGUF", "not a GGUF file");
        assert!(bytes.len() as u64 > partial, "resume added nothing");
        assert!(!part.exists(), ".part should be renamed away on success");
        println!("final size {} bytes, resumed from {partial}", bytes.len());

        let _ = fs::remove_dir_all(&dir);
    }
}
