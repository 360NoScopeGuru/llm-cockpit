/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! `llama-server` process manager.
//!
//! Resolves a llama.cpp `llama-server` binary (preferring a CUDA build), launches
//! a selected GGUF model with configurable GPU layers / context, and tracks the
//! process lifecycle + health. This is what turns the cockpit from an inspector
//! into a runner. One server at a time in v1.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// A discovered `llama-server` binary and what backend it targets.
#[derive(Debug, Clone, Serialize)]
pub struct LlamaBinary {
    pub path: String,
    pub label: String,
    pub backend: String, // "cuda" | "vulkan" | "cpu" | "unknown"
    pub source: String,  // "lm-studio" | "path"
    /// Higher = preferred (CUDA > Vulkan > CPU).
    pub rank: u32,
}

/// Launch parameters for a server instance (sent from the frontend).
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaServerConfig {
    pub model_path: String,
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Optional explicit binary; otherwise the best-ranked one is used.
    #[serde(default)]
    pub binary_path: Option<String>,
    #[serde(default)]
    pub flash_attn: bool,
    /// KV cache element type (`f16` default, `q8_0` ≈ half the KV memory for
    /// the same context, `q4_0` ≈ a quarter). Trading KV precision for context
    /// length is the cheapest way to grow the usable window on fixed VRAM.
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    /// Slide the window instead of halting when a conversation fills the
    /// context. llama.cpp defaults this OFF, which makes long chats die with
    /// `finish_reason: "length"` rather than degrade gracefully.
    #[serde(default)]
    pub context_shift: bool,

    /// Draft model for speculative decoding. When set, llama-server runs this
    /// small model ahead of the main one and verifies several of its guesses
    /// per forward pass, so accepted tokens cost almost nothing.
    ///
    /// The remaining `draft_*` fields are ignored unless this is set.
    #[serde(default)]
    pub draft_model_path: Option<String>,
    /// Tokens to draft per step (llama.cpp's default is 3). Higher wins more
    /// when the draft is accurate and wastes more when it is not.
    #[serde(default)]
    pub draft_n_max: Option<u32>,
    #[serde(default)]
    pub draft_n_min: Option<u32>,
    /// Minimum draft-token probability to bother speculating on.
    #[serde(default)]
    pub draft_p_min: Option<f32>,
    /// GPU layers for the draft. A draft on the CPU is slower than the model
    /// it is drafting for, so this normally wants to be all of them.
    #[serde(default)]
    pub draft_n_gpu_layers: Option<u32>,
    /// The draft keeps its own KV cache, quantizable independently.
    #[serde(default)]
    pub draft_cache_type_k: Option<String>,
    #[serde(default)]
    pub draft_cache_type_v: Option<String>,

    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_port() -> u16 {
    8137
}

/// Everything off, port at the app default. Exists so that callers building a
/// config can name only the fields they care about — this struct grows a field
/// every time llama-server gains a launch knob, and without this every
/// construction site has to be touched for a flag it does not use.
impl Default for LlamaServerConfig {
    fn default() -> Self {
        LlamaServerConfig {
            model_path: String::new(),
            n_gpu_layers: None,
            ctx_size: None,
            port: default_port(),
            binary_path: None,
            flash_attn: false,
            cache_type_k: None,
            cache_type_v: None,
            context_shift: false,
            draft_model_path: None,
            draft_n_max: None,
            draft_n_min: None,
            draft_p_min: None,
            draft_n_gpu_layers: None,
            draft_cache_type_k: None,
            draft_cache_type_v: None,
            extra_args: Vec::new(),
        }
    }
}

/// Snapshot of the manager state for the UI.
#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub running: bool,
    pub health: String, // "starting" | "loading" | "ok" | "error" | "unreachable" | "stopped"
    pub pid: Option<u32>,
    pub base_url: Option<String>,
    pub model_path: Option<String>,
    pub binary_label: Option<String>,
    pub uptime_ms: Option<u128>,
    pub error: Option<String>,
}

impl ServerStatus {
    fn stopped() -> Self {
        ServerStatus {
            running: false,
            health: "stopped".into(),
            pid: None,
            base_url: None,
            model_path: None,
            binary_label: None,
            uptime_ms: None,
            error: None,
        }
    }
}

/// Inference-side metrics scraped from llama-server's Prometheus `/metrics`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct InferenceMetrics {
    pub prompt_tokens_total: f64,
    pub predicted_tokens_total: f64,
    /// Current prompt-processing (prefill) speed, tokens/sec.
    pub prompt_tokens_per_sec: f64,
    /// Current generation (decode) speed, tokens/sec.
    pub predicted_tokens_per_sec: f64,
    /// KV cache fill fraction, 0.0–1.0.
    pub kv_cache_usage_ratio: f64,
    pub kv_cache_tokens: f64,
    pub requests_processing: f64,
}

struct RunningServer {
    child: Child,
    base_url: String,
    model_path: String,
    binary_label: String,
    started: Instant,
    log_path: PathBuf,
}

/// Tauri-managed single-server manager.
pub struct LlamaManager {
    inner: Mutex<Option<RunningServer>>,
}

impl LlamaManager {
    pub fn new() -> Self {
        LlamaManager {
            inner: Mutex::new(None),
        }
    }

    pub fn start(&self, cfg: LlamaServerConfig) -> Result<ServerStatus, String> {
        let mut guard = self.inner.lock().unwrap();
        // v1: single server — replace any existing one.
        if let Some(mut old) = guard.take() {
            let _ = old.child.kill();
            let _ = old.child.wait();
        }

        // Explicit binary > persisted preference (if it still exists) > best-ranked.
        let preferred = cfg
            .binary_path
            .clone()
            .or_else(|| crate::settings::load().preferred_binary);
        let binary = match preferred {
            Some(p) if Path::new(&p).is_file() => resolve_binaries()
                .into_iter()
                .find(|b| b.path.eq_ignore_ascii_case(&p))
                .unwrap_or(LlamaBinary {
                    label: format!("custom ({p})"),
                    backend: "unknown".into(),
                    source: "path".into(),
                    rank: 0,
                    path: p,
                }),
            _ => best_binary().ok_or_else(|| {
                "no llama-server binary found (looked on PATH and in LM Studio backends)"
                    .to_string()
            })?,
        };

        if !Path::new(&cfg.model_path).is_file() {
            return Err(format!("model file not found: {}", cfg.model_path));
        }

        let log_path = std::env::temp_dir().join("tokamak-llama-server.log");
        let log = File::create(&log_path).map_err(|e| format!("cannot open log: {e}"))?;
        let log2 = log
            .try_clone()
            .map_err(|e| format!("cannot clone log handle: {e}"))?;

        let args = build_args(&cfg);
        let mut cmd = Command::new(&binary.path);
        cmd.args(&args);

        // Resolve dependency DLLs. LM Studio's CUDA/Vulkan builds keep their
        // runtime DLLs (cudart, cublas, …) in a sibling `vendor/<name>/` dir
        // rather than next to the exe, so we prepend those to the child's PATH.
        let bin_path = PathBuf::from(&binary.path);
        let search = dll_search_dirs(&bin_path);
        if let Some(dir) = bin_path.parent() {
            cmd.current_dir(dir);
        }
        let orig_path = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = search.clone();
        paths.extend(std::env::split_paths(&orig_path));
        if let Ok(new_path) = std::env::join_paths(&paths) {
            cmd.env("PATH", new_path);
        }

        #[cfg(unix)]
        {
            let lib_env = if cfg!(target_os = "macos") { "DYLD_LIBRARY_PATH" } else { "LD_LIBRARY_PATH" };
            let orig_lib = std::env::var_os(lib_env).unwrap_or_default();
            let mut libs = search.clone();
            libs.extend(std::env::split_paths(&orig_lib));
            if let Ok(new_lib) = std::env::join_paths(&libs) {
                cmd.env(lib_env, new_lib);
            }
        }

        cmd.stdout(Stdio::from(log)).stderr(Stdio::from(log2));
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd.spawn().map_err(|e| format!("failed to spawn: {e}"))?;

        // Give it a moment; if it dies immediately, surface the log tail.
        std::thread::sleep(Duration::from_millis(500));
        if let Ok(Some(status)) = child.try_wait() {
            let tail = read_log_tail(&log_path, 2500);
            return Err(format!(
                "llama-server exited immediately ({status}):\n{tail}"
            ));
        }

        let base_url = format!("http://127.0.0.1:{}", cfg.port);
        let mut running = RunningServer {
            child,
            base_url,
            model_path: cfg.model_path.clone(),
            binary_label: binary.label.clone(),
            started: Instant::now(),
            log_path,
        };
        let status = status_of(&mut running, "starting");
        *guard = Some(running);
        Ok(status)
    }

    pub fn stop(&self) -> Result<(), String> {
        let mut guard = self.inner.lock().unwrap();
        if let Some(mut server) = guard.take() {
            server.child.kill().map_err(|e| e.to_string())?;
            let _ = server.child.wait();
        }
        Ok(())
    }

    /// Base URL of the running server, if any.
    pub fn base_url(&self) -> Option<String> {
        let guard = self.inner.lock().unwrap();
        guard.as_ref().map(|s| s.base_url.clone())
    }

    /// Scrape the running server's `/metrics`. Returns None if nothing is
    /// running. The lock is released before the HTTP call so status polling on
    /// another thread isn't blocked.
    pub fn metrics(&self) -> Option<InferenceMetrics> {
        fetch_metrics(&self.base_url()?)
    }

    pub fn status(&self) -> ServerStatus {
        let mut guard = self.inner.lock().unwrap();
        let Some(server) = guard.as_mut() else {
            return ServerStatus::stopped();
        };

        // Did the process exit on its own (e.g. model load failure)?
        if let Ok(Some(exit)) = server.child.try_wait() {
            let tail = read_log_tail(&server.log_path, 2500);
            let mut st = ServerStatus::stopped();
            st.health = "error".into();
            st.model_path = Some(server.model_path.clone());
            st.error = Some(format!("llama-server exited ({exit}):\n{tail}"));
            *guard = None;
            return st;
        }

        let health = probe_health(&server.base_url);
        status_of(server, health)
    }
}

impl Default for LlamaManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Never leave a llama-server orphaned: kill the child when the manager is
/// dropped (app exit, or a panicking test unwinding). Also matters because an
/// orphan holds inherited pipe handles, wedging whatever spawned *us*.
impl Drop for LlamaManager {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(server) = guard.as_mut() {
                let _ = server.child.kill();
                let _ = server.child.wait();
            }
        }
    }
}

fn status_of(server: &mut RunningServer, health: &str) -> ServerStatus {
    ServerStatus {
        running: true,
        health: health.to_string(),
        pid: Some(server.child.id()),
        base_url: Some(server.base_url.clone()),
        model_path: Some(server.model_path.clone()),
        binary_label: Some(server.binary_label.clone()),
        uptime_ms: Some(server.started.elapsed().as_millis()),
        error: None,
    }
}

/// Build the full `llama-server` argument vector (excluding the binary itself).
fn build_args(cfg: &LlamaServerConfig) -> Vec<String> {
    let mut args = vec![
        "--model".into(),
        cfg.model_path.clone(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        cfg.port.to_string(),
        // Expose the Prometheus /metrics endpoint for the inference cockpit.
        "--metrics".into(),
    ];
    if let Some(ngl) = cfg.n_gpu_layers {
        args.push("-ngl".into());
        args.push(ngl.to_string());
    }
    if let Some(c) = cfg.ctx_size {
        args.push("-c".into());
        args.push(c.to_string());
    }
    if cfg.flash_attn {
        args.push("-fa".into());
    }
    // Quantized KV requires flash attention to be active; the binary defaults
    // `-fa auto`, so only force it on when the caller has not already.
    if cfg.cache_type_k.is_some() || cfg.cache_type_v.is_some() {
        if !cfg.flash_attn {
            args.push("-fa".into());
            args.push("on".into());
        }
        if let Some(k) = &cfg.cache_type_k {
            args.push("-ctk".into());
            args.push(k.clone());
        }
        if let Some(v) = &cfg.cache_type_v {
            args.push("-ctv".into());
            args.push(v.clone());
        }
    }
    if cfg.context_shift {
        args.push("--context-shift".into());
    }

    // Speculative decoding. The `--spec-draft-*` spelling is the current one:
    // llama.cpp *removed* `--draft-max`/`--draft-min`, and every build this app
    // can reach (managed b10242, LM Studio 2.23.1 and 2.24.0) keeps them only
    // as stubs that error with "the argument has been removed". Emitting the
    // old names would fail on all of them, so there is deliberately no
    // fallback spelling here — a binary old enough to need one would fail
    // loudly rather than silently ignoring the draft.
    if let Some(draft) = &cfg.draft_model_path {
        // REQUIRED, and easy to miss: --spec-type defaults to `none`. Passing a
        // draft model without it loads the draft, spends its VRAM (~1 GiB for a
        // 0.6B Q8_0) and then never speculates — the slot reports
        // `"speculative": false` and no draft counters come back at all. Only a
        // verbose log says why:
        //     spec common_specu: no implementations specified
        // `draft-simple` is the implementation that uses a draft model.
        // llama.cpp also offers draft-free n-gram types (ngram-simple,
        // ngram-cache, ...) which need no second model; those are a separate
        // feature, not a substitute here.
        args.push("--spec-type".into());
        args.push("draft-simple".into());
        args.push("--spec-draft-model".into());
        args.push(draft.clone());
        if let Some(n) = cfg.draft_n_max {
            args.push("--spec-draft-n-max".into());
            args.push(n.to_string());
        }
        if let Some(n) = cfg.draft_n_min {
            args.push("--spec-draft-n-min".into());
            args.push(n.to_string());
        }
        if let Some(p) = cfg.draft_p_min {
            args.push("--spec-draft-p-min".into());
            args.push(p.to_string());
        }
        if let Some(ngl) = cfg.draft_n_gpu_layers {
            args.push("--spec-draft-ngl".into());
            args.push(ngl.to_string());
        }
        // Quantized KV needs flash attention, same as the main cache above;
        // by this point `-fa` has already been forced on if either cache is
        // quantized, so only add it if the draft is the sole reason.
        let draft_kv_quantized =
            cfg.draft_cache_type_k.is_some() || cfg.draft_cache_type_v.is_some();
        let main_kv_quantized = cfg.cache_type_k.is_some() || cfg.cache_type_v.is_some();
        if draft_kv_quantized && !main_kv_quantized && !cfg.flash_attn {
            args.push("-fa".into());
            args.push("on".into());
        }
        if let Some(k) = &cfg.draft_cache_type_k {
            args.push("--spec-draft-type-k".into());
            args.push(k.clone());
        }
        if let Some(v) = &cfg.draft_cache_type_v {
            args.push("--spec-draft-type-v".into());
            args.push(v.clone());
        }
    }

    args.extend(cfg.extra_args.iter().cloned());
    args
}

/// Fetch + parse the server's Prometheus `/metrics`.
fn fetch_metrics(base_url: &str) -> Option<InferenceMetrics> {
    let url = format!("{base_url}/metrics");
    let body = ureq::get(&url)
        .timeout(Duration::from_millis(800))
        .call()
        .ok()?
        .into_string()
        .ok()?;
    Some(parse_metrics(&body))
}

/// Parse the subset of llama-server Prometheus metrics we display. Lines are
/// `name value` (or `name{labels} value`); comments start with `#`.
fn parse_metrics(body: &str) -> InferenceMetrics {
    let mut m = InferenceMetrics::default();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(key), Some(val)) = (parts.next(), parts.next()) else {
            continue;
        };
        // Strip any `{label="…"}` suffix.
        let key = key.split('{').next().unwrap_or(key);
        let v: f64 = val.parse().unwrap_or(0.0);
        match key {
            "llamacpp:prompt_tokens_total" => m.prompt_tokens_total = v,
            "llamacpp:tokens_predicted_total" => m.predicted_tokens_total = v,
            "llamacpp:prompt_tokens_seconds" => m.prompt_tokens_per_sec = v,
            "llamacpp:predicted_tokens_seconds" => m.predicted_tokens_per_sec = v,
            "llamacpp:kv_cache_usage_ratio" => m.kv_cache_usage_ratio = v,
            "llamacpp:kv_cache_tokens" => m.kv_cache_tokens = v,
            "llamacpp:requests_processing" => m.requests_processing = v,
            _ => {}
        }
    }
    m
}

/// Probe `GET /health`. 200 => ok, 503 => still loading the model.
fn probe_health(base_url: &str) -> &'static str {
    let url = format!("{base_url}/health");
    match ureq::get(&url).timeout(Duration::from_millis(800)).call() {
        Ok(_) => "ok",
        Err(ureq::Error::Status(503, _)) => "loading",
        Err(ureq::Error::Status(_, _)) => "error",
        Err(_) => "unreachable",
    }
}

/// Directories to add to the child's DLL search path: the binary's own dir,
/// plus any `vendor/<name>/` dirs (LM Studio layout) that contain DLLs.
fn dll_search_dirs(binary_path: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let Some(bin_dir) = binary_path.parent() else {
        return dirs;
    };
    dirs.push(bin_dir.to_path_buf());

    // LM Studio: <backends>/<backend>/llama-server.exe, DLLs in <backends>/vendor/*/.
    if let Some(backends) = bin_dir.parent() {
        let vendor = backends.join("vendor");
        if let Ok(entries) = std::fs::read_dir(&vendor) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() && dir_has_dll(&p) {
                    dirs.push(p);
                }
            }
        }
    }
    dirs
}

fn dir_has_dll(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                e.path()
                    .extension()
                    .map(|x| {
                        x.eq_ignore_ascii_case("dll")
                            || x.eq_ignore_ascii_case("so")
                            || x.eq_ignore_ascii_case("dylib")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn read_log_tail(path: &Path, max_bytes: u64) -> String {
    let Ok(mut f) = File::open(path) else {
        return String::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(max_bytes);
    let _ = f.seek(SeekFrom::Start(start));
    let mut buf = String::new();
    let _ = f.read_to_string(&mut buf);
    buf
}

// ---- binary resolution ----

const EXE: &str = if cfg!(windows) {
    "llama-server.exe"
} else {
    "llama-server"
};

/// All `llama-server` binaries we can find, best-ranked first.
pub fn resolve_binaries() -> Vec<LlamaBinary> {
    let mut found: Vec<LlamaBinary> = Vec::new();

    // Tokamak's own managed runtime, if the user has installed one. Ranked
    // above LM Studio's: it was chosen for this machine on purpose.
    if let Some(exe) = crate::runtime::installed_binary() {
        // The archive unpacks into a folder called "runtime", which says
        // nothing about the backend — use the id recorded at install time.
        // classify_backend reads the trailing dash-segment as a version, so
        // feed it "<backend>-<release tag>" to get "Vulkan b10242" not "Vulkan vulkan".
        let hint = match (
            crate::runtime::installed_backend(),
            crate::runtime::status().version,
        ) {
            (Some(b), Some(v)) => format!("{b}-{v}"),
            (Some(b), None) => b,
            _ => "unknown".into(),
        };
        let (rank, backend, label) = classify_backend(&hint);
        found.push(LlamaBinary {
            path: exe.to_string_lossy().into_owned(),
            label: format!("{label} (managed)"),
            backend,
            source: "managed".into(),
            rank: rank + 100,
        });
    }

    // LM Studio bundles per-backend builds under extensions/backends/<name>/.
    if let Some(home) = dirs::home_dir() {
        let backends = home.join(".lmstudio").join("extensions").join("backends");
        if backends.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&backends) {
                for e in entries.flatten() {
                    let dir = e.path();
                    let exe = dir.join(EXE);
                    if exe.is_file() {
                        let name = dir
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let (rank, backend, label) = classify_backend(&name);
                        found.push(LlamaBinary {
                            path: exe.to_string_lossy().into_owned(),
                            label: format!("{label} (LM Studio)"),
                            backend,
                            source: "lm-studio".into(),
                            rank,
                        });
                    }
                }
            }
        }
    }

    // Anything on PATH.
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let exe = dir.join(EXE);
            if exe.is_file() {
                found.push(LlamaBinary {
                    path: exe.to_string_lossy().into_owned(),
                    label: "llama-server (PATH)".into(),
                    backend: "unknown".into(),
                    source: "path".into(),
                    rank: 250,
                });
            }
        }
    }

    found.sort_by(|a, b| b.rank.cmp(&a.rank).then(b.label.cmp(&a.label)));
    found
}

pub fn best_binary() -> Option<LlamaBinary> {
    resolve_binaries().into_iter().next()
}

/// Rank + label a backend from an LM Studio backend directory name.
fn classify_backend(dir_name: &str) -> (u32, String, String) {
    let n = dir_name.to_lowercase();
    // Trailing version, e.g. "...-2.24.0", used for the label.
    let version = n.rsplit('-').next().unwrap_or("").to_string();
    if n.contains("cuda12") {
        (420, "cuda".into(), format!("CUDA 12 {version}"))
    } else if n.contains("cuda") {
        (400, "cuda".into(), format!("CUDA {version}"))
    } else if n.contains("vulkan") {
        (300, "vulkan".into(), format!("Vulkan {version}"))
    } else if n.contains("avx") || n.contains("cpu") {
        (100, "cpu".into(), format!("CPU {version}"))
    } else {
        (50, "unknown".into(), dir_name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_expected_args() {
        let cfg = LlamaServerConfig {
            model_path: "C:/models/foo.gguf".into(),
            n_gpu_layers: Some(999),
            ctx_size: Some(8192),
            port: 8137,
            binary_path: None,
            flash_attn: true,
            cache_type_k: None,
            cache_type_v: None,
            context_shift: false,
            extra_args: vec!["--verbose".into()],
            ..Default::default()
        };
        let args = build_args(&cfg);
        assert_eq!(args[0], "--model");
        assert_eq!(args[1], "C:/models/foo.gguf");
        assert!(args.windows(2).any(|w| w[0] == "-ngl" && w[1] == "999"));
        assert!(args.windows(2).any(|w| w[0] == "-c" && w[1] == "8192"));
        assert!(args.contains(&"-fa".to_string()));
        assert!(args.contains(&"--verbose".to_string()));
    }

    /// Helper: find the value following a flag, if the flag is present.
    fn val_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].as_str())
    }

    #[test]
    fn speculative_decoding_uses_the_current_flag_spelling() {
        let cfg = LlamaServerConfig {
            model_path: "target.gguf".into(),
            draft_model_path: Some("draft.gguf".into()),
            draft_n_max: Some(5),
            draft_n_min: Some(1),
            draft_n_gpu_layers: Some(99),
            ..Default::default()
        };
        let args = build_args(&cfg);
        // Without this the draft loads, costs VRAM, and never speculates:
        // --spec-type defaults to `none`. Measured on a real pair — omitting
        // it produced `"speculative": false` and zero draft counters, while
        // adding it produced 58.8% acceptance and a 1.22x decode speedup.
        assert_eq!(val_after(&args, "--spec-type"), Some("draft-simple"));
        assert_eq!(val_after(&args, "--spec-draft-model"), Some("draft.gguf"));
        assert_eq!(val_after(&args, "--spec-draft-n-max"), Some("5"));
        assert_eq!(val_after(&args, "--spec-draft-n-min"), Some("1"));
        assert_eq!(val_after(&args, "--spec-draft-ngl"), Some("99"));

        // llama.cpp removed these; emitting one would make the server exit
        // with "the argument has been removed" instead of speculating.
        for dead in ["--draft-max", "--draft-min", "--draft", "--draft-n"] {
            assert!(
                !args.iter().any(|a| a == dead),
                "must not emit the removed flag {dead}"
            );
        }
    }

    #[test]
    fn no_draft_flags_without_a_draft_model() {
        // The knobs are meaningless alone, and passing them without
        // --spec-draft-model would be a launch failure rather than a no-op.
        let cfg = LlamaServerConfig {
            model_path: "m.gguf".into(),
            draft_n_max: Some(5),
            draft_cache_type_k: Some("q8_0".into()),
            ..Default::default()
        };
        let args = build_args(&cfg);
        assert!(!args.iter().any(|a| a.starts_with("--spec-draft")));
        assert!(!args.iter().any(|a| a == "--spec-type"));
    }

    #[test]
    fn a_quantized_draft_cache_forces_flash_attention_on() {
        let cfg = LlamaServerConfig {
            model_path: "m.gguf".into(),
            draft_model_path: Some("d.gguf".into()),
            draft_cache_type_k: Some("q8_0".into()),
            draft_cache_type_v: Some("q8_0".into()),
            ..Default::default()
        };
        let args = build_args(&cfg);
        assert_eq!(val_after(&args, "-fa"), Some("on"), "quantized KV needs -fa");
        assert_eq!(val_after(&args, "--spec-draft-type-k"), Some("q8_0"));
        assert_eq!(val_after(&args, "--spec-draft-type-v"), Some("q8_0"));
    }

    /// `-fa` must not be passed twice when both caches are quantized: the main
    /// cache already forces it on further up.
    #[test]
    fn flash_attention_is_only_forced_once() {
        let cfg = LlamaServerConfig {
            model_path: "m.gguf".into(),
            cache_type_k: Some("q8_0".into()),
            draft_model_path: Some("d.gguf".into()),
            draft_cache_type_k: Some("q8_0".into()),
            ..Default::default()
        };
        let args = build_args(&cfg);
        assert_eq!(
            args.iter().filter(|a| *a == "-fa").count(),
            1,
            "got: {args:?}"
        );
    }

    #[test]
    fn omits_optional_args_when_unset() {
        let cfg = LlamaServerConfig {
            model_path: "m.gguf".into(),
            n_gpu_layers: None,
            ctx_size: None,
            port: 8137,
            binary_path: None,
            flash_attn: false,
            cache_type_k: None,
            cache_type_v: None,
            context_shift: false,
            extra_args: vec![],
            ..Default::default()
        };
        let args = build_args(&cfg);
        assert!(!args.contains(&"-ngl".to_string()));
        assert!(!args.contains(&"-c".to_string()));
        assert!(!args.contains(&"-fa".to_string()));
    }

    #[test]
    fn kv_quant_and_context_shift_args() {
        let cfg = LlamaServerConfig {
            model_path: "m.gguf".into(),
            n_gpu_layers: None,
            ctx_size: Some(32768),
            port: 8137,
            binary_path: None,
            flash_attn: false,
            cache_type_k: Some("q8_0".into()),
            cache_type_v: Some("q8_0".into()),
            context_shift: true,
            extra_args: vec![],
            ..Default::default()
        };
        let args = build_args(&cfg);
        assert!(args.windows(2).any(|w| w[0] == "-ctk" && w[1] == "q8_0"));
        assert!(args.windows(2).any(|w| w[0] == "-ctv" && w[1] == "q8_0"));
        assert!(args.contains(&"--context-shift".to_string()));
        // Quantized KV is only valid with flash attention, so it must be forced
        // on even though the caller left flash_attn false.
        assert!(args.windows(2).any(|w| w[0] == "-fa" && w[1] == "on"));
    }

    #[test]
    fn parses_prometheus_metrics() {
        let body = "\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 42
llamacpp:tokens_predicted_total 128
llamacpp:prompt_tokens_seconds 512.5
llamacpp:predicted_tokens_seconds 87.3
llamacpp:kv_cache_usage_ratio 0.25
llamacpp:kv_cache_tokens 1024
llamacpp:requests_processing 1
";
        let m = parse_metrics(body);
        assert_eq!(m.prompt_tokens_total, 42.0);
        assert_eq!(m.predicted_tokens_total, 128.0);
        assert_eq!(m.predicted_tokens_per_sec, 87.3);
        assert_eq!(m.kv_cache_usage_ratio, 0.25);
        assert_eq!(m.requests_processing, 1.0);
    }

    #[test]
    fn parses_metrics_with_labels() {
        let m = parse_metrics("llamacpp:predicted_tokens_seconds{model=\"x\"} 12.5\n");
        assert_eq!(m.predicted_tokens_per_sec, 12.5);
    }

    #[test]
    fn ranks_cuda_over_vulkan_over_cpu() {
        let cuda = classify_backend("llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.24.0").0;
        let vulkan = classify_backend("llama.cpp-win-x86_64-vulkan-avx2-2.23.1").0;
        let cpu = classify_backend("llama.cpp-win-x86_64-avx2-2.23.1").0;
        assert!(cuda > vulkan && vulkan > cpu);
    }

    /// Real launch of the small 4B model via the resolved (CUDA) binary.
    /// Ignored by default (hardware + model dependent); run with:
    ///   cargo test -- --ignored --nocapture launch_real_model
    #[test]
    #[ignore]
    fn launch_real_model() {
        let Some(home) = dirs::home_dir() else {
            return;
        };
        let model = home.join(".lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf");
        if !model.is_file() {
            eprintln!("test model not present, skipping");
            return;
        }

        println!("resolved binaries:");
        for b in resolve_binaries() {
            println!("  [{}] {} -> {}", b.rank, b.label, b.path);
        }

        let mgr = LlamaManager::new();
        let cfg = LlamaServerConfig {
            model_path: model.to_string_lossy().into_owned(),
            n_gpu_layers: Some(999),
            ctx_size: Some(4096),
            port: 8137,
            binary_path: None,
            flash_attn: false,
            cache_type_k: None,
            cache_type_v: None,
            context_shift: false,
            extra_args: vec![],
            ..Default::default()
        };

        let started = mgr.start(cfg).expect("start should succeed");
        println!("started with {:?}", started.binary_label);

        let mut health = String::new();
        for _ in 0..90 {
            let st = mgr.status();
            if let Some(err) = &st.error {
                panic!("server errored: {err}");
            }
            health = st.health.clone();
            println!("health={health} uptime_ms={:?}", st.uptime_ms);
            if health == "ok" {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        assert_eq!(health, "ok", "server should become healthy");

        // Confirm the model actually loaded via /props.
        let props_ok = ureq::get("http://127.0.0.1:8137/props")
            .timeout(Duration::from_secs(5))
            .call()
            .is_ok();
        println!("/props reachable: {props_ok}");
        assert!(props_ok, "/props should be reachable when healthy");

        // Generate a few tokens so the metrics endpoint has real data.
        let _ = ureq::post("http://127.0.0.1:8137/completion")
            .timeout(Duration::from_secs(30))
            .set("Content-Type", "application/json")
            .send_string(
                r#"{"prompt":"Count from one to five:","n_predict":24,"stream":false}"#,
            );

        let metrics = fetch_metrics("http://127.0.0.1:8137").expect("metrics");
        println!(
            "metrics: predicted_total={} decode={:.1} tok/s, prefill={:.1} tok/s, kv={:.1}%",
            metrics.predicted_tokens_total,
            metrics.predicted_tokens_per_sec,
            metrics.prompt_tokens_per_sec,
            metrics.kv_cache_usage_ratio * 100.0,
        );
        assert!(
            metrics.predicted_tokens_total > 0.0,
            "should have generated tokens"
        );

        mgr.stop().expect("stop should succeed");
        println!("stopped");
    }
}

#[cfg(test)]
mod discovery_probe {
    use super::*;

    /// What this machine's binary discovery actually resolves to, best first.
    /// Ignored by default (machine-dependent); run with:
    ///   cargo test -- --ignored --nocapture what_binary_would_we_use
    #[test]
    #[ignore]
    fn what_binary_would_we_use() {
        let all = resolve_binaries();
        println!("\n--- {} binary(ies), best first ---", all.len());
        for b in &all {
            println!(
                "  rank {:>4}  {:<8} {:<28} {}",
                b.rank, b.backend, b.label, b.path
            );
        }
        if let Some(best) = best_binary() {
            println!("\nbest_binary() -> {} [{}]", best.label, best.backend);
            let dirs = dll_search_dirs(std::path::Path::new(&best.path));
            println!("dll search dirs ({}):", dirs.len());
            for d in dirs {
                println!("   {}", d.display());
            }
        }
    }
}
