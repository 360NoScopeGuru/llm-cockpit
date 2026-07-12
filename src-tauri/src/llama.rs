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
    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_port() -> u16 {
    8137
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

        let binary = match &cfg.binary_path {
            Some(p) => LlamaBinary {
                label: format!("custom ({p})"),
                backend: "unknown".into(),
                source: "path".into(),
                rank: 0,
                path: p.clone(),
            },
            None => best_binary().ok_or_else(|| {
                "no llama-server binary found (looked on PATH and in LM Studio backends)"
                    .to_string()
            })?,
        };

        if !Path::new(&cfg.model_path).is_file() {
            return Err(format!("model file not found: {}", cfg.model_path));
        }

        let log_path = std::env::temp_dir().join("llm-cockpit-llama-server.log");
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
        let orig_path = std::env::var("PATH").unwrap_or_default();
        let joined: Vec<String> = search
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        cmd.env("PATH", format!("{};{}", joined.join(";"), orig_path));

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
    args.extend(cfg.extra_args.iter().cloned());
    args
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
                    .map(|x| x.eq_ignore_ascii_case("dll"))
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
            extra_args: vec!["--verbose".into()],
        };
        let args = build_args(&cfg);
        assert_eq!(args[0], "--model");
        assert_eq!(args[1], "C:/models/foo.gguf");
        assert!(args.windows(2).any(|w| w[0] == "-ngl" && w[1] == "999"));
        assert!(args.windows(2).any(|w| w[0] == "-c" && w[1] == "8192"));
        assert!(args.contains(&"-fa".to_string()));
        assert!(args.contains(&"--verbose".to_string()));
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
            extra_args: vec![],
        };
        let args = build_args(&cfg);
        assert!(!args.contains(&"-ngl".to_string()));
        assert!(!args.contains(&"-c".to_string()));
        assert!(!args.contains(&"-fa".to_string()));
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
            extra_args: vec![],
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

        mgr.stop().expect("stop should succeed");
        println!("stopped");
    }
}
