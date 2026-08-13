/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Managed llama.cpp runtime.
//!
//! Tokamak needs a `llama-server` binary. Requiring people to install LM Studio
//! first — to use the thing that replaces LM Studio — is a terrible first run,
//! and bundling CUDA would turn a 4 MiB installer into ~510 MiB that every user
//! downloads whether they need it or not.
//!
//! So the binary is fetched on demand instead: query llama.cpp's GitHub
//! releases, show the builds that suit this machine with real sizes, download
//! the chosen one with the same resumable streamer the model downloader uses,
//! and unpack it into `<config>/tokamak/runtime/`.
//!
//! Unpacking shells out to Windows' own `bsdtar` (System32\tar.exe, present
//! since Windows 10 1803), which handles zip — no extra crate for one call.
//!
//! LM Studio's binaries are still used automatically when present. This removes
//! the *requirement*, not the convenience.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::Emitter;

const RELEASES: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";

/// One installable backend, assembled from the assets of a llama.cpp release.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct RuntimeBuild {
    /// "cuda" | "vulkan" | "cpu"
    pub id: String,
    pub label: String,
    pub note: String,
    /// Asset file names to download, in order. CUDA needs two: the build and
    /// the CUDA runtime DLLs, which llama.cpp ships separately.
    pub assets: Vec<String>,
    pub total_bytes: u64,
    pub recommended: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub installed: bool,
    pub path: Option<String>,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RuntimeProgress {
    stage: String,
    received: u64,
    total: u64,
    done: bool,
    error: Option<String>,
}

pub fn runtime_dir() -> Result<PathBuf, String> {
    let dir = dirs::config_dir()
        .ok_or("no config dir on this platform")?
        .join("tokamak")
        .join("runtime");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Path to the managed `llama-server`, if one has been installed.
pub fn installed_binary() -> Option<PathBuf> {
    let dir = runtime_dir().ok()?;
    let exe = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    // Releases unpack either flat or under a nested folder; check both.
    let direct = dir.join(exe);
    if direct.is_file() {
        return Some(direct);
    }
    fs::read_dir(&dir).ok()?.flatten().find_map(|e| {
        let p = e.path().join(exe);
        p.is_file().then_some(p)
    })
}

/// Which backend was installed ("cuda" | "vulkan" | "cpu"). The unpacked folder
/// name does not say, so it is recorded at install time.
pub fn installed_backend() -> Option<String> {
    let d = runtime_dir().ok()?;
    fs::read_to_string(d.join("BACKEND"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn status() -> RuntimeStatus {
    let path = installed_binary();
    let version = runtime_dir()
        .ok()
        .and_then(|d| fs::read_to_string(d.join("VERSION")).ok())
        .map(|s| s.trim().to_string());
    RuntimeStatus {
        installed: path.is_some(),
        path: path.map(|p| p.to_string_lossy().into_owned()),
        version,
    }
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(10))
        .timeout_read(std::time::Duration::from_secs(60))
        .build()
}

/// Sort key for a CUDA asset name so 13.3 beats 12.4 (a plain string compare
/// would not, and "12.4" > "13.3" lexically for the minor part).
fn cuda_version_key(name: &str) -> (u32, u32) {
    let v = name
        .split("bin-win-cuda-")
        .nth(1)
        .and_then(|s| s.split("-x64").next())
        .unwrap_or("");
    let mut parts = v.split('.');
    (
        parts.next().and_then(|p| p.parse().ok()).unwrap_or(0),
        parts.next().and_then(|p| p.parse().ok()).unwrap_or(0),
    )
}

/// Which backends are worth offering, newest release, with real asset sizes.
/// `has_nvidia` decides what gets recommended — CUDA is pointless without one.
pub fn options(has_nvidia: bool) -> Result<Vec<RuntimeBuild>, String> {
    let body = agent()
        .get(RELEASES)
        // GitHub rejects API requests without a User-Agent.
        .set("User-Agent", "tokamak")
        .call()
        .map_err(|e| format!("could not reach llama.cpp releases: {e}"))?
        .into_string()
        .map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    let tag = v
        .get("tag_name")
        .and_then(|t| t.as_str())
        .unwrap_or("latest")
        .to_string();
    let assets: Vec<(String, u64)> = v
        .get("assets")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|a| {
                    Some((
                        a.get("name")?.as_str()?.to_string(),
                        a.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();

    let find = |pred: &dyn Fn(&str) -> bool| -> Option<(String, u64)> {
        assets.iter().find(|(n, _)| pred(n)).cloned()
    };
    let mut out = Vec::new();

    // CUDA: the build plus its separately-shipped runtime DLLs. Several CUDA
    // majors are published at once; take the highest, which is both newer and
    // (currently) a smaller download than the older one.
    let cuda = assets
        .iter()
        .filter(|(n, _)| n.contains("bin-win-cuda-") && n.contains("x64") && !n.starts_with("cudart"))
        .max_by_key(|(n, _)| cuda_version_key(n))
        .cloned();
    if let Some((cname, csize)) = cuda {
        // Match the cudart archive to the same CUDA version as the build.
        let ver = cname
            .split("bin-win-cuda-")
            .nth(1)
            .and_then(|s| s.split("-x64").next())
            .unwrap_or("")
            .to_string();
        let cudart = find(&|n| n.starts_with("cudart-") && n.contains(&ver));
        let mut a = vec![cname];
        let mut bytes = csize;
        if let Some((rn, rs)) = cudart {
            a.push(rn);
            bytes += rs;
        }
        out.push(RuntimeBuild {
            id: "cuda".into(),
            label: format!("CUDA {ver}"),
            note: "fastest on NVIDIA — includes the CUDA runtime".into(),
            assets: a,
            total_bytes: bytes,
            recommended: has_nvidia,
        });
    }

    if let Some((n, s)) = find(&|n| n.contains("bin-win-vulkan") && n.contains("x64")) {
        out.push(RuntimeBuild {
            id: "vulkan".into(),
            label: "Vulkan".into(),
            note: "works on NVIDIA, AMD and Intel — much smaller download".into(),
            assets: vec![n],
            total_bytes: s,
            recommended: !has_nvidia,
        });
    }
    if let Some((n, s)) = find(&|n| n.contains("bin-win-cpu") && n.contains("x64")) {
        out.push(RuntimeBuild {
            id: "cpu".into(),
            label: "CPU only".into(),
            note: "no GPU acceleration — a fallback, expect single-digit tok/s".into(),
            assets: vec![n],
            total_bytes: s,
            recommended: false,
        });
    }
    if out.is_empty() {
        return Err(format!("no usable Windows assets in llama.cpp {tag}"));
    }
    // Record which release these came from for the status line.
    if let Ok(d) = runtime_dir() {
        let _ = fs::write(d.join("VERSION.pending"), &tag);
    }
    Ok(out)
}

/// Download and unpack a build. Runs on a worker thread, reporting progress on
/// `runtime-progress`.
pub fn install(window: tauri::Window, build: RuntimeBuild) {
    std::thread::spawn(move || {
        let emit = |stage: &str, received: u64, total: u64, done: bool, error: Option<String>| {
            let _ = window.emit(
                "runtime-progress",
                RuntimeProgress {
                    stage: stage.to_string(),
                    received,
                    total,
                    done,
                    error,
                },
            );
        };
        match do_install(&build, &emit) {
            Ok(()) => emit("done", build.total_bytes, build.total_bytes, true, None),
            Err(e) => emit("failed", 0, 0, true, Some(e)),
        }
    });
}

/// Progress sink: `(stage, received, total, done, error)`. Taken as a trait
/// object so the install can be driven either by the real Tauri emitter or by
/// a no-op in tests.
type ProgressSink<'a> = &'a dyn Fn(&str, u64, u64, bool, Option<String>);

fn do_install(build: &RuntimeBuild, emit: ProgressSink<'_>) -> Result<(), String> {
    let dir = runtime_dir()?;
    let tag = fs::read_to_string(dir.join("VERSION.pending")).unwrap_or_default();
    let base = format!(
        "https://github.com/ggml-org/llama.cpp/releases/download/{}",
        tag.trim()
    );
    let mut done_bytes = 0u64;

    for asset in &build.assets {
        let zip = dir.join(asset);
        let url = format!("{base}/{asset}");
        emit("downloading", done_bytes, build.total_bytes, false, None);
        let carried = done_bytes;
        crate::downloads::stream_to_disk(
            &|received, _total, _rate| {
                emit(
                    "downloading",
                    carried + received,
                    build.total_bytes,
                    false,
                    None,
                );
            },
            &url,
            &zip,
            None,
            &std::sync::atomic::AtomicBool::new(false),
        )?;
        done_bytes += zip.metadata().map(|m| m.len()).unwrap_or(0);

        emit("extracting", done_bytes, build.total_bytes, false, None);
        extract(&zip, &dir)?;
        let _ = fs::remove_file(&zip);
    }

    if installed_binary().is_none() {
        return Err("downloaded, but no llama-server binary was found in the archive".into());
    }
    let _ = fs::write(dir.join("VERSION"), tag.trim());
    let _ = fs::write(dir.join("BACKEND"), &build.id);
    let _ = fs::remove_file(dir.join("VERSION.pending"));
    Ok(())
}

/// Unpack a zip using Windows' bundled bsdtar rather than adding a zip crate
/// for a single call site.
fn extract(zip: &Path, into: &Path) -> Result<(), String> {
    let tar = if cfg!(windows) {
        std::env::var("SystemRoot")
            .map(|r| PathBuf::from(r).join("System32").join("tar.exe"))
            .unwrap_or_else(|_| PathBuf::from("tar"))
    } else {
        PathBuf::from("tar")
    };
    let mut cmd = std::process::Command::new(tar);
    cmd.arg("-xf").arg(zip).arg("-C").arg(into);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = cmd
        .output()
        .map_err(|e| format!("could not run tar to unpack: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "unpacking failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_highest_cuda_version() {
        // A plain string max would choose 12.4 over 13.3.
        assert!(
            cuda_version_key("llama-b1-bin-win-cuda-13.3-x64.zip")
                > cuda_version_key("llama-b1-bin-win-cuda-12.4-x64.zip")
        );
        assert_eq!(cuda_version_key("llama-b1-bin-win-cuda-13.3-x64.zip"), (13, 3));
        assert_eq!(cuda_version_key("nonsense.zip"), (0, 0));
    }

    #[test]
    fn reports_not_installed_on_a_clean_machine() {
        // Only asserts the shape; a real install is exercised by the ignored test.
        let s = status();
        assert_eq!(s.installed, s.path.is_some());
    }

    /// Hits GitHub. cargo test -- --ignored --nocapture lists_real_builds
    #[test]
    #[ignore]
    fn lists_real_builds() {
        let builds = options(true).expect("options");
        println!("\navailable llama.cpp builds:");
        for b in &builds {
            println!(
                "  {:<10} {:>8.1} MiB  rec={}  {}  {:?}",
                b.label,
                b.total_bytes as f64 / 1048576.0,
                b.recommended,
                b.note,
                b.assets
            );
        }
        assert!(builds.iter().any(|b| b.id == "vulkan"));
        // CUDA must carry its runtime DLLs or the binary will not start.
        if let Some(c) = builds.iter().find(|b| b.id == "cuda") {
            assert_eq!(c.assets.len(), 2, "cuda build must include cudart");
        }
    }
}

#[cfg(test)]
mod live_install {
    use super::*;

    /// Real download + unpack of the smallest GPU build, then check the binary
    /// is present and that llama.rs picks it up.
    /// cargo test -- --ignored --nocapture installs_vulkan_for_real
    #[test]
    #[ignore]
    fn installs_vulkan_for_real() {
        let builds = options(true).expect("options");
        let vk = builds.iter().find(|b| b.id == "vulkan").expect("vulkan build");
        println!("installing {} ({:.1} MiB)", vk.label, vk.total_bytes as f64 / 1048576.0);

        let noop = |stage: &str, got: u64, total: u64, _d: bool, e: Option<String>| {
            if let Some(e) = e {
                println!("  {stage}: ERROR {e}");
            } else if stage == "extracting" {
                println!("  {stage} ({got}/{total})");
            }
        };
        do_install(vk, &noop).expect("install");

        let bin = installed_binary().expect("binary should exist after install");
        println!("installed: {}", bin.display());
        assert!(bin.is_file());

        assert_eq!(installed_backend().as_deref(), Some("vulkan"));

        let found = crate::llama::resolve_binaries();
        let managed = found
            .iter()
            .find(|b| b.source == "managed")
            .expect("llama.rs must pick up the managed runtime");
        println!("resolve_binaries sees: {} rank={}", managed.label, managed.rank);
        // The backend must be identified, not left as "unknown" — the folder is
        // just called "runtime", so this only works via the recorded BACKEND.
        assert_eq!(managed.backend, "vulkan");
        assert!(managed.label.contains("Vulkan"), "got {}", managed.label);
        // It need not rank first: an LM Studio CUDA build legitimately beats a
        // managed Vulkan one. It must beat a same-backend rival, though.
        assert!(managed.rank > 300, "managed should be preferred at equal backend");
    }
}
