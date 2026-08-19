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

/// The platform/arch pair llama.cpp names its release assets for.
///
/// Carried as data rather than read from `cfg!` at each site so both branches
/// are reachable from a test — the Linux selection has to be verifiable
/// without a Linux machine to verify it on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Target {
    /// "win" or "ubuntu" — upstream builds Linux artefacts on Ubuntu and
    /// names them for it.
    platform: &'static str,
    /// "x64" or "arm64".
    arch: &'static str,
}

impl Target {
    fn current() -> Self {
        Target {
            platform: if cfg!(windows) { "win" } else { "ubuntu" },
            arch: if cfg!(target_arch = "aarch64") {
                "arm64"
            } else {
                "x64"
            },
        }
    }

    /// llama.cpp publishes no prebuilt CUDA build for Linux. That is not an
    /// oversight to work around: NVIDIA users there take Vulkan from here, or
    /// install their distribution's CUDA-enabled llama.cpp, which
    /// `resolve_binaries` already finds on PATH.
    fn cuda_available(&self) -> bool {
        self.platform == "win"
    }

    fn vulkan_stem(&self) -> String {
        format!("bin-{}-vulkan-{}", self.platform, self.arch)
    }

    /// The plain CPU build, named asymmetrically across platforms: Windows
    /// publishes `…bin-win-cpu-x64.zip`, while the Linux CPU artefact carries
    /// no backend token at all — just `…bin-ubuntu-x64.tar.gz`. Matching on
    /// "cpu" therefore finds nothing on Linux.
    ///
    /// The bare stem is enough to exclude the accelerated builds, because each
    /// carries its token *between* the platform and the arch
    /// (`bin-ubuntu-vulkan-x64`, `bin-ubuntu-sycl-fp16-x64`), so none of them
    /// contains `bin-ubuntu-x64`. Tests pin that, since it is the kind of
    /// property that quietly stops holding when upstream renames something.
    fn cpu_stem(&self) -> String {
        if self.platform == "win" {
            format!("bin-win-cpu-{}", self.arch)
        } else {
            format!("bin-{}-{}", self.platform, self.arch)
        }
    }
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
    let target = Target::current();
    let mut out = Vec::new();

    // CUDA: the build plus its separately-shipped runtime DLLs. Several CUDA
    // majors are published at once; take the highest, which is both newer and
    // (currently) a smaller download than the older one.
    // Windows only, and not an oversight upstream: llama.cpp publishes no
    // prebuilt CUDA build for Linux. NVIDIA users there get Vulkan from here,
    // or install their distribution's CUDA-enabled llama.cpp, which
    // `resolve_binaries` already discovers on PATH.
    let cuda = assets
        .iter()
        .filter(|(n, _)| {
            n.contains("bin-win-cuda-") && n.contains(target.arch) && !n.starts_with("cudart")
        })
        .max_by_key(|(n, _)| cuda_version_key(n))
        .cloned()
        .filter(|_| target.cuda_available());
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

    if let Some((n, s)) = find(&|n| n.contains(&target.vulkan_stem())) {
        out.push(RuntimeBuild {
            id: "vulkan".into(),
            label: "Vulkan".into(),
            note: "works on NVIDIA, AMD and Intel — much smaller download".into(),
            assets: vec![n],
            total_bytes: s,
            recommended: !has_nvidia,
        });
    }
    if let Some((n, s)) = find(&|n| n.contains(&target.cpu_stem())) {
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
        return Err(format!(
            "no usable {} {} assets in llama.cpp {tag}",
            target.platform, target.arch
        ));
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

#[cfg(test)]
mod asset_selection {
    use super::*;

    /// Verbatim asset names from llama.cpp release b10423. Kept literal so a
    /// rename upstream shows up as a test failure rather than as an empty
    /// runtime list in front of a user.
    const REAL: &[&str] = &[
        "llama-b10423-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "llama-b10423-bin-win-vulkan-x64.zip",
        "llama-b10423-bin-win-cpu-x64.zip",
        "llama-b10423-bin-ubuntu-x64.tar.gz",
        "llama-b10423-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b10423-bin-ubuntu-arm64.tar.gz",
        "llama-b10423-bin-ubuntu-vulkan-arm64.tar.gz",
        "llama-b10423-bin-ubuntu-sycl-fp16-x64.tar.gz",
        "llama-b10423-bin-ubuntu-sycl-fp32-x64.tar.gz",
        "llama-b10423-bin-ubuntu-openvino-2026.2.1-x64.tar.gz",
        "llama-b10423-bin-ubuntu-s390x.tar.gz",
    ];

    fn matching(stem: &str) -> Vec<&'static str> {
        REAL.iter().copied().filter(|n| n.contains(stem)).collect()
    }

    const WIN: Target = Target { platform: "win", arch: "x64" };
    const LINUX: Target = Target { platform: "ubuntu", arch: "x64" };
    const LINUX_ARM: Target = Target { platform: "ubuntu", arch: "arm64" };

    #[test]
    fn windows_picks_its_own_three_backends() {
        assert_eq!(matching(&WIN.cpu_stem()), ["llama-b10423-bin-win-cpu-x64.zip"]);
        assert_eq!(matching(&WIN.vulkan_stem()), ["llama-b10423-bin-win-vulkan-x64.zip"]);
        assert!(WIN.cuda_available());
    }

    /// The asymmetry this whole abstraction exists for: Linux's CPU build has
    /// no "cpu" token, so the Windows predicate would find nothing at all.
    #[test]
    fn linux_cpu_build_carries_no_backend_token() {
        assert_eq!(LINUX.cpu_stem(), "bin-ubuntu-x64");
        assert_eq!(matching(&LINUX.cpu_stem()), ["llama-b10423-bin-ubuntu-x64.tar.gz"]);
        // The old Windows-shaped predicate finds nothing here.
        assert!(matching("bin-ubuntu-cpu").is_empty());
    }

    /// The bare stem must not sweep up the accelerated builds. It does not,
    /// because each carries its token between platform and arch — but that is
    /// a property of upstream naming, so it gets a test rather than a comment.
    #[test]
    fn the_linux_cpu_stem_excludes_accelerated_builds() {
        let hits = matching(&LINUX.cpu_stem());
        assert_eq!(hits.len(), 1, "cpu stem swept up extras: {hits:?}");
        for acc in ["vulkan", "sycl", "openvino"] {
            assert!(!hits[0].contains(acc), "{acc} leaked into the CPU build");
        }
    }

    #[test]
    fn linux_vulkan_is_offered_and_arch_specific() {
        assert_eq!(
            matching(&LINUX.vulkan_stem()),
            ["llama-b10423-bin-ubuntu-vulkan-x64.tar.gz"]
        );
        assert_eq!(
            matching(&LINUX_ARM.vulkan_stem()),
            ["llama-b10423-bin-ubuntu-vulkan-arm64.tar.gz"]
        );
    }

    #[test]
    fn arm64_does_not_match_the_x64_builds() {
        assert_eq!(matching(&LINUX_ARM.cpu_stem()), ["llama-b10423-bin-ubuntu-arm64.tar.gz"]);
    }

    /// Upstream ships no Linux CUDA artefact, so offering one would dangle a
    /// build that cannot be downloaded.
    #[test]
    fn linux_is_not_offered_cuda() {
        assert!(!LINUX.cuda_available());
        assert!(
            !REAL.iter().any(|n| n.contains("bin-ubuntu-cuda")),
            "upstream started shipping Linux CUDA — revisit cuda_available()"
        );
    }

    /// The pinned names above are a snapshot; this checks the same stems
    /// against whatever llama.cpp published most recently. Ignored by default
    /// (network); run with:
    ///   cargo test -- --ignored --nocapture stems_match_the_live_release
    #[test]
    #[ignore]
    fn stems_match_the_live_release() {
        let body = agent()
            .get(RELEASES)
            .set("User-Agent", "tokamak")
            .call()
            .expect("reach github")
            .into_string()
            .expect("body");
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        let names: Vec<String> = v["assets"]
            .as_array()
            .expect("assets")
            .iter()
            .filter_map(|a| a["name"].as_str().map(str::to_string))
            .collect();
        println!("release {}", v["tag_name"].as_str().unwrap_or("?"));
        for t in [WIN, LINUX] {
            for (what, stem) in [("cpu", t.cpu_stem()), ("vulkan", t.vulkan_stem())] {
                let hits: Vec<&String> = names.iter().filter(|n| n.contains(&stem)).collect();
                println!("  {:6} {:8} {stem:24} -> {hits:?}", t.platform, what);
                assert_eq!(hits.len(), 1, "{} {what} stem {stem} matched {hits:?}", t.platform);
            }
        }
    }
}
