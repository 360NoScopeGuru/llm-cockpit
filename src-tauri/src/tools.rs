//! Agent tools: filesystem + command access for the console's agent mode,
//! sandboxed to a single user-chosen workspace folder.
//!
//! Every tool takes paths relative to the workspace root. Escapes are blocked
//! lexically (`..` components are resolved before joining, absolute paths must
//! already sit under the root), so the model can never read or write outside
//! the folder the user granted. Destructive tools (write_file, run_command)
//! are additionally gated behind an explicit per-call approval click in the
//! frontend — the backend just does what it is told, the UI is the consent
//! layer.

use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Serialize;

const MAX_READ_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_BYTES: usize = 96 * 1024;
const MAX_DIR_ENTRIES: usize = 500;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Serialize)]
pub struct DirEntryInfo {
    pub name: String,
    pub is_dir: bool,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct ReadFileResult {
    pub content: String,
    pub size_bytes: u64,
    pub truncated: bool,
}

#[derive(Debug, Serialize)]
pub struct RunCommandResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
}

/// Resolve `rel` against the workspace root, refusing anything that would
/// land outside it. `..` is resolved lexically so a nonexistent target (a
/// file about to be created) can still be validated.
fn resolve(root: &str, rel: &str) -> Result<PathBuf, String> {
    let root = fs::canonicalize(root).map_err(|e| format!("workspace not accessible: {e}"))?;
    let raw = Path::new(rel);
    let joined = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        root.join(raw)
    };
    // Lexical normalization: strip `.`, resolve `..` without touching the fs.
    let mut norm = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !norm.pop() {
                    return Err("path escapes the workspace".into());
                }
            }
            other => norm.push(other),
        }
    }
    // Compare canonically where possible (the path may not exist yet, so
    // canonicalize the nearest existing ancestor).
    let mut probe = norm.clone();
    let mut tail = PathBuf::new();
    let canon = loop {
        match fs::canonicalize(&probe) {
            Ok(c) => {
                break if tail.as_os_str().is_empty() {
                    c
                } else {
                    c.join(&tail)
                }
            }
            Err(_) => {
                let Some(name) = probe.file_name() else {
                    return Err("path escapes the workspace".into());
                };
                tail = if tail.as_os_str().is_empty() {
                    PathBuf::from(name)
                } else {
                    Path::new(name).join(&tail)
                };
                if !probe.pop() {
                    return Err("path escapes the workspace".into());
                }
            }
        }
    };
    if !canon.starts_with(&root) {
        return Err(format!(
            "path is outside the workspace: {}",
            norm.display()
        ));
    }
    Ok(canon)
}

pub fn list_dir(root: &str, path: &str) -> Result<Vec<DirEntryInfo>, String> {
    let dir = resolve(root, path)?;
    let rd = fs::read_dir(&dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let md = entry.metadata().ok();
        out.push(DirEntryInfo {
            name: entry.file_name().to_string_lossy().into_owned(),
            is_dir: md.as_ref().map(|m| m.is_dir()).unwrap_or(false),
            size_bytes: md.map(|m| m.len()).unwrap_or(0),
        });
        if out.len() >= MAX_DIR_ENTRIES {
            break;
        }
    }
    out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
    Ok(out)
}

pub fn read_file(root: &str, path: &str) -> Result<ReadFileResult, String> {
    let file = resolve(root, path)?;
    let md = fs::metadata(&file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
    if md.is_dir() {
        return Err("that path is a directory — use list_dir".into());
    }
    let mut f = fs::File::open(&file).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; MAX_READ_BYTES.min(md.len() as usize + 1)];
    let mut read = 0;
    while read < buf.len() {
        match f.read(&mut buf[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) => return Err(e.to_string()),
        }
    }
    buf.truncate(read);
    if buf.iter().take(4096).any(|&b| b == 0) {
        return Err(format!(
            "binary file ({} bytes) — refusing to read as text",
            md.len()
        ));
    }
    let truncated = (md.len() as usize) > read;
    Ok(ReadFileResult {
        content: String::from_utf8_lossy(&buf).into_owned(),
        size_bytes: md.len(),
        truncated,
    })
}

pub fn write_file(root: &str, path: &str, content: &str) -> Result<String, String> {
    let file = resolve(root, path)?;
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(&file, content).map_err(|e| e.to_string())?;
    Ok(format!("wrote {} bytes to {}", content.len(), file.display()))
}

pub fn run_command(root: &str, command: &str) -> Result<RunCommandResult, String> {
    let cwd = fs::canonicalize(root).map_err(|e| format!("workspace not accessible: {e}"))?;
    let mut cmd = Command::new("powershell");
    cmd.args(["-NoProfile", "-NonInteractive", "-Command", command])
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| format!("failed to spawn: {e}"))?;

    // Drain pipes on threads so a chatty command can't deadlock on a full pipe.
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let out_handle = std::thread::spawn(move || {
        let mut s = Vec::new();
        if let Some(p) = stdout_pipe.as_mut() {
            let _ = p.read_to_end(&mut s);
        }
        s
    });
    let err_handle = std::thread::spawn(move || {
        let mut s = Vec::new();
        if let Some(p) = stderr_pipe.as_mut() {
            let _ = p.read_to_end(&mut s);
        }
        s
    });

    let started = Instant::now();
    let mut timed_out = false;
    let exit_code = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.code(),
            Ok(None) => {
                if started.elapsed() > COMMAND_TIMEOUT {
                    let _ = child.kill();
                    timed_out = true;
                    break None;
                }
                std::thread::sleep(Duration::from_millis(80));
            }
            Err(e) => return Err(e.to_string()),
        }
    };

    let clip = |v: Vec<u8>| {
        let mut s = String::from_utf8_lossy(&v).into_owned();
        if s.len() > MAX_OUTPUT_BYTES {
            s.truncate(MAX_OUTPUT_BYTES);
            s.push_str("\n…[output truncated]");
        }
        s
    };
    Ok(RunCommandResult {
        stdout: clip(out_handle.join().unwrap_or_default()),
        stderr: clip(err_handle.join().unwrap_or_default()),
        exit_code,
        timed_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sandboxing is the security boundary — test it without extra dev-deps by
    // using a folder created under the OS temp dir.
    fn mk_root() -> (PathBuf, String) {
        let root = std::env::temp_dir().join(format!("tokamak-tools-test-{}", std::process::id()));
        let _ = fs::create_dir_all(root.join("sub"));
        fs::write(root.join("a.txt"), "hello").unwrap();
        fs::write(root.join("sub").join("b.txt"), "world").unwrap();
        let s = root.to_string_lossy().into_owned();
        (root, s)
    }

    #[test]
    fn sandbox_blocks_escapes() {
        let (_root, r) = mk_root();
        assert!(read_file(&r, "..\\outside.txt").is_err());
        assert!(read_file(&r, "sub\\..\\..\\outside.txt").is_err());
        assert!(read_file(&r, "C:\\Windows\\win.ini").is_err());
        assert!(write_file(&r, "..\\evil.txt", "x").is_err());
        // Legal traversal that stays inside is fine.
        assert!(read_file(&r, "sub\\..\\a.txt").is_ok());
    }

    #[test]
    fn read_list_write_roundtrip() {
        let (_root, r) = mk_root();
        let entries = list_dir(&r, ".").unwrap();
        assert!(entries.iter().any(|e| e.name == "a.txt"));
        assert!(entries.iter().any(|e| e.name == "sub" && e.is_dir));
        assert_eq!(read_file(&r, "a.txt").unwrap().content, "hello");
        write_file(&r, "new\\c.txt", "made").unwrap();
        assert_eq!(read_file(&r, "new\\c.txt").unwrap().content, "made");
    }

    #[test]
    #[ignore] // spawns powershell; run with --ignored
    fn run_command_captures_output() {
        let (_root, r) = mk_root();
        let res = run_command(&r, "Write-Output hi; Write-Error nope").unwrap();
        assert!(res.stdout.contains("hi"));
        assert!(res.stderr.contains("nope"));
        assert!(!res.timed_out);
    }
}
