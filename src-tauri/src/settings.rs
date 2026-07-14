//! Persisted app settings.
//!
//! A small JSON file in the OS config dir (e.g. `%APPDATA%\llm-cockpit` on
//! Windows) so user choices — currently the extra model folders to scan —
//! survive restarts and stay out of the frontend's webview storage.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub extra_model_dirs: Vec<String>,
    /// Full path of the llama-server binary the user picked; falls back to the
    /// best-ranked discovered binary when unset or missing on disk.
    #[serde(default)]
    pub preferred_binary: Option<String>,
}

fn settings_path() -> Option<PathBuf> {
    Some(dirs::config_dir()?.join("llm-cockpit").join("settings.json"))
}

/// Load settings; any missing/corrupt file yields defaults.
pub fn load() -> Settings {
    settings_path()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save(settings: &Settings) -> Result<(), String> {
    let path = settings_path().ok_or("no config dir on this platform")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    fs::write(&path, json).map_err(|e| e.to_string())
}

/// Add a model folder (validated to exist), de-duplicated case-insensitively
/// (Windows paths). Returns the updated settings.
pub fn add_model_dir(dir: &str) -> Result<Settings, String> {
    let path = PathBuf::from(dir);
    if !path.is_dir() {
        return Err(format!("not a directory: {dir}"));
    }
    let mut s = load();
    let exists = s
        .extra_model_dirs
        .iter()
        .any(|d| d.eq_ignore_ascii_case(dir));
    if !exists {
        s.extra_model_dirs.push(dir.to_string());
        save(&s)?;
    }
    Ok(s)
}

/// Remove a model folder from the persisted list. Returns the updated settings.
pub fn remove_model_dir(dir: &str) -> Result<Settings, String> {
    let mut s = load();
    s.extra_model_dirs.retain(|d| !d.eq_ignore_ascii_case(dir));
    save(&s)?;
    Ok(s)
}

/// Persist the preferred llama-server binary (None restores auto-select).
pub fn set_preferred_binary(path: Option<String>) -> Result<Settings, String> {
    if let Some(p) = &path {
        if !PathBuf::from(p).is_file() {
            return Err(format!("binary not found: {p}"));
        }
    }
    let mut s = load();
    s.preferred_binary = path;
    save(&s)?;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trips the real config file (add → present → remove → absent),
    /// restoring prior state by construction. Ignored by default since it
    /// touches the machine's config dir; run with:
    ///   cargo test -- --ignored --nocapture settings_roundtrip
    #[test]
    #[ignore]
    fn settings_roundtrip() {
        let dir = std::env::temp_dir()
            .to_string_lossy()
            .trim_end_matches(['\\', '/'])
            .to_string();

        let s = add_model_dir(&dir).expect("add should succeed");
        assert!(
            s.extra_model_dirs.iter().any(|d| d.eq_ignore_ascii_case(&dir)),
            "dir should be persisted after add"
        );
        // Adding again must not duplicate.
        let s2 = add_model_dir(&dir).expect("re-add should succeed");
        assert_eq!(
            s2.extra_model_dirs.len(),
            s.extra_model_dirs.len(),
            "re-add must be a no-op"
        );

        let s3 = remove_model_dir(&dir).expect("remove should succeed");
        assert!(
            !s3.extra_model_dirs.iter().any(|d| d.eq_ignore_ascii_case(&dir)),
            "dir should be gone after remove"
        );

        assert!(
            add_model_dir("Z:\\definitely\\not\\a\\real\\dir").is_err(),
            "nonexistent dir must be rejected"
        );
        println!("settings file round-trip OK");
    }
}
