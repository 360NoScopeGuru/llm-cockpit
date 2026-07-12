mod gguf;
mod scanner;

use scanner::{ModelEntry, ScanRoot};

/// Scan default caches + any extra folders for GGUF models the user already has.
#[tauri::command]
fn scan_models(extra_dirs: Vec<String>) -> Vec<ModelEntry> {
    scanner::scan_models(&extra_dirs)
}

/// Report the default scan roots and whether each exists (for the settings UI).
#[tauri::command]
fn scan_roots() -> Vec<ScanRoot> {
    scanner::default_roots_info()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![scan_models, scan_roots])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
