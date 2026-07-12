mod gguf;
mod llama;
mod scanner;
mod telemetry;

use tauri::State;

use llama::{LlamaBinary, LlamaManager, LlamaServerConfig, ServerStatus};
use scanner::{ModelEntry, ScanRoot};
use telemetry::{TelemetrySnapshot, TelemetryState};

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

/// One live snapshot of GPU + system telemetry. Polled by the frontend.
#[tauri::command]
fn gpu_telemetry(state: State<'_, TelemetryState>) -> TelemetrySnapshot {
    state.snapshot()
}

/// List available llama-server binaries, best-ranked first.
#[tauri::command]
fn llama_binaries() -> Vec<LlamaBinary> {
    llama::resolve_binaries()
}

/// Launch a model with llama-server (replaces any running instance).
#[tauri::command]
fn llama_start(
    state: State<'_, LlamaManager>,
    config: LlamaServerConfig,
) -> Result<ServerStatus, String> {
    state.start(config)
}

/// Stop the running server (if any).
#[tauri::command]
fn llama_stop(state: State<'_, LlamaManager>) -> Result<(), String> {
    state.stop()
}

/// Current server status + live health probe.
#[tauri::command]
fn llama_status(state: State<'_, LlamaManager>) -> ServerStatus {
    state.status()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(TelemetryState::new())
        .manage(LlamaManager::new())
        .invoke_handler(tauri::generate_handler![
            scan_models,
            scan_roots,
            gpu_telemetry,
            llama_binaries,
            llama_start,
            llama_stop,
            llama_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
