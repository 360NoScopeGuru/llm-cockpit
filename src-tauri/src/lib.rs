mod estimator;
mod gguf;
mod llama;
mod scanner;
mod telemetry;

use std::path::Path;

use tauri::State;

use estimator::VramEstimate;
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

/// Estimate the optimal GPU-offload + context config for a model on this GPU.
#[tauri::command]
fn estimate_config(
    telemetry: State<'_, TelemetryState>,
    model_path: String,
) -> Result<VramEstimate, String> {
    let path = Path::new(&model_path);
    let md = gguf::read_gguf_metadata(path).map_err(|e| e.to_string())?;
    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| e.to_string())?;

    let snap = telemetry.snapshot();
    let (gpu_total, gpu_free) = snap
        .gpus
        .first()
        .map(|g| {
            (
                g.vram_total_bytes,
                g.vram_total_bytes.saturating_sub(g.vram_used_bytes),
            )
        })
        .unwrap_or((0, 0));
    if gpu_total == 0 {
        return Err("no GPU detected to estimate against".into());
    }

    let mut notes = Vec::new();
    let shape = estimator::shape_from_metadata(&md, file_size, &mut notes)
        .ok_or_else(|| "insufficient model metadata to estimate".to_string())?;
    Ok(estimator::estimate(&shape, gpu_total, gpu_free, notes))
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
            llama_status,
            estimate_config
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
