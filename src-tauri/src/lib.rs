mod benchmark;
mod estimator;
mod gguf;
mod llama;
mod scanner;
mod settings;
mod telemetry;

use std::path::Path;

use tauri::State;

use estimator::VramEstimate;
use llama::{InferenceMetrics, LlamaBinary, LlamaManager, LlamaServerConfig, ServerStatus};
use scanner::{ModelEntry, ScanRoot};
use telemetry::{TelemetrySnapshot, TelemetryState};

/// Scan default caches + persisted user folders + any extra ad-hoc folders.
#[tauri::command]
fn scan_models(extra_dirs: Vec<String>) -> Vec<ModelEntry> {
    let mut dirs = settings::load().extra_model_dirs;
    dirs.extend(extra_dirs);
    scanner::scan_models(&dirs)
}

/// Report all scan roots — defaults plus persisted user folders — and whether
/// each currently exists (for the roots UI).
#[tauri::command]
fn scan_roots() -> Vec<ScanRoot> {
    let mut roots = scanner::default_roots_info();
    for dir in settings::load().extra_model_dirs {
        roots.push(ScanRoot {
            exists: Path::new(&dir).is_dir(),
            path: dir,
            source: "folder".into(),
        });
    }
    roots
}

/// Persist a new model folder to scan. Returns the updated settings.
#[tauri::command]
fn add_model_dir(dir: String) -> Result<settings::Settings, String> {
    settings::add_model_dir(&dir)
}

/// Remove a persisted model folder. Returns the updated settings.
#[tauri::command]
fn remove_model_dir(dir: String) -> Result<settings::Settings, String> {
    settings::remove_model_dir(&dir)
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

/// Inference-side metrics (tok/s, KV-cache usage) from the running server.
#[tauri::command]
fn inference_metrics(state: State<'_, LlamaManager>) -> Option<InferenceMetrics> {
    state.metrics()
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

/// Measured benchmark: launch each config for real and measure tok/s + peak
/// VRAM. Stops any running model first (to free VRAM) and emits a
/// `benchmark-progress` event as each config completes.
#[tauri::command]
fn benchmark_model(
    window: tauri::Window,
    llama: State<'_, LlamaManager>,
    model_path: String,
    configs: Vec<benchmark::BenchConfig>,
) -> Vec<benchmark::BenchResult> {
    use tauri::Emitter;
    let _ = llama.stop();
    benchmark::run_benchmark(&model_path, &configs, |r| {
        let _ = window.emit("benchmark-progress", r);
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(TelemetryState::new())
        .manage(LlamaManager::new())
        .invoke_handler(tauri::generate_handler![
            scan_models,
            scan_roots,
            add_model_dir,
            remove_model_dir,
            gpu_telemetry,
            llama_binaries,
            llama_start,
            llama_stop,
            llama_status,
            inference_metrics,
            estimate_config,
            benchmark_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
