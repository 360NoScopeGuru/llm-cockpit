mod benchmark;
mod chat;
mod estimator;
mod gguf;
mod history;
mod llama;
mod scanner;
mod settings;
mod telemetry;
mod tools;

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

/// Current persisted settings.
#[tauri::command]
fn get_settings() -> settings::Settings {
    settings::load()
}

/// Persist the preferred llama-server binary (None = auto-select best).
#[tauri::command]
fn set_preferred_binary(path: Option<String>) -> Result<settings::Settings, String> {
    settings::set_preferred_binary(path)
}

/// List saved chat/code sessions, newest first.
#[tauri::command]
fn history_list() -> Result<Vec<history::SessionMeta>, String> {
    history::list()
}

/// Load one saved session in full.
#[tauri::command]
fn history_get(id: String) -> Result<history::Session, String> {
    history::get(&id)
}

/// Upsert a session (the frontend saves after every completed turn).
#[tauri::command]
fn history_save(session: history::Session) -> Result<(), String> {
    history::save(&session)
}

/// Delete a saved session.
#[tauri::command]
fn history_delete(id: String) -> Result<(), String> {
    history::delete(&id)
}

/// Persist the user's UI zoom factor.
#[tauri::command]
fn set_ui_scale(scale: f64) -> Result<settings::Settings, String> {
    settings::set_ui_scale(scale)
}

/// Persist the agent workspace folder (None disables the agent).
#[tauri::command]
fn set_agent_workspace(dir: Option<String>) -> Result<settings::Settings, String> {
    settings::set_agent_workspace(dir)
}

/// Agent tool: list a directory inside the workspace sandbox.
#[tauri::command]
fn agent_list_dir(root: String, path: String) -> Result<Vec<tools::DirEntryInfo>, String> {
    tools::list_dir(&root, &path)
}

/// Agent tool: read a text file inside the workspace sandbox.
#[tauri::command]
fn agent_read_file(root: String, path: String) -> Result<tools::ReadFileResult, String> {
    tools::read_file(&root, &path)
}

/// Agent tool: write a file inside the workspace sandbox. The frontend gates
/// this behind an explicit user approval click.
#[tauri::command]
fn agent_write_file(root: String, path: String, content: String) -> Result<String, String> {
    tools::write_file(&root, &path, &content)
}

/// Agent tool: run a PowerShell command in the workspace. The frontend gates
/// this behind an explicit user approval click.
#[tauri::command]
fn agent_run_command(root: String, command: String) -> Result<tools::RunCommandResult, String> {
    tools::run_command(&root, &command)
}

/// Start a streaming chat generation against the running server.
#[tauri::command]
fn chat_send(
    window: tauri::Window,
    llama: State<'_, LlamaManager>,
    chat_state: State<'_, chat::ChatState>,
    id: u64,
    messages: Vec<chat::ChatMessage>,
    params: chat::ChatParams,
) -> Result<(), String> {
    let base_url = llama
        .base_url()
        .ok_or("no model is running — launch one first")?;
    chat::start_stream(window, &chat_state, base_url, id, messages, params);
    Ok(())
}

/// Cancel the in-flight chat generation, if any.
#[tauri::command]
fn chat_cancel(chat_state: State<'_, chat::ChatState>) {
    chat_state.cancel();
}

/// One live snapshot of GPU + system telemetry. Polled by the frontend.
#[tauri::command]
fn gpu_telemetry(state: State<'_, TelemetryState>) -> TelemetrySnapshot {
    state.snapshot()
}

/// The window's REAL client size in physical pixels, straight from Win32.
/// tao/WebView2 can disagree with the OS about DPI (reporting the intended
/// logical size as physical), which makes every in-page metric self-consistent
/// while the actual window clips the overflow — so the DPI corrector must
/// measure against this ground truth instead.
#[cfg(windows)]
#[tauri::command]
fn true_client_size(window: tauri::Window) -> Option<(i32, i32)> {
    #[repr(C)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }
    #[link(name = "user32")]
    extern "system" {
        fn GetClientRect(hwnd: *mut core::ffi::c_void, rect: *mut Rect) -> i32;
    }
    let hwnd = window.hwnd().ok()?;
    let mut r = Rect {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };
    unsafe {
        if GetClientRect(hwnd.0, &mut r) == 0 {
            return None;
        }
    }
    Some((r.right - r.left, r.bottom - r.top))
}

#[cfg(not(windows))]
#[tauri::command]
fn true_client_size(_window: tauri::Window) -> Option<(i32, i32)> {
    None
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
    let mut est = estimator::estimate(&shape, gpu_total, gpu_free, notes);
    est.quant_advice = estimator::quant_advice(
        &shape,
        md.quant_label.as_deref(),
        md.parameter_count,
        gpu_total,
    );
    Ok(est)
}

/// Export accumulated suite results as a Markdown report in Documents.
#[tauri::command]
fn export_bench_report(
    telemetry: State<'_, TelemetryState>,
    rows: Vec<benchmark::ReportRow>,
) -> Result<String, String> {
    let gpu_name = telemetry
        .snapshot()
        .gpus
        .first()
        .map(|g| g.name.clone())
        .unwrap_or_else(|| "unknown GPU".into());
    benchmark::export_report(&gpu_name, &rows)
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
        .manage(chat::ChatState::default())
        .invoke_handler(tauri::generate_handler![
            scan_models,
            scan_roots,
            add_model_dir,
            remove_model_dir,
            get_settings,
            set_preferred_binary,
            set_ui_scale,
            set_agent_workspace,
            agent_list_dir,
            agent_read_file,
            agent_write_file,
            agent_run_command,
            history_list,
            history_get,
            history_save,
            history_delete,
            gpu_telemetry,
            true_client_size,
            llama_binaries,
            llama_start,
            llama_stop,
            llama_status,
            inference_metrics,
            estimate_config,
            benchmark_model,
            export_bench_report,
            chat_send,
            chat_cancel
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
