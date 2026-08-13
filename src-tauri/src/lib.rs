/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

mod benchmark;
mod chat;
mod downloads;
mod estimator;
mod gguf;
mod history;
mod llama;
mod ollama;
mod runtime;
mod scanner;
mod settings;
mod speculative;
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

/// Rank the local library as draft models for `target_path`, for speculative
/// decoding. Answers up front what `llama-server` would otherwise only tell you
/// by refusing to start after a long model load.
#[tauri::command]
fn draft_candidates(
    target_path: String,
    extra_dirs: Vec<String>,
) -> Result<Vec<speculative::DraftCandidate>, String> {
    let path = std::path::Path::new(&target_path);
    let target = gguf::read_gguf_metadata(path).map_err(|e| e.to_string())?;
    let target_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let mut dirs = settings::load().extra_model_dirs;
    dirs.extend(extra_dirs);
    let models = scanner::scan_models(&dirs);

    Ok(speculative::rank_drafts(
        &target_path,
        &target,
        target_size,
        &models,
    ))
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

/// Persist the KV cache element type ("f16" | "q8_0" | "q4_0").
#[tauri::command]
fn set_kv_cache_type(kind: String) -> Result<settings::Settings, String> {
    settings::set_kv_cache_type(kind)
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

/// Is a managed llama.cpp runtime installed?
#[tauri::command]
fn runtime_status() -> runtime::RuntimeStatus {
    runtime::status()
}

/// Installable llama.cpp builds for this machine, with real download sizes.
#[tauri::command]
fn runtime_options(telemetry: State<'_, TelemetryState>) -> Result<Vec<runtime::RuntimeBuild>, String> {
    let has_nvidia = !telemetry.snapshot().gpus.is_empty();
    runtime::options(has_nvidia)
}

/// Download and unpack a llama.cpp build.
#[tauri::command]
fn runtime_install(window: tauri::Window, build: runtime::RuntimeBuild) {
    runtime::install(window, build);
}

/// Search Hugging Face for GGUF repos.
#[tauri::command]
fn hf_search(query: String) -> Result<Vec<downloads::HfModel>, String> {
    downloads::search(&query, settings::load().hf_token.as_deref())
}

/// List the GGUF files (with sizes) in a Hugging Face repo.
#[tauri::command]
fn hf_files(repo: String) -> Result<Vec<downloads::HfFile>, String> {
    downloads::list_files(&repo, settings::load().hf_token.as_deref())
}

/// Which download sources are usable on this machine.
#[tauri::command]
fn download_sources() -> serde_json::Value {
    serde_json::json!({
        "huggingface": true,
        "url": true,
        "ollama": downloads::ollama_available(),
    })
}

/// Where downloaded models land.
#[tauri::command]
fn downloads_dir() -> Result<String, String> {
    downloads::models_dir().map(|p| p.to_string_lossy().into_owned())
}

/// Start a download. `source` is "huggingface" | "url" | "ollama".
///
/// Wide by design: Tauri commands take their payload as flat parameters, and
/// which of `repo`/`file`/`url`/`model` are present depends on `source`. Two of
/// the eight (`window`, `state`) are injected by Tauri, not passed by the
/// caller — bundling the rest into a struct would only move the same fields
/// behind an extra layer on both sides of the bridge.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
fn download_start(
    window: tauri::Window,
    state: State<'_, downloads::DownloadState>,
    id: u64,
    source: String,
    repo: Option<String>,
    file: Option<String>,
    url: Option<String>,
    model: Option<String>,
) -> Result<(), String> {
    match source.as_str() {
        "huggingface" => downloads::start_hf(
            window,
            &state,
            id,
            repo.ok_or("missing repo")?,
            file.ok_or("missing file")?,
            settings::load().hf_token,
        ),
        "url" => downloads::start_url(window, &state, id, url.ok_or("missing url")?),
        "ollama" => downloads::start_ollama(window, &state, id, model.ok_or("missing model")?),
        other => Err(format!("unknown download source: {other}")),
    }
}

/// Cancel an in-flight download (partial data is kept so it can resume).
#[tauri::command]
fn download_cancel(state: State<'_, downloads::DownloadState>, id: u64) {
    state.cancel(id);
}

/// Persist a Hugging Face token for gated repos (None clears it).
#[tauri::command]
fn set_hf_token(token: Option<String>) -> Result<settings::Settings, String> {
    settings::set_hf_token(token)
}

/// Ask the running model to name a session. Runs outside the streaming
/// single-flight path so it cannot cancel a generation in progress.
#[tauri::command]
fn chat_title(llama: State<'_, LlamaManager>, transcript: String) -> Result<String, String> {
    let base_url = llama
        .base_url()
        .ok_or("no model is running — launch one first")?;
    chat::generate_title(&base_url, &transcript)
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
    // The KV cache type is a launch setting, and it changes how much context
    // fits — so the estimate has to be made under the same setting the server
    // will actually run with, or the ladder promises context that won't exist.
    let kv = estimator::KvType::parse(settings::load().kv_cache_type.as_deref());
    let mut est = estimator::estimate(&shape, gpu_total, gpu_free, kv, notes);
    est.quant_advice = estimator::quant_advice(
        &shape,
        md.quant_label.as_deref(),
        md.parameter_count,
        gpu_total,
        kv,
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
        .manage(downloads::DownloadState::default())
        .invoke_handler(tauri::generate_handler![
            scan_models,
            scan_roots,
            draft_candidates,
            add_model_dir,
            remove_model_dir,
            get_settings,
            set_preferred_binary,
            set_ui_scale,
            set_kv_cache_type,
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
            chat_cancel,
            chat_title,
            runtime_status,
            runtime_options,
            runtime_install,
            hf_search,
            hf_files,
            download_sources,
            downloads_dir,
            download_start,
            download_cancel,
            set_hf_token
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
