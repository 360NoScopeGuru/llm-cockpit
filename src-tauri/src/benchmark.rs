//! Measured benchmark — the moat.
//!
//! For each candidate config, actually launch llama-server, load the model, run
//! a fixed generation, and measure *real* prefill/decode tok/s (from the
//! completion response `timings`) plus *real* peak VRAM (sampling NVML during
//! the run). This upgrades the auto-config estimate from predicted to measured.
//!
//! Runs on a dedicated port with its own server instances, sequentially, so it
//! never collides with the user's running model.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};

use crate::llama::{LlamaManager, LlamaServerConfig};

/// Dedicated port for benchmark server instances (distinct from the user's 8137).
const BENCH_PORT: u16 = 8139;
/// Tokens to generate per config — enough for a stable decode-rate reading.
const N_PREDICT: u32 = 96;
/// A fixed prompt with enough length to give prefill measurable work.
const BENCH_PROMPT: &str = "The quick brown fox jumps over the lazy dog. \
Sphinx of black quartz, judge my vow. Pack my box with five dozen liquor jugs. \
How razorback jumping frogs can level six piqued gymnasts. Summarize the above.";

/// How long to wait for a config's server to become healthy before giving up.
const HEALTH_TIMEOUT_SECS: u64 = 90;

#[derive(Debug, Clone, Deserialize)]
pub struct BenchConfig {
    pub n_gpu_layers: u32,
    pub ctx_size: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub n_gpu_layers: u32,
    pub ctx_size: u32,
    pub loaded: bool,
    pub load_ms: u64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub peak_vram_bytes: u64,
    pub error: Option<String>,
}

impl BenchResult {
    fn failed(cfg: &BenchConfig, error: String) -> Self {
        BenchResult {
            n_gpu_layers: cfg.n_gpu_layers,
            ctx_size: cfg.ctx_size,
            loaded: false,
            load_ms: 0,
            prefill_tok_s: 0.0,
            decode_tok_s: 0.0,
            peak_vram_bytes: 0,
            error: Some(error),
        }
    }
}

/// Benchmark each config sequentially, invoking `on_progress` after each one
/// completes (so the UI can stream results). Returns all results.
pub fn run_benchmark<F: Fn(&BenchResult)>(
    model_path: &str,
    configs: &[BenchConfig],
    on_progress: F,
) -> Vec<BenchResult> {
    let mut results = Vec::new();
    for cfg in configs.iter().take(6) {
        let result = benchmark_one(model_path, cfg);
        on_progress(&result);
        results.push(result);
        // Let the OS fully release the port before the next config binds it.
        thread::sleep(Duration::from_millis(600));
    }
    results
}

fn benchmark_one(model_path: &str, cfg: &BenchConfig) -> BenchResult {
    let mgr = LlamaManager::new();
    let server_cfg = LlamaServerConfig {
        model_path: model_path.to_string(),
        n_gpu_layers: Some(cfg.n_gpu_layers),
        ctx_size: Some(cfg.ctx_size),
        port: BENCH_PORT,
        binary_path: None,
        flash_attn: false,
        extra_args: vec![],
    };

    let start = Instant::now();
    if let Err(e) = mgr.start(server_cfg) {
        return BenchResult::failed(cfg, e);
    }

    // Wait for health (model load can take a while for big models).
    let mut healthy = false;
    let deadline = Instant::now() + Duration::from_secs(HEALTH_TIMEOUT_SECS);
    while Instant::now() < deadline {
        let st = mgr.status();
        if st.health == "ok" {
            healthy = true;
            break;
        }
        if let Some(err) = st.error {
            let _ = mgr.stop();
            return BenchResult::failed(cfg, err);
        }
        thread::sleep(Duration::from_millis(500));
    }
    if !healthy {
        let _ = mgr.stop();
        return BenchResult::failed(cfg, "did not become healthy in time".into());
    }
    let load_ms = start.elapsed().as_millis() as u64;

    // Sample peak VRAM in the background while we generate.
    let stop_flag = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let sampler = {
        let stop_flag = stop_flag.clone();
        let peak = peak.clone();
        thread::spawn(move || {
            let nvml = Nvml::init().ok();
            while !stop_flag.load(Ordering::Relaxed) {
                if let Some(used) = nvml
                    .as_ref()
                    .and_then(|n| n.device_by_index(0).ok())
                    .and_then(|d| d.memory_info().ok())
                    .map(|m| m.used)
                {
                    peak.fetch_max(used, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(100));
            }
        })
    };

    let body = post_completion(BENCH_PORT);

    stop_flag.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let peak_vram_bytes = peak.load(Ordering::Relaxed);

    let (prefill_tok_s, decode_tok_s) =
        body.as_deref().and_then(parse_timings).unwrap_or((0.0, 0.0));

    let _ = mgr.stop();

    BenchResult {
        n_gpu_layers: cfg.n_gpu_layers,
        ctx_size: cfg.ctx_size,
        loaded: true,
        load_ms,
        prefill_tok_s,
        decode_tok_s,
        peak_vram_bytes,
        error: if body.is_none() {
            Some("generation request failed".into())
        } else {
            None
        },
    }
}

fn post_completion(port: u16) -> Option<String> {
    let url = format!("http://127.0.0.1:{port}/completion");
    // ignore_eos forces exactly N_PREDICT tokens so the decode rate is measured
    // over a real, fixed workload instead of a near-instant early stop.
    let body = format!(
        r#"{{"prompt":"{BENCH_PROMPT}","n_predict":{N_PREDICT},"ignore_eos":true,"stream":false}}"#
    );
    // Generous: a config spilled to CPU can take minutes for 96 tokens, and a
    // slow measured number beats a false "generation failed".
    ureq::post(&url)
        .timeout(Duration::from_secs(180))
        .set("Content-Type", "application/json")
        .send_string(&body)
        .ok()?
        .into_string()
        .ok()
}

/// Extract (prefill_tok_s, decode_tok_s) from a llama-server completion response.
/// Computed from token counts / elapsed ms directly — llama.cpp's own
/// `*_per_second` fields can overflow to huge/inf values on near-zero timings.
fn parse_timings(body: &str) -> Option<(f64, f64)> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let t = v.get("timings")?;
    let f = |k: &str| t.get(k).and_then(|x| x.as_f64());
    let rate = |n: Option<f64>, ms: Option<f64>| match (n, ms) {
        (Some(n), Some(ms)) if ms > 0.0 => n * 1000.0 / ms,
        _ => 0.0,
    };
    let prefill = rate(f("prompt_n"), f("prompt_ms"));
    let decode = rate(f("predicted_n"), f("predicted_ms"));
    Some((prefill, decode))
}

/// One row of a cross-model benchmark report (frontend supplies the rows it
/// accumulated from suite runs).
#[derive(Debug, Clone, Deserialize)]
pub struct ReportRow {
    pub model: String,
    pub quant: Option<String>,
    pub n_gpu_layers: u32,
    pub ctx_size: u32,
    pub load_ms: u64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub peak_vram_bytes: u64,
}

/// Write a Markdown benchmark report to Documents\tokamak and return its
/// path. Rows are written in the order given; a ranking column is derived from
/// decode speed.
pub fn export_report(gpu_name: &str, rows: &[ReportRow]) -> Result<String, String> {
    if rows.is_empty() {
        return Err("no benchmark rows to export".into());
    }
    let dir = dirs::document_dir()
        .ok_or("no Documents dir on this platform")?
        .join("tokamak");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("bench-report-{now}.md"));

    let best = rows
        .iter()
        .map(|r| r.decode_tok_s)
        .fold(0.0_f64, f64::max);

    let mut md = String::new();
    md.push_str(&format!(
        "# Tokamak benchmark report\n\nGPU: **{gpu_name}**  \nConfigs: auto-recommended per model (max offload + context that fit).  \nAll numbers measured on this machine, not estimates.\n\n"
    ));
    md.push_str(
        "| Model | Quant | GPU layers | Ctx | Load | Prefill tok/s | Decode tok/s | Peak VRAM | vs best |\n|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        let rel = if best > 0.0 {
            format!("{:.0}%", r.decode_tok_s / best * 100.0)
        } else {
            "—".into()
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.1}s | {:.0} | **{:.1}** | {:.2} GB | {} |\n",
            r.model,
            r.quant.as_deref().unwrap_or("?"),
            r.n_gpu_layers,
            r.ctx_size,
            r.load_ms as f64 / 1000.0,
            r.prefill_tok_s,
            r.decode_tok_s,
            r.peak_vram_bytes as f64 / 1e9,
            rel,
        ));
    }
    md.push('\n');

    std::fs::write(&path, md).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_completion_timings() {
        // 40 prompt tokens in 100 ms => 400 tok/s; 96 predicted in 500 ms => 192 tok/s.
        let body = r#"{"content":"...","timings":{"prompt_n":40,"prompt_ms":100.0,"predicted_n":96,"predicted_ms":500.0}}"#;
        let (prefill, decode) = parse_timings(body).unwrap();
        assert_eq!(prefill, 400.0);
        assert_eq!(decode, 192.0);
    }

    #[test]
    fn timings_zero_ms_is_safe() {
        let body = r#"{"timings":{"prompt_n":5,"prompt_ms":0.0,"predicted_n":0,"predicted_ms":0.0}}"#;
        assert_eq!(parse_timings(body), Some((0.0, 0.0)));
    }

    #[test]
    fn timings_missing_is_none() {
        assert!(parse_timings(r#"{"content":"x"}"#).is_none());
    }

    /// Real benchmark of the 4B model: full GPU offload vs half offload.
    /// Ignored by default; run with:
    ///   cargo test -- --ignored --nocapture bench_real_model
    #[test]
    #[ignore]
    fn bench_real_model() {
        let Some(home) = dirs::home_dir() else {
            return;
        };
        let model = home.join(".lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf");
        if !model.is_file() {
            eprintln!("model not present, skipping");
            return;
        }
        let configs = vec![
            BenchConfig { n_gpu_layers: 999, ctx_size: 4096 }, // full offload
            BenchConfig { n_gpu_layers: 12, ctx_size: 4096 },  // partial (slower)
        ];
        let results = run_benchmark(&model.to_string_lossy(), &configs, |r| {
            println!(
                "  config ngl={} ctx={} -> loaded={} load={}ms prefill={:.1} decode={:.1} tok/s peak_vram={:.2}GB {}",
                r.n_gpu_layers,
                r.ctx_size,
                r.loaded,
                r.load_ms,
                r.prefill_tok_s,
                r.decode_tok_s,
                r.peak_vram_bytes as f64 / 1e9,
                r.error.as_deref().unwrap_or(""),
            );
        });
        assert_eq!(results.len(), 2);
        assert!(results[0].loaded, "full offload should load");
        assert!(results[0].decode_tok_s > 0.0, "should measure a decode rate");
    }
}
