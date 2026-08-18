/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

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

/// GPU memory is reported in GiB everywhere in this app, matching what the
/// driver and the vendor mean by "GB" for VRAM.
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

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

/// `Default` is for construction sites only — `n_gpu_layers` and `ctx_size`
/// have no `serde(default)`, so a config arriving from the frontend still has
/// to state them.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct BenchConfig {
    pub n_gpu_layers: u32,
    pub ctx_size: u32,
    /// Draft model for speculative decoding. When set, this config is run
    /// *twice* — once without the draft and once with — because the only
    /// honest way to report a speedup is to measure both ends on the same
    /// machine, back to back.
    #[serde(default)]
    pub draft_model_path: Option<String>,
    /// Tokens to draft per step. `None` leaves llama.cpp's default of 3.
    #[serde(default)]
    pub draft_n_max: Option<u32>,
}

/// What speculative decoding actually bought, measured rather than predicted.
#[derive(Debug, Clone, Serialize)]
pub struct SpecResult {
    /// Tokens the draft model proposed.
    pub draft_n: u64,
    /// How many of those the target accepted.
    pub draft_n_accepted: u64,
    /// `draft_n_accepted / draft_n`. The number that decides whether a draft
    /// is worth its VRAM: rejected tokens are wasted work on both models.
    pub accept_rate: f64,
    /// Decode speed for the identical config with no draft model.
    pub baseline_decode_tok_s: f64,
    /// Decode speed with the draft.
    pub decode_tok_s: f64,
    /// `decode / baseline`. Below 1.0 means the draft made generation slower,
    /// which is a real and common outcome — it is reported, not hidden.
    pub speedup: f64,
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
    /// Present only when the config named a draft model and both halves of
    /// the A/B ran.
    pub speculative: Option<SpecResult>,
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
            speculative: None,
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
        let mut result = benchmark_one(model_path, cfg, None);

        // A config that names a draft is really two measurements. The run
        // above is the baseline; run it again with the draft attached and
        // report the pair. Doing both here, back to back, is what makes the
        // speedup attributable — comparing against a number measured in some
        // earlier session would fold in whatever else the GPU was doing.
        if let (true, Some(draft)) = (result.loaded, cfg.draft_model_path.as_deref()) {
            thread::sleep(Duration::from_millis(600));
            let with_draft = benchmark_one(model_path, cfg, Some(draft));
            result = merge_speculative(result, with_draft);
        }

        on_progress(&result);
        results.push(result);
        // Let the OS fully release the port before the next config binds it.
        thread::sleep(Duration::from_millis(600));
    }
    results
}

/// Fold a with-draft run into its baseline, producing one row that carries the
/// comparison.
///
/// The reported prefill/decode/VRAM become the *speculative* ones, since that
/// is the configuration being proposed; the baseline survives inside
/// `SpecResult` so the speedup can be checked rather than taken on faith. If
/// the speculative half failed to load or produced no draft statistics, the
/// baseline is returned untouched with its error preserved — a broken pair
/// must not read as "speculation made no difference".
fn merge_speculative(baseline: BenchResult, spec: BenchResult) -> BenchResult {
    if !spec.loaded {
        return BenchResult {
            error: spec.error.or(baseline.error.clone()),
            ..baseline
        };
    }
    let (draft_n, draft_n_accepted) = match spec.speculative.as_ref() {
        Some(s) => (s.draft_n, s.draft_n_accepted),
        None => (0, 0),
    };
    let speedup = if baseline.decode_tok_s > 0.0 {
        spec.decode_tok_s / baseline.decode_tok_s
    } else {
        0.0
    };
    BenchResult {
        speculative: Some(SpecResult {
            draft_n,
            draft_n_accepted,
            accept_rate: if draft_n > 0 {
                draft_n_accepted as f64 / draft_n as f64
            } else {
                0.0
            },
            baseline_decode_tok_s: baseline.decode_tok_s,
            decode_tok_s: spec.decode_tok_s,
            speedup,
        }),
        ..spec
    }
}

fn benchmark_one(model_path: &str, cfg: &BenchConfig, draft: Option<&str>) -> BenchResult {
    let mgr = LlamaManager::new();
    let server_cfg = LlamaServerConfig {
        model_path: model_path.to_string(),
        n_gpu_layers: Some(cfg.n_gpu_layers),
        ctx_size: Some(cfg.ctx_size),
        port: BENCH_PORT,
        draft_model_path: draft.map(str::to_string),
        // Offload the whole draft. A draft running on the CPU is slower than
        // the model it is drafting for, which would measure the wrong thing.
        draft_n_gpu_layers: draft.map(|_| 999),
        draft_n_max: cfg.draft_n_max,
        ..Default::default()
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
    // Only meaningful on the with-draft half; `merge_speculative` fills in the
    // baseline and speedup once both halves have run.
    let draft_stats = draft.and_then(|_| body.as_deref().and_then(parse_draft_stats));

    let _ = mgr.stop();

    BenchResult {
        n_gpu_layers: cfg.n_gpu_layers,
        ctx_size: cfg.ctx_size,
        loaded: true,
        load_ms,
        prefill_tok_s,
        decode_tok_s,
        peak_vram_bytes,
        speculative: draft_stats.map(|(draft_n, draft_n_accepted)| SpecResult {
            draft_n,
            draft_n_accepted,
            accept_rate: 0.0,
            baseline_decode_tok_s: 0.0,
            decode_tok_s,
            speedup: 0.0,
        }),
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

/// Pull `(draft_n, draft_n_accepted)` out of a completion response.
///
/// These are the only direct evidence that speculation is working: they are
/// *not* exposed on the Prometheus `/metrics` endpoint the telemetry cockpit
/// polls (that surface has no draft counters at all), so a per-response read
/// is the only way to get them.
///
/// Checked under `timings` and at the top level. llama.cpp reports most
/// per-request counters inside `timings`, but the draft pair is newer than
/// that convention and the two spellings have moved between releases; looking
/// in both costs nothing and avoids silently reporting a 0% accept rate —
/// which would be indistinguishable from a draft that is genuinely useless.
fn parse_draft_stats(body: &str) -> Option<(u64, u64)> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let pick = |obj: &serde_json::Value| -> Option<(u64, u64)> {
        let n = obj.get("draft_n")?.as_u64()?;
        let acc = obj.get("draft_n_accepted")?.as_u64()?;
        Some((n, acc))
    };
    v.get("timings").and_then(pick).or_else(|| pick(&v))
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
            r.peak_vram_bytes as f64 / GIB,
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

    fn result(decode: f64, loaded: bool) -> BenchResult {
        BenchResult {
            n_gpu_layers: 99,
            ctx_size: 4096,
            loaded,
            load_ms: 1000,
            prefill_tok_s: 500.0,
            decode_tok_s: decode,
            peak_vram_bytes: 8_000_000_000,
            speculative: None,
            error: None,
        }
    }

    #[test]
    fn reads_draft_stats_from_timings() {
        let body = r#"{"content":"x","timings":{"predicted_n":96,"predicted_ms":500.0,
                      "draft_n":120,"draft_n_accepted":78}}"#;
        assert_eq!(parse_draft_stats(body), Some((120, 78)));
    }

    /// The counters have moved between llama.cpp releases, so the top level is
    /// checked too. Reading neither would report a 0% accept rate, which is
    /// indistinguishable from a genuinely useless draft.
    #[test]
    fn reads_draft_stats_from_the_top_level_too() {
        let body = r#"{"draft_n":50,"draft_n_accepted":25,"timings":{"predicted_n":96}}"#;
        assert_eq!(parse_draft_stats(body), Some((50, 25)));
    }

    #[test]
    fn absent_draft_stats_are_none_not_zero() {
        let body = r#"{"timings":{"predicted_n":96,"predicted_ms":500.0}}"#;
        assert_eq!(parse_draft_stats(body), None);
    }

    #[test]
    fn merge_reports_accept_rate_and_speedup() {
        let baseline = result(40.0, true);
        let mut spec = result(68.0, true);
        spec.speculative = Some(SpecResult {
            draft_n: 200,
            draft_n_accepted: 150,
            accept_rate: 0.0,
            baseline_decode_tok_s: 0.0,
            decode_tok_s: 68.0,
            speedup: 0.0,
        });
        let merged = merge_speculative(baseline, spec);
        let s = merged.speculative.expect("speculative result");
        assert_eq!(s.accept_rate, 0.75);
        assert_eq!(s.baseline_decode_tok_s, 40.0);
        assert_eq!(s.speedup, 1.7);
        // The headline numbers become the speculative ones, since that is the
        // configuration being proposed.
        assert_eq!(merged.decode_tok_s, 68.0);
    }

    /// A draft that makes things slower is a real outcome and must be visible,
    /// not rounded away into "no difference".
    #[test]
    fn a_slower_draft_reports_a_speedup_below_one() {
        let baseline = result(50.0, true);
        let mut spec = result(35.0, true);
        spec.speculative = Some(SpecResult {
            draft_n: 200,
            draft_n_accepted: 20,
            accept_rate: 0.0,
            baseline_decode_tok_s: 0.0,
            decode_tok_s: 35.0,
            speedup: 0.0,
        });
        let s = merge_speculative(baseline, spec).speculative.unwrap();
        assert_eq!(s.accept_rate, 0.1);
        assert!(s.speedup < 1.0, "got {}", s.speedup);
    }

    /// If the speculative half never loaded — a mismatched pair, or one that
    /// did not fit — the baseline is kept and the error surfaced. Reporting a
    /// bare baseline would read as "speculation changed nothing".
    #[test]
    fn a_failed_speculative_half_keeps_the_error() {
        let baseline = result(40.0, true);
        let mut failed = result(0.0, false);
        failed.error = Some("draft model vocab must match target model".into());
        let merged = merge_speculative(baseline, failed);
        assert!(merged.speculative.is_none());
        assert_eq!(merged.decode_tok_s, 40.0);
        assert!(merged.error.unwrap().contains("vocab must match"));
    }

    #[test]
    fn a_zero_baseline_does_not_produce_infinite_speedup() {
        let baseline = result(0.0, true);
        let mut spec = result(30.0, true);
        spec.speculative = Some(SpecResult {
            draft_n: 10, draft_n_accepted: 5, accept_rate: 0.0,
            baseline_decode_tok_s: 0.0, decode_tok_s: 30.0, speedup: 0.0,
        });
        let s = merge_speculative(baseline, spec).speculative.unwrap();
        assert_eq!(s.speedup, 0.0);
        assert!(s.speedup.is_finite());
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
            BenchConfig { n_gpu_layers: 999, ctx_size: 4096, ..Default::default() }, // full offload
            BenchConfig { n_gpu_layers: 12, ctx_size: 4096, ..Default::default() },  // partial (slower)
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
                r.peak_vram_bytes as f64 / GIB,
                r.error.as_deref().unwrap_or(""),
            );
        });
        assert_eq!(results.len(), 2);
        assert!(results[0].loaded, "full offload should load");
        assert!(results[0].decode_tok_s > 0.0, "should measure a decode rate");
    }
}
