//! Hardware-aware auto-config estimator (Pillar 2, tier 1).
//!
//! Given a model's GGUF shape and a GPU's VRAM, estimate the best
//! (GPU-offload layers, context size) that fits — the recommendation nobody
//! else surfaces. This is the fast *arithmetic* tier; a later tier will refine
//! it by actually benchmarking candidate configs.
//!
//! The KV-cache term is the context-dependent one and dominates the tradeoff:
//!   kv_bytes = n_layers_on_gpu * ctx * n_head_kv * (key_len + value_len) * elem_bytes
//! Weights are approximated from the GGUF file size scaled by the offload
//! fraction, plus a flat compute/overhead fudge.

use serde::Serialize;

use crate::gguf::GgufMetadata;

/// Fraction of total VRAM we're willing to budget (leave headroom for the
/// desktop compositor, driver, and our own estimation slop).
const VRAM_HEADROOM: f64 = 0.90;

/// Flat allowance for llama.cpp's compute buffers + misc allocations.
const OVERHEAD_BYTES: u64 = 400 * 1024 * 1024;

/// f16 KV cache: 2 bytes per element (the llama.cpp default).
const KV_ELEM_BYTES: u64 = 2;

/// Candidate context sizes to consider, filtered to the model's native max.
const CANDIDATE_CTX: &[u32] = &[
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
];

/// The model dimensions the estimator needs, resolved from GGUF metadata with
/// sensible fallbacks noted in `assumptions`.
#[derive(Debug, Clone)]
pub struct ModelShape {
    pub file_size: u64,
    pub n_layers: u64,
    pub n_head_kv: u64,
    pub head_dim_k: u64,
    pub head_dim_v: u64,
    pub native_ctx: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextOption {
    pub ctx: u32,
    pub est_total_bytes: u64,
    pub fits: bool,
}

/// One quant level evaluated by the advisor.
#[derive(Debug, Clone, Serialize)]
pub struct QuantOption {
    pub label: String,
    pub est_weights_bytes: u64,
    /// Headroom left under the VRAM budget after weights + KV + overhead
    /// (negative values are clamped to 0 with fits=false).
    pub headroom_bytes: u64,
    pub fits: bool,
    pub is_current: bool,
}

/// "Which GGUF quant should I download for this GPU?" — the sweet spot is the
/// highest-quality quant that fully fits with KV + headroom to spare.
#[derive(Debug, Clone, Serialize)]
pub struct QuantAdvice {
    /// Estimated parameter count used (from metadata, or derived from file
    /// size / current quant bpw).
    pub est_params_b: f64,
    pub current_label: Option<String>,
    pub current_fits: bool,
    /// Best-quality quant that fully fits (None if even the smallest doesn't).
    pub recommended: Option<String>,
    pub options: Vec<QuantOption>,
}

/// Known GGUF quant levels, best quality first, with effective bits-per-weight
/// (llama.cpp's commonly cited averages, which include the quant's metadata).
const QUANT_LADDER: &[(&str, f64)] = &[
    ("F16", 16.0),
    ("Q8_0", 8.5),
    ("Q6_K", 6.59),
    ("Q5_K_M", 5.69),
    ("Q4_K_M", 4.85),
    ("IQ4_XS", 4.25),
    ("Q3_K_M", 3.91),
    ("IQ3_XS", 3.30),
    ("Q2_K", 2.63),
];

fn bpw_for(label: &str) -> Option<f64> {
    // Match loosely: "Q4_K_M" but also mixed labels like "Q4_K - Medium".
    let up = label.to_ascii_uppercase();
    QUANT_LADDER
        .iter()
        .find(|(l, _)| up.starts_with(l) || up.replace(' ', "").starts_with(l))
        .map(|(_, b)| *b)
        .or(match up.as_str() {
            "F32" => Some(32.0),
            "BF16" => Some(16.0),
            "Q4_K_S" => Some(4.58),
            "Q5_K_S" => Some(5.54),
            "Q5_0" => Some(5.54),
            "Q4_0" => Some(4.55),
            "Q3_K_S" => Some(3.50),
            "Q3_K_L" => Some(4.27),
            _ => None,
        })
}

/// Build quant advice for a model: which quant levels of *this same model*
/// would fit this GPU, and which is the sweet spot. Parameter count comes from
/// metadata when present, else is derived from file size and the current
/// quant's bits-per-weight.
pub fn quant_advice(
    shape: &ModelShape,
    current_quant: Option<&str>,
    metadata_params: Option<u64>,
    gpu_total: u64,
) -> Option<QuantAdvice> {
    let current_bpw = current_quant.and_then(bpw_for);
    let params: f64 = match metadata_params {
        Some(p) if p > 0 => p as f64,
        _ => {
            // Derive from file size: params = bytes * 8 / bpw.
            let bpw = current_bpw?;
            shape.file_size as f64 * 8.0 / bpw
        }
    };

    let budget = (gpu_total as f64 * VRAM_HEADROOM) as u64;
    // Judge each quant at a practical working context (KV at 8K or native max).
    let ctx = 8192u64.min(shape.native_ctx);
    let kv = kv_bytes(shape, ctx, shape.n_layers);

    let current_up = current_quant.map(|c| c.to_ascii_uppercase());
    let mut options = Vec::new();
    let mut recommended: Option<String> = None;
    let mut current_fits = false;

    for (label, bpw) in QUANT_LADDER {
        // ~5% on top of raw weights for embeddings/output layers not captured
        // by average bpw.
        let weights = (params * bpw / 8.0 * 1.05) as u64;
        let total = weights + kv + OVERHEAD_BYTES;
        let fits = total <= budget;
        let headroom = budget.saturating_sub(total);
        let is_current = current_up
            .as_deref()
            .map(|c| c.starts_with(label))
            .unwrap_or(false);
        if is_current && fits {
            current_fits = true;
        }
        if fits && recommended.is_none() {
            recommended = Some(label.to_string());
        }
        options.push(QuantOption {
            label: label.to_string(),
            est_weights_bytes: weights,
            headroom_bytes: headroom,
            fits,
            is_current,
        });
    }

    Some(QuantAdvice {
        est_params_b: params / 1e9,
        current_label: current_quant.map(str::to_string),
        current_fits,
        recommended,
        options,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct VramEstimate {
    pub fits: bool,
    pub full_offload: bool,
    pub n_gpu_layers: u32,
    pub ctx_size: u32,
    pub est_weights_bytes: u64,
    pub est_kv_bytes: u64,
    pub est_overhead_bytes: u64,
    pub est_total_bytes: u64,
    pub budget_bytes: u64,
    pub gpu_total_bytes: u64,
    pub gpu_free_bytes: u64,
    /// Full-offload feasibility across candidate contexts (for a tradeoff view).
    pub context_options: Vec<ContextOption>,
    /// Which quant levels of this model would fit this GPU (filled by the
    /// command layer, which knows the file's quant + parameter count).
    pub quant_advice: Option<QuantAdvice>,
    pub notes: Vec<String>,
}

/// Resolve a `ModelShape` from GGUF metadata + on-disk file size, recording any
/// assumptions made when fields are missing.
pub fn shape_from_metadata(
    md: &GgufMetadata,
    file_size: u64,
    notes: &mut Vec<String>,
) -> Option<ModelShape> {
    let n_layers = md.block_count.filter(|&v| v > 0).or_else(|| {
        notes.push("block_count missing; cannot estimate layers".into());
        None
    })?;

    let embedding_length = md.embedding_length.unwrap_or(0);
    let n_head = md.head_count.filter(|&v| v > 0);
    let n_head_kv = md
        .head_count_kv
        .or(n_head)
        .filter(|&v| v > 0)
        .unwrap_or_else(|| {
            notes.push("head_count_kv missing; assuming 8".into());
            8
        });

    // head_dim: prefer explicit key/value lengths; else embedding / n_head.
    let derived_head_dim = match (embedding_length, n_head) {
        (e, Some(h)) if e > 0 && h > 0 => e / h,
        _ => {
            notes.push("head dim unknown; assuming 128".into());
            128
        }
    };
    let head_dim_k = md.key_length.filter(|&v| v > 0).unwrap_or(derived_head_dim);
    let head_dim_v = md
        .value_length
        .filter(|&v| v > 0)
        .unwrap_or(derived_head_dim);

    let native_ctx = md.context_length.filter(|&v| v > 0).unwrap_or_else(|| {
        notes.push("context_length missing; assuming 8192".into());
        8192
    });

    Some(ModelShape {
        file_size,
        n_layers,
        n_head_kv,
        head_dim_k,
        head_dim_v,
        native_ctx,
    })
}

fn kv_bytes(shape: &ModelShape, ctx: u64, layers_on_gpu: u64) -> u64 {
    layers_on_gpu
        .saturating_mul(ctx)
        .saturating_mul(shape.n_head_kv)
        .saturating_mul(shape.head_dim_k + shape.head_dim_v)
        .saturating_mul(KV_ELEM_BYTES)
}

fn weights_bytes(shape: &ModelShape, layers_on_gpu: u64) -> u64 {
    if shape.n_layers == 0 {
        return 0;
    }
    // Approximate: file size scaled by offloaded layer fraction.
    ((shape.file_size as u128 * layers_on_gpu as u128) / shape.n_layers as u128) as u64
}

/// Compute a recommendation for `shape` given the GPU's VRAM.
pub fn estimate(shape: &ModelShape, gpu_total: u64, gpu_free: u64, mut notes: Vec<String>) -> VramEstimate {
    let budget = (gpu_total as f64 * VRAM_HEADROOM) as u64;

    let candidates: Vec<u32> = CANDIDATE_CTX
        .iter()
        .copied()
        .filter(|&c| (c as u64) <= shape.native_ctx)
        .chain(std::iter::once(shape.native_ctx.min(u32::MAX as u64) as u32))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    // Full-offload feasibility per candidate context.
    let weights_full = weights_bytes(shape, shape.n_layers);
    let context_options: Vec<ContextOption> = candidates
        .iter()
        .map(|&ctx| {
            let total =
                weights_full + OVERHEAD_BYTES + kv_bytes(shape, ctx as u64, shape.n_layers);
            ContextOption {
                ctx,
                est_total_bytes: total,
                fits: total <= budget,
            }
        })
        .collect();

    // Prefer full offload at the largest context that fits.
    if let Some(best) = context_options.iter().filter(|o| o.fits).max_by_key(|o| o.ctx) {
        let kv = kv_bytes(shape, best.ctx as u64, shape.n_layers);
        return VramEstimate {
            fits: true,
            full_offload: true,
            n_gpu_layers: shape.n_layers.min(u32::MAX as u64) as u32,
            ctx_size: best.ctx,
            est_weights_bytes: weights_full,
            est_kv_bytes: kv,
            est_overhead_bytes: OVERHEAD_BYTES,
            est_total_bytes: best.est_total_bytes,
            budget_bytes: budget,
            gpu_total_bytes: gpu_total,
            gpu_free_bytes: gpu_free,
            context_options,
            quant_advice: None,
            notes,
        };
    }

    // Can't fully offload — pick a modest context and maximize offloaded layers.
    let ctx = 4096u64.min(shape.native_ctx);
    let mut best_layers = 0u64;
    for n in (0..=shape.n_layers).rev() {
        let total = weights_bytes(shape, n) + OVERHEAD_BYTES + kv_bytes(shape, ctx, n);
        if total <= budget {
            best_layers = n;
            break;
        }
    }

    if best_layers == 0 {
        notes.push("model won't fit in VRAM even partially; will run on CPU".into());
    } else {
        notes.push(format!(
            "partial offload: {best_layers}/{} layers fit at {ctx} ctx",
            shape.n_layers
        ));
    }

    let weights = weights_bytes(shape, best_layers);
    let kv = kv_bytes(shape, ctx, best_layers);
    let total = weights + OVERHEAD_BYTES + kv;
    VramEstimate {
        fits: best_layers > 0,
        full_offload: false,
        n_gpu_layers: best_layers.min(u32::MAX as u64) as u32,
        ctx_size: ctx as u32,
        est_weights_bytes: weights,
        est_kv_bytes: kv,
        est_overhead_bytes: OVERHEAD_BYTES,
        est_total_bytes: total,
        budget_bytes: budget,
        gpu_total_bytes: gpu_total,
        gpu_free_bytes: gpu_free,
        context_options,
        quant_advice: None,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A ~7B-ish shape: 32 layers, GQA 8 kv heads, head_dim 128, 4.3 GB file.
    fn shape_7b() -> ModelShape {
        ModelShape {
            file_size: 4_300_000_000,
            n_layers: 32,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            native_ctx: 131072,
        }
    }

    #[test]
    fn full_offload_on_a_big_gpu() {
        // 24 GB GPU easily fits a 4.3 GB 7B model; should recommend full offload.
        let est = estimate(&shape_7b(), 24 * 1_000_000_000, 22 * 1_000_000_000, vec![]);
        assert!(est.fits);
        assert!(est.full_offload);
        assert_eq!(est.n_gpu_layers, 32);
        assert!(est.ctx_size >= 4096);
        assert!(est.est_total_bytes <= est.budget_bytes);
    }

    #[test]
    fn larger_gpu_allows_larger_context() {
        let small = estimate(&shape_7b(), 8 * 1_000_000_000, 8 * 1_000_000_000, vec![]);
        let big = estimate(&shape_7b(), 24 * 1_000_000_000, 24 * 1_000_000_000, vec![]);
        assert!(big.ctx_size >= small.ctx_size);
    }

    #[test]
    fn quant_advisor_recommends_downsizing_for_fp16() {
        // A 14B model at F16 on a 16 GB GPU: F16/Q8 can't fit, a mid quant can.
        let shape = ModelShape {
            file_size: 28_000_000_000, // ~14B * 16bpw
            n_layers: 40,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            native_ctx: 32768,
        };
        let advice = quant_advice(&shape, Some("F16"), Some(14_000_000_000), 16_000_000_000)
            .expect("advice");
        assert!(!advice.current_fits, "F16 14B must not fit in 16GB");
        let rec = advice.recommended.expect("some quant should fit");
        assert!(
            ["Q6_K", "Q5_K_M", "Q4_K_M"].contains(&rec.as_str()),
            "expected a mid quant, got {rec}"
        );
        // Ladder must be ordered best-first and the recommended one must fit
        // with headroom.
        let rec_opt = advice.options.iter().find(|o| o.label == rec).unwrap();
        assert!(rec_opt.fits && rec_opt.headroom_bytes > 0);
    }

    #[test]
    fn quant_advisor_keeps_small_models_at_high_quality() {
        // A 4B model on a 16 GB GPU: even Q8_0 fits; F16 likely too.
        let shape = ModelShape {
            file_size: 2_800_000_000,
            n_layers: 36,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            native_ctx: 32768,
        };
        let advice =
            quant_advice(&shape, Some("Q4_K_M"), Some(4_000_000_000), 16_000_000_000).unwrap();
        assert!(advice.current_fits);
        let rec = advice.recommended.unwrap();
        assert!(
            ["F16", "Q8_0"].contains(&rec.as_str()),
            "small model should recommend high quality, got {rec}"
        );
    }

    #[test]
    fn quant_advisor_derives_params_from_file_size() {
        // No metadata param count: derive from size/bpw (4.85 bpw Q4_K_M).
        let shape = ModelShape {
            file_size: 4_850_000_000,
            n_layers: 32,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            native_ctx: 8192,
        };
        let advice = quant_advice(&shape, Some("Q4_K_M"), None, 24_000_000_000).unwrap();
        assert!((advice.est_params_b - 8.0).abs() < 0.5, "~8B expected, got {}", advice.est_params_b);
    }

    #[test]
    fn partial_offload_when_weights_exceed_vram() {
        // A 40 GB model on an 8 GB GPU can't fully offload.
        let mut shape = shape_7b();
        shape.file_size = 40_000_000_000;
        let est = estimate(&shape, 8 * 1_000_000_000, 8 * 1_000_000_000, vec![]);
        assert!(!est.full_offload);
        assert!(est.n_gpu_layers < shape.n_layers as u32);
    }

    /// Estimate configs for real local models against this machine's real GPU.
    /// Ignored by default; run with:
    ///   cargo test -- --ignored --nocapture estimate_real_models
    #[test]
    #[ignore]
    fn estimate_real_models() {
        use crate::gguf::read_gguf_metadata;
        use nvml_wrapper::Nvml;
        use std::path::PathBuf;

        let Some(home) = dirs::home_dir() else {
            return;
        };
        let (gpu_total, gpu_free) = match Nvml::init()
            .ok()
            .and_then(|n| n.device_by_index(0).ok().and_then(|d| d.memory_info().ok()))
        {
            Some(m) => (m.total, m.free),
            None => {
                eprintln!("no NVML; skipping");
                return;
            }
        };
        let gb = |b: u64| b as f64 / 1e9;
        println!(
            "\nGPU VRAM: {:.1} GB total, {:.1} GB free\n",
            gb(gpu_total),
            gb(gpu_free)
        );

        let base = home.join(".lmstudio/models/lmstudio-community");
        let models = [
            "NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf",
            "GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf",
            "Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf",
        ];
        for rel in models {
            let path = PathBuf::from(&base).join(rel);
            if !path.is_file() {
                continue;
            }
            let Ok(md) = read_gguf_metadata(&path) else {
                continue;
            };
            let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let mut notes = Vec::new();
            let Some(shape) = shape_from_metadata(&md, file_size, &mut notes) else {
                continue;
            };
            let est = estimate(&shape, gpu_total, gpu_free, notes);
            let name = path.file_name().unwrap().to_string_lossy();
            println!("{name}  ({:.1} GB, {} layers)", gb(file_size), shape.n_layers);
            println!(
                "  -> {} | ngl={} ctx={} | weights {:.1} + kv {:.1} + oh {:.1} = {:.1} GB / budget {:.1} GB",
                if est.full_offload { "FULL offload" } else { "partial" },
                est.n_gpu_layers,
                est.ctx_size,
                gb(est.est_weights_bytes),
                gb(est.est_kv_bytes),
                gb(est.est_overhead_bytes),
                gb(est.est_total_bytes),
                gb(est.budget_bytes),
            );
            for n in &est.notes {
                println!("     note: {n}");
            }
        }
    }
}
