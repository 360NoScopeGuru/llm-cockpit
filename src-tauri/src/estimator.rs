/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

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

/// KV cache element type. Quantizing the cache is the cheapest way to buy
/// context on fixed VRAM: the cache is the only term that scales with context
/// length, so halving its element size roughly doubles the context that fits.
///
/// Sizes are llama.cpp's block formats, so they include per-block scale
/// overhead: q8_0 packs 32 values into 34 bytes (8.5 bpw), q4_0 into 18 bytes
/// (4.5 bpw). Held as bits×2 so the arithmetic stays exact in integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvType {
    fn bits_x2(self) -> u64 {
        match self {
            KvType::F16 => 32, // 16.0 bpw
            KvType::Q8_0 => 17, // 8.5 bpw
            KvType::Q4_0 => 9,  // 4.5 bpw
        }
    }

    /// Parse the llama.cpp `-ctk`/`-ctv` spelling. Unknown values fall back to
    /// f16 — the conservative choice, since it over-estimates rather than
    /// promising context that will not fit.
    pub fn parse(s: Option<&str>) -> Self {
        match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("q8_0") => KvType::Q8_0,
            Some("q4_0") => KvType::Q4_0,
            _ => KvType::F16,
        }
    }
}

/// Context to aim for when a model cannot be fully offloaded. Below this,
/// agent runs and long documents fill the window mid-task; the layers traded
/// away to reach it are the cheaper loss.
const TARGET_PARTIAL_CTX: u64 = 8192;

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
    /// True when every layer fits on the GPU at this context.
    pub fits: bool,
    /// Layers that actually fit at this context. A rung with `fits: false` but
    /// a non-zero count is still perfectly usable — just partially offloaded.
    /// Reporting only `fits` made the whole ladder read as unusable on models
    /// that were running fine at partial offload.
    pub n_gpu_layers: u32,
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

pub(crate) fn bpw_for(label: &str) -> Option<f64> {
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
    kv_type: KvType,
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
    let kv = kv_bytes(shape, ctx, shape.n_layers, kv_type);

    let current_up = current_quant.map(|c| c.to_ascii_uppercase());
    let mut options = Vec::new();
    let mut recommended: Option<String> = None;
    // Judge the current file by its own bpw, not by ladder membership — quants
    // like Q4_K_S or Q3_K_L aren't ladder rows but still have a known size.
    let mut current_fits = current_bpw
        .map(|bpw| {
            let weights = (params * bpw / 8.0 * 1.05) as u64;
            weights + kv + OVERHEAD_BYTES <= budget
        })
        .unwrap_or(false);

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

/// How a target/draft pair lands on the GPU when speculating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairVerdict {
    /// Both models fully on the GPU, and the target keeps every layer it had
    /// on its own. The only shape worth recommending.
    Fits,
    /// The pair fits, but only by pushing target layers onto the CPU. Those
    /// layers now run at a fraction of the speed on *every* token, which is
    /// almost always a bigger loss than speculation is a win.
    CostsTargetLayers,
    /// The pair does not fit. There is no partially-offloaded-draft verdict:
    /// a draft running partly on CPU is slower than the model it is meant to
    /// run ahead of, so that configuration is a failure, not an option.
    TooBig,
}

/// VRAM arithmetic for running a draft model alongside its target.
///
/// Speculation needs both models resident simultaneously, each with its own KV
/// cache. The interesting number is not "does it fit" but what it *costs*: a
/// draft that fits only by evicting target layers to system RAM slows down
/// every token the target produces, which no accept rate repays.
#[derive(Debug, Clone, Serialize)]
pub struct PairEstimate {
    pub verdict: PairVerdict,
    pub ctx_size: u32,
    pub target_layers_on_gpu: u32,
    pub target_layers_total: u32,
    /// Target layers that were on the GPU without a draft and are not with
    /// one. This is the real price of speculating.
    pub target_layers_evicted: u32,
    pub draft_layers_on_gpu: u32,
    pub draft_layers_total: u32,
    pub est_target_bytes: u64,
    pub est_draft_bytes: u64,
    pub est_total_bytes: u64,
    pub budget_bytes: u64,
    pub notes: Vec<String>,
}

/// Budget a target and a draft model together on one GPU.
///
/// The draft is allocated first and in full. That is not favouritism — a draft
/// that is partly on the CPU is slower than the model it is drafting for,
/// making the whole exercise pointless, so there is no useful configuration in
/// which the draft is partially offloaded. Whatever remains goes to the
/// target, and the layers it loses are reported.
///
/// Both models are charged `OVERHEAD_BYTES` for compute buffers. Two loaded
/// models really do mean two sets of them; charging the draft the same flat
/// allowance as the target overstates its cost, which is the safe direction
/// for a recommendation — this predicts "won't fit" slightly too eagerly
/// rather than promising a pair that then OOMs.
///
/// The draft's KV cache is sized at the *target's* context. llama-server does
/// not expose a separate draft context length, and assuming the full window is
/// the conservative reading.
pub fn estimate_pair(
    target: &ModelShape,
    draft: &ModelShape,
    ctx: u64,
    gpu_total: u64,
    kv: KvType,
) -> PairEstimate {
    let budget = (gpu_total as f64 * VRAM_HEADROOM) as u64;
    let mut notes: Vec<String> = Vec::new();

    // What the target manages on its own, so the cost of adding a draft can be
    // stated rather than guessed at.
    let baseline_layers = max_layers_at(target, ctx, budget, kv);

    let draft_weights = weights_bytes(draft, draft.n_layers);
    let draft_kv = kv_bytes(draft, ctx, draft.n_layers, kv);
    let draft_total = draft_weights + draft_kv + OVERHEAD_BYTES;

    let base = |verdict, target_layers: u64, draft_layers: u64, target_bytes: u64| PairEstimate {
        verdict,
        ctx_size: ctx.min(u32::MAX as u64) as u32,
        target_layers_on_gpu: target_layers.min(u32::MAX as u64) as u32,
        target_layers_total: target.n_layers.min(u32::MAX as u64) as u32,
        target_layers_evicted: baseline_layers.saturating_sub(target_layers).min(u32::MAX as u64)
            as u32,
        draft_layers_on_gpu: draft_layers.min(u32::MAX as u64) as u32,
        draft_layers_total: draft.n_layers.min(u32::MAX as u64) as u32,
        est_target_bytes: target_bytes,
        est_draft_bytes: draft_total,
        est_total_bytes: target_bytes + draft_total,
        budget_bytes: budget,
        notes: Vec::new(),
    };

    if draft_total > budget {
        notes.push("the draft model alone does not fit in VRAM".into());
        let mut e = base(PairVerdict::TooBig, 0, 0, 0);
        e.notes = notes;
        return e;
    }

    let remaining = budget - draft_total;
    let target_layers = max_layers_at(target, ctx, remaining, kv);
    let target_bytes = if target_layers > 0 {
        weights_bytes(target, target_layers) + kv_bytes(target, ctx, target_layers, kv) + OVERHEAD_BYTES
    } else {
        0
    };

    let verdict = if target_layers == 0 {
        notes.push("no room left for the target model once the draft is loaded".into());
        PairVerdict::TooBig
    } else if target_layers < baseline_layers {
        notes.push(format!(
            "speculating costs {} target layer(s): {baseline_layers} fit alone, {target_layers} with the draft",
            baseline_layers - target_layers
        ));
        PairVerdict::CostsTargetLayers
    } else {
        // No layers lost. The target may still be partially offloaded, but
        // that was already true before the draft was added, so it is not a
        // cost of speculating — worth saying out loud rather than letting
        // "Fits" imply everything is on the GPU.
        if target_layers < target.n_layers {
            notes.push(format!(
                "the draft is free here, but the target was already partial: \
                 {target_layers}/{} layers on GPU",
                target.n_layers
            ));
        }
        PairVerdict::Fits
    };

    let mut e = base(verdict, target_layers, draft.n_layers, target_bytes);
    e.notes = notes;
    e
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

fn kv_bytes(shape: &ModelShape, ctx: u64, layers_on_gpu: u64, kv: KvType) -> u64 {
    layers_on_gpu
        .saturating_mul(ctx)
        .saturating_mul(shape.n_head_kv)
        .saturating_mul(shape.head_dim_k + shape.head_dim_v)
        .saturating_mul(kv.bits_x2())
        / 16
}

/// Most layers that fit on the GPU at a given context, 0 if none do.
fn max_layers_at(shape: &ModelShape, ctx: u64, budget: u64, kv: KvType) -> u64 {
    for n in (0..=shape.n_layers).rev() {
        if weights_bytes(shape, n) + OVERHEAD_BYTES + kv_bytes(shape, ctx, n, kv) <= budget {
            return n;
        }
    }
    0
}

fn weights_bytes(shape: &ModelShape, layers_on_gpu: u64) -> u64 {
    if shape.n_layers == 0 {
        return 0;
    }
    // Approximate: file size scaled by offloaded layer fraction.
    ((shape.file_size as u128 * layers_on_gpu as u128) / shape.n_layers as u128) as u64
}

/// Compute a recommendation for `shape` given the GPU's VRAM.
pub fn estimate(
    shape: &ModelShape,
    gpu_total: u64,
    gpu_free: u64,
    kv: KvType,
    mut notes: Vec<String>,
) -> VramEstimate {
    let budget = (gpu_total as f64 * VRAM_HEADROOM) as u64;

    let candidates: Vec<u32> = CANDIDATE_CTX
        .iter()
        .copied()
        .filter(|&c| (c as u64) <= shape.native_ctx)
        .chain(std::iter::once(shape.native_ctx.min(u32::MAX as u64) as u32))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    // Per candidate context: does it fit fully, and if not, how many layers do
    // fit? Both numbers matter — a model that cannot fully offload is still
    // usable, and the largest usable context is usually a partial-offload one.
    let weights_full = weights_bytes(shape, shape.n_layers);
    let context_options: Vec<ContextOption> = candidates
        .iter()
        .map(|&ctx| {
            let total =
                weights_full + OVERHEAD_BYTES + kv_bytes(shape, ctx as u64, shape.n_layers, kv);
            ContextOption {
                ctx,
                est_total_bytes: total,
                fits: total <= budget,
                n_gpu_layers: max_layers_at(shape, ctx as u64, budget, kv)
                    .min(u32::MAX as u64) as u32,
            }
        })
        .collect();

    // Prefer full offload at the largest context that fits.
    if let Some(best) = context_options.iter().filter(|o| o.fits).max_by_key(|o| o.ctx) {
        let kv = kv_bytes(shape, best.ctx as u64, shape.n_layers, kv);
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

    // Can't fully offload. Aim for a context that is actually usable rather
    // than the smallest one that works: 4K is below what agent or long-document
    // work needs, and the layers given up to reach 8K cost far less than
    // running out of context mid-task. Step down only if the target won't fit.
    let target = TARGET_PARTIAL_CTX.min(shape.native_ctx);
    let (ctx, best_layers) = [target, 4096, 2048]
        .into_iter()
        .map(|c| c.min(shape.native_ctx))
        .filter(|&c| c > 0)
        .map(|c| (c, max_layers_at(shape, c, budget, kv)))
        .find(|&(_, layers)| layers > 0)
        .unwrap_or((4096u64.min(shape.native_ctx).max(1), 0));

    if best_layers == 0 {
        notes.push("model won't fit in VRAM even partially; will run on CPU".into());
    } else {
        notes.push(format!(
            "partial offload: {best_layers}/{} layers fit at {ctx} ctx",
            shape.n_layers
        ));
        if best_layers < shape.n_layers {
            notes.push(
                "more context is available lower down the ladder — each rung trades \
                 GPU layers (speed) for context length"
                    .into(),
            );
        }
    }

    let weights = weights_bytes(shape, best_layers);
    let kv_b = kv_bytes(shape, ctx, best_layers, kv);
    let total = weights + OVERHEAD_BYTES + kv_b;
    VramEstimate {
        fits: best_layers > 0,
        full_offload: false,
        n_gpu_layers: best_layers.min(u32::MAX as u64) as u32,
        ctx_size: ctx as u32,
        est_weights_bytes: weights,
        est_kv_bytes: kv_b,
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
        let est = estimate(&shape_7b(), 24 * 1_000_000_000, 22 * 1_000_000_000, KvType::F16, vec![]);
        assert!(est.fits);
        assert!(est.full_offload);
        assert_eq!(est.n_gpu_layers, 32);
        assert!(est.ctx_size >= 4096);
        assert!(est.est_total_bytes <= est.budget_bytes);
    }

    #[test]
    fn larger_gpu_allows_larger_context() {
        let small = estimate(&shape_7b(), 8 * 1_000_000_000, 8 * 1_000_000_000, KvType::F16, vec![]);
        let big = estimate(&shape_7b(), 24 * 1_000_000_000, 24 * 1_000_000_000, KvType::F16, vec![]);
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
        let advice = quant_advice(&shape, Some("F16"), Some(14_000_000_000), 16_000_000_000, KvType::F16)
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
            quant_advice(&shape, Some("Q4_K_M"), Some(4_000_000_000), 16_000_000_000, KvType::F16).unwrap();
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
        let advice = quant_advice(&shape, Some("Q4_K_M"), None, 24_000_000_000, KvType::F16).unwrap();
        assert!((advice.est_params_b - 8.0).abs() < 0.5, "~8B expected, got {}", advice.est_params_b);
    }

    #[test]
    fn partial_offload_when_weights_exceed_vram() {
        // A 40 GB model on an 8 GB GPU can't fully offload.
        let mut shape = shape_7b();
        shape.file_size = 40_000_000_000;
        let est = estimate(&shape, 8 * 1_000_000_000, 8 * 1_000_000_000, KvType::F16, vec![]);
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
            let est = estimate(&shape, gpu_total, gpu_free, KvType::F16, notes);
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

#[cfg(test)]
mod ladder {
    use super::*;

    fn shape(file_size: u64, n_layers: u64) -> ModelShape {
        ModelShape {
            file_size,
            n_layers,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            native_ctx: 262_144,
        }
    }

    /// A model too big to fully offload is still perfectly usable. The ladder
    /// used to report only full-offload feasibility, so every rung read as
    /// unusable on exactly the models people run at partial offload.
    #[test]
    fn partial_rungs_report_usable_layers() {
        // 27B Q4_K_M on a 17.1 GB card: never fully offloads.
        let est = estimate(&shape(16_500_000_000, 64), 17_100_000_000, 17_100_000_000, KvType::F16, vec![]);
        assert!(!est.full_offload);
        assert!(est.context_options.iter().all(|o| !o.fits));
        // ...yet every rung up to a large context still runs partially.
        for o in est.context_options.iter().filter(|o| o.ctx <= 32_768) {
            assert!(o.n_gpu_layers > 0, "{}K should be usable", o.ctx / 1024);
        }
        // Bigger context costs layers — the tradeoff must be monotonic.
        let layers_at = |c: u32| {
            est.context_options.iter().find(|o| o.ctx == c).unwrap().n_gpu_layers
        };
        assert!(layers_at(4096) > layers_at(8192));
        assert!(layers_at(8192) > layers_at(16384));
    }

    /// The partial path used to hardcode 4096, so a big model on a big card
    /// always landed at 4K however much room a few fewer layers would buy.
    #[test]
    fn partial_offload_targets_a_usable_context() {
        let est = estimate(&shape(16_500_000_000, 64), 17_100_000_000, 17_100_000_000, KvType::F16, vec![]);
        assert_eq!(est.ctx_size, 8192);
        assert!(est.n_gpu_layers > 0 && est.n_gpu_layers < 64);
    }

    /// A model that fits comfortably should still prefer full offload at the
    /// largest context, not get dragged down by the partial-path target.
    #[test]
    fn small_model_still_fully_offloads_at_a_large_context() {
        let est = estimate(&shape(4_700_000_000, 28), 17_100_000_000, 17_100_000_000, KvType::F16, vec![]);
        assert!(est.full_offload);
        assert!(est.ctx_size >= 32_768, "got {}", est.ctx_size);
    }
}

#[cfg(test)]
mod kv_quant {
    use super::*;

    fn shape_27b() -> ModelShape {
        ModelShape {
            file_size: 16_500_000_000,
            n_layers: 64,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            native_ctx: 262_144,
        }
    }

    #[test]
    fn quantized_kv_costs_proportionally_less() {
        let s = shape_27b();
        let f16 = kv_bytes(&s, 8192, 64, KvType::F16);
        let q8 = kv_bytes(&s, 8192, 64, KvType::Q8_0);
        let q4 = kv_bytes(&s, 8192, 64, KvType::Q4_0);
        // 8.5 and 4.5 bpw against 16 — block scales included, so not exactly
        // half and a quarter.
        assert_eq!(q8, f16 * 17 / 32);
        assert_eq!(q4, f16 * 9 / 32);
        assert!(q8 < f16 && q4 < q8);
    }

    /// The point of the feature: the same card holds more context.
    #[test]
    fn quantized_kv_buys_layers_and_context() {
        let s = shape_27b();
        let gpu = 17_100_000_000;
        let f16 = estimate(&s, gpu, gpu, KvType::F16, vec![]);
        let q8 = estimate(&s, gpu, gpu, KvType::Q8_0, vec![]);

        let layers_at = |e: &VramEstimate, c: u32| {
            e.context_options.iter().find(|o| o.ctx == c).unwrap().n_gpu_layers
        };
        // At any given context, quantized KV leaves room for more layers.
        assert!(layers_at(&q8, 16384) > layers_at(&f16, 16384));
        assert!(layers_at(&q8, 32768) > layers_at(&f16, 32768));
        // And the KV term itself is smaller at the chosen config.
        assert!(q8.est_kv_bytes < f16.est_kv_bytes);
    }

    #[test]
    fn parses_names() {
        assert_eq!(KvType::parse(Some("q8_0")), KvType::Q8_0);
        assert_eq!(KvType::parse(Some("Q4_0")), KvType::Q4_0);
        assert_eq!(KvType::parse(Some("f16")), KvType::F16);
        // Unknown/absent must fall back to f16 so we never promise context
        // that the server will not actually have room for.
        assert_eq!(KvType::parse(Some("nonsense")), KvType::F16);
        assert_eq!(KvType::parse(None), KvType::F16);
    }
}

#[cfg(test)]
mod spec_pair {
    use super::*;

    fn shape(file_size: u64, n_layers: u64) -> ModelShape {
        ModelShape {
            file_size,
            n_layers,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            native_ctx: 131072,
        }
    }

    fn shape_7b() -> ModelShape {
        shape(4_300_000_000, 32)
    }

    /// A plausible small draft: ~0.6B at Q4, few layers.
    fn draft_shape() -> ModelShape {
        ModelShape {
            file_size: 500_000_000,
            n_layers: 28,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            native_ctx: 32768,
        }
    }

    #[test]
    fn a_small_draft_on_a_big_gpu_costs_the_target_nothing() {
        let e = estimate_pair(
            &shape_7b(),
            &draft_shape(),
            8192,
            24 * 1_000_000_000,
            KvType::F16,
        );
        assert_eq!(e.verdict, PairVerdict::Fits);
        assert_eq!(e.target_layers_evicted, 0);
        assert_eq!(e.target_layers_on_gpu, 32);
        assert_eq!(e.draft_layers_on_gpu, 28);
        assert!(e.est_total_bytes <= e.budget_bytes);
    }

    /// The verdict that matters. On a GPU where the target only just fits,
    /// adding a draft has to come out of the target's layers — and layers
    /// pushed to the CPU slow down every token, which no accept rate repays.
    #[test]
    fn a_draft_that_evicts_target_layers_is_called_out() {
        let target = shape(10_000_000_000, 40);
        let alone = estimate(&target, 12 * 1_000_000_000, 12 * 1_000_000_000, KvType::F16, vec![]);
        let pair = estimate_pair(
            &target,
            &draft_shape(),
            8192,
            12 * 1_000_000_000,
            KvType::F16,
        );
        assert_eq!(pair.verdict, PairVerdict::CostsTargetLayers);
        assert!(
            pair.target_layers_evicted > 0,
            "expected the draft to cost layers; target alone got {}",
            alone.n_gpu_layers
        );
        assert!(pair.notes.iter().any(|n| n.contains("costs")));
    }

    #[test]
    fn a_draft_too_big_for_the_gpu_is_rejected_outright() {
        let e = estimate_pair(
            &shape_7b(),
            &shape(30_000_000_000, 60),
            8192,
            8 * 1_000_000_000,
            KvType::F16,
        );
        assert_eq!(e.verdict, PairVerdict::TooBig);
        assert_eq!(e.target_layers_on_gpu, 0);
    }

    /// The draft's cache is charged at the target's context, so a longer
    /// window costs VRAM twice over. That compounding is the whole reason
    /// speculation can stop fitting as context grows.
    #[test]
    fn draft_kv_scales_with_the_targets_context() {
        let short = estimate_pair(
            &shape_7b(),
            &draft_shape(),
            2048,
            24 * 1_000_000_000,
            KvType::F16,
        );
        let long = estimate_pair(
            &shape_7b(),
            &draft_shape(),
            16384,
            24 * 1_000_000_000,
            KvType::F16,
        );
        assert!(
            long.est_draft_bytes > short.est_draft_bytes,
            "draft KV must grow with context: {} vs {}",
            short.est_draft_bytes,
            long.est_draft_bytes
        );
    }

    #[test]
    fn quantized_kv_buys_back_room_for_the_pair() {
        let f16 = estimate_pair(
            &shape_7b(),
            &draft_shape(),
            32768,
            12 * 1_000_000_000,
            KvType::F16,
        );
        let q8 = estimate_pair(
            &shape_7b(),
            &draft_shape(),
            32768,
            12 * 1_000_000_000,
            KvType::Q8_0,
        );
        assert!(
            q8.est_total_bytes < f16.est_total_bytes,
            "q8_0 must shrink the pair's footprint"
        );
        assert!(q8.target_layers_on_gpu >= f16.target_layers_on_gpu);
    }

    /// A target that was already partially offloaded before any draft existed
    /// should not have that blamed on the draft.
    #[test]
    fn a_pre_existing_partial_target_is_not_charged_to_the_draft() {
        let target = shape(20_000_000_000, 60);
        let gpu = 10 * 1_000_000_000;
        let alone = max_layers_at(&target, 8192, (gpu as f64 * VRAM_HEADROOM) as u64, KvType::F16);
        assert!(alone > 0 && alone < 60, "target must start out partial");
        let pair = estimate_pair(&target, &draft_shape(), 8192, gpu, KvType::F16);
        if pair.target_layers_evicted == 0 && pair.verdict == PairVerdict::Fits {
            assert!(pair.notes.iter().any(|n| n.contains("already partial")));
        }
    }
}
