/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Draft-model compatibility for speculative decoding.
//!
//! Speculative decoding runs a small "draft" model ahead of the real one and
//! lets the big model verify several tokens per forward pass. It only works if
//! both models share a tokenizer — otherwise the draft's token IDs mean
//! something different to the target, and `llama-server` refuses to start.
//!
//! Finding that out from a failed launch is a bad experience: the server exits
//! with a message in a log the user never sees, after a multi-second model
//! load. Since Tokamak already parses GGUF metadata for the library view, it
//! can answer the same question up front, for every model on disk, in the time
//! it takes to read a few KB of header.
//!
//! ## What is actually checked
//!
//! These mirror `common_speculative_are_compatible` in llama.cpp, which
//! rejects a pair when any of the following differ:
//!
//! | Check | GGUF key |
//! |---|---|
//! | vocab type | `tokenizer.ggml.model` |
//! | BOS token id and whether it is auto-added | `tokenizer.ggml.bos_token_id`, `…add_bos_token` |
//! | EOS token id and whether it is auto-added | `tokenizer.ggml.eos_token_id`, `…add_eos_token` |
//! | vocab size, within a tolerance | length of `tokenizer.ggml.tokens` |
//!
//! Note the size rule is a *tolerance*, not equality — llama.cpp's own wording
//! is "vocab must closely match". Models in one family routinely differ by a
//! few padding tokens, and demanding equality would reject pairs that work.
//!
//! ## What cannot be checked here
//!
//! llama.cpp also compares the *text* of every token from id 256 upward. That
//! needs the vocab materialized, which this parser deliberately skips
//! (`gguf.rs`) so that scanning a library of 50 models stays cheap. So a
//! `Compatible` verdict means "no reason to reject it was found", not "it is
//! guaranteed to load" — the remaining failure mode is two models with
//! identically-shaped but differently-populated vocabs, which in practice
//! means different model families that happen to share a tokenizer size.

use serde::Serialize;

use crate::estimator::{self, KvType, PairEstimate, PairVerdict};
use crate::gguf::GgufMetadata;
use crate::scanner::ModelEntry;

/// The hardware and launch settings a pair would actually run under. Without
/// this, ranking can only reason about compute cost; with it, it can say
/// whether the two models fit on the card at once.
#[derive(Debug, Clone, Copy)]
pub struct PairContext {
    pub gpu_total_bytes: u64,
    pub ctx: u64,
    pub kv: KvType,
}

/// Largest permitted difference in vocab size, mirroring llama.cpp's
/// `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`. Hard-coded upstream, so it can drift; if
/// a pair this says is fine gets rejected by the server with "difference N,
/// max allowed M", M is the number to update here.
const MAX_VOCAB_SIZE_DIFFERENCE: u64 = 128;

/// Whether a model can serve as a draft for a given target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum DraftVerdict {
    /// Nothing disqualifying found. See the module note on what this does not
    /// promise.
    Compatible,
    /// At least one of llama.cpp's rules is definitely violated. Launching
    /// this pair will fail.
    Incompatible { reasons: Vec<String> },
    /// One of the models is missing the metadata needed to decide. Offer it,
    /// but do not claim it will work.
    Unverifiable { reasons: Vec<String> },
}

/// Compare two parsed GGUF headers and decide whether `draft` can draft for
/// `target`.
///
/// Definite mismatches win over missing metadata: if the vocab types disagree
/// *and* the BOS id is absent, the pair is `Incompatible`, not `Unverifiable`
/// — there is no point qualifying a verdict that is already decided.
pub fn check_draft(target: &GgufMetadata, draft: &GgufMetadata) -> DraftVerdict {
    let mut bad: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();

    match (target.vocab_type.as_deref(), draft.vocab_type.as_deref()) {
        (Some(t), Some(d)) if t != d => {
            bad.push(format!("vocab type differs: target is {t}, draft is {d}"));
        }
        (Some(_), Some(_)) => {}
        _ => unknown.push("vocab type missing from one of the models".into()),
    }

    match (target.vocab_size, draft.vocab_size) {
        (Some(t), Some(d)) => {
            let diff = t.abs_diff(d);
            if diff > MAX_VOCAB_SIZE_DIFFERENCE {
                bad.push(format!(
                    "vocab size differs by {diff} tokens ({t} vs {d}); \
                     llama.cpp allows at most {MAX_VOCAB_SIZE_DIFFERENCE}"
                ));
            }
        }
        _ => unknown.push("vocab size missing from one of the models".into()),
    }

    check_special_token(
        "BOS",
        target.bos_token_id,
        draft.bos_token_id,
        target.add_bos_token,
        draft.add_bos_token,
        &mut bad,
        &mut unknown,
    );
    check_special_token(
        "EOS",
        target.eos_token_id,
        draft.eos_token_id,
        target.add_eos_token,
        draft.add_eos_token,
        &mut bad,
        &mut unknown,
    );

    if !bad.is_empty() {
        DraftVerdict::Incompatible { reasons: bad }
    } else if !unknown.is_empty() {
        DraftVerdict::Unverifiable { reasons: unknown }
    } else {
        DraftVerdict::Compatible
    }
}

/// BOS/EOS must agree on both the token id and whether it is auto-prepended.
///
/// The `add_*` flags are compared only when both models state them: llama.cpp
/// derives a default from the vocab type when the key is absent, so two files
/// that both omit it still agree, and treating absence as a mismatch would
/// reject valid pairs.
#[allow(clippy::too_many_arguments)]
fn check_special_token(
    label: &str,
    target_id: Option<u64>,
    draft_id: Option<u64>,
    target_add: Option<bool>,
    draft_add: Option<bool>,
    bad: &mut Vec<String>,
    unknown: &mut Vec<String>,
) {
    match (target_id, draft_id) {
        (Some(t), Some(d)) if t != d => {
            bad.push(format!("{label} token id differs: target {t}, draft {d}"));
        }
        (Some(_), Some(_)) => {}
        _ => unknown.push(format!("{label} token id missing from one of the models")),
    }

    if let (Some(t), Some(d)) = (target_add, draft_add) {
        if t != d {
            bad.push(format!(
                "{label} auto-add differs: target {t}, draft {d}"
            ));
        }
    }
}

/// Whether a draft is cheap enough relative to its target to be worth running.
///
/// Compatibility and economics are different questions: `llama-server` will
/// run any compatible pair, including ones that are slower than not
/// speculating at all. Only the first of these is worth recommending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DraftEconomics {
    /// Roughly an order of magnitude cheaper — the usual shape of a useful
    /// draft.
    Recommended,
    /// Cheaper, but not by enough to be confident. Worth benchmarking rather
    /// than assuming.
    Marginal,
    /// As expensive as the target, or worse. Drafting will cost more than it
    /// saves.
    Counterproductive,
    /// Not enough metadata to say.
    Unknown,
}

/// Ratio of draft cost to target cost at or below which a draft is worth
/// recommending outright, and the ceiling above which it is a clear loss.
const RECOMMEND_MAX_RATIO: f64 = 0.20;
const MARGINAL_MAX_RATIO: f64 = 0.60;

/// A model on disk considered as a draft for some target.
#[derive(Debug, Clone, Serialize)]
pub struct DraftCandidate {
    pub path: String,
    /// What to show in the picker — Ollama blobs have hash file names, so the
    /// scanner's `display_name` wins when present.
    pub label: String,
    pub size_bytes: u64,
    pub parameter_count: Option<u64>,
    pub quant_label: Option<String>,
    pub verdict: DraftVerdict,
    /// Estimated parameters actually evaluated per token — for a
    /// mixture-of-experts model, far fewer than it contains.
    pub active_params: Option<u64>,
    /// This candidate's active parameters as a fraction of the target's.
    pub cost_ratio: Option<f64>,
    pub economics: DraftEconomics,
    /// What running this pair actually costs in VRAM. `None` when no GPU
    /// context was supplied or either model's shape could not be resolved.
    pub pair: Option<PairEstimate>,
}

/// Total parameters, stated if the file says so and derived from its size if
/// not.
///
/// `general.parameter_count` turns out to be absent from most GGUFs in the
/// wild — none of the fifteen models on the development machine carried it —
/// so relying on it alone left every verdict `Unknown`. The fallback inverts
/// the estimator's own weights formula (`params × bpw / 8 × 1.05`), which
/// recovers the count to within a few percent: a 27B Q4_K_M file measures back
/// as 26B.
fn total_params(meta: &GgufMetadata, size_bytes: u64) -> Option<u64> {
    if let Some(p) = meta.parameter_count.filter(|p| *p > 0) {
        return Some(p);
    }
    if size_bytes == 0 {
        return None;
    }
    let bpw = meta
        .quant_label
        .as_deref()
        .and_then(crate::estimator::bpw_for)?;
    Some((size_bytes as f64 * 8.0 / bpw / 1.05) as u64)
}

/// Estimate the parameters actually evaluated per token.
///
/// A dense model runs all of them. A mixture-of-experts model runs only
/// `expert_used_count` of `expert_count`, which is why a 35B-A3B file can be
/// cheaper per token than a dense 4B despite being five times the size — and
/// why comparing file sizes picks disastrous drafts.
///
/// Scaling the whole parameter count by the expert ratio *understates* the
/// real figure, because attention and embeddings run every token regardless.
/// That is the safe direction here: it makes the target look cheaper, so a
/// draft has to be genuinely small to earn a recommendation.
fn active_params(meta: &GgufMetadata, size_bytes: u64) -> Option<u64> {
    let total = total_params(meta, size_bytes)?;
    match (meta.expert_count, meta.expert_used_count) {
        (Some(n), Some(used)) if n > 0 && used < n => {
            Some(((total as f64) * (used as f64 / n as f64)) as u64)
        }
        _ => Some(total),
    }
}

/// Rank every scanned model as a possible draft for `target`.
///
/// Ordering is compatible first, then cheapest first.
///
/// When `pair_ctx` is supplied, each candidate is also budgeted against the
/// target on the real GPU, and one that only fits by evicting target layers is
/// demoted to `Counterproductive` however cheap it looks per token: layers
/// pushed to system RAM slow down every token the target produces, which no
/// accept rate repays. Without a GPU context, the ranking falls back to
/// comparing compute cost alone.
pub fn rank_drafts(
    target_path: &str,
    target: &GgufMetadata,
    target_size_bytes: u64,
    models: &[ModelEntry],
    pair_ctx: Option<PairContext>,
) -> Vec<DraftCandidate> {
    let target_cost = active_params(target, target_size_bytes);
    // Resolving the target's shape can fail on sparse metadata; if it does,
    // no pair arithmetic is possible for any candidate.
    let target_shape = pair_ctx.and_then(|_| {
        let mut ignored = Vec::new();
        estimator::shape_from_metadata(target, target_size_bytes, &mut ignored)
    });

    let mut out: Vec<DraftCandidate> = models
        .iter()
        .filter(|m| {
            // A model cannot draft for itself, and neither shard tails nor
            // vision projectors are loadable as a standalone draft. Nor is
            // anything the runtime will refuse outright — offering a draft
            // that cannot load only moves the failure to launch time.
            m.path != target_path
                && !m.is_shard_continuation
                && !m.is_mmproj
                && m.load_blocker.is_none()
        })
        .filter_map(|m| {
            let meta = m.metadata.as_ref()?;
            let cost = active_params(meta, m.size_bytes);
            let ratio = match (cost, target_cost) {
                (Some(d), Some(t)) if t > 0 => Some(d as f64 / t as f64),
                _ => None,
            };
            let mut economics = match ratio {
                Some(r) if r <= RECOMMEND_MAX_RATIO => DraftEconomics::Recommended,
                Some(r) if r <= MARGINAL_MAX_RATIO => DraftEconomics::Marginal,
                Some(_) => DraftEconomics::Counterproductive,
                None => DraftEconomics::Unknown,
            };
            // Budget the two models together when the hardware is known.
            let pair = match (pair_ctx, target_shape.as_ref()) {
                (Some(cx), Some(ts)) => {
                    let mut ignored = Vec::new();
                    estimator::shape_from_metadata(meta, m.size_bytes, &mut ignored).map(|ds| {
                        estimator::estimate_pair(ts, &ds, cx.ctx, cx.gpu_total_bytes, cx.kv)
                    })
                }
                _ => None,
            };

            match pair.as_ref().map(|p| p.verdict) {
                // A pair that does not fit, or fits only by demoting the
                // target to partial offload, is not a trade worth making.
                Some(PairVerdict::TooBig) | Some(PairVerdict::CostsTargetLayers) => {
                    economics = DraftEconomics::Counterproductive;
                }
                Some(PairVerdict::Fits) => {}
                // No hardware context. Fall back to the crude guard: both
                // models are resident at once, so a draft whose weights
                // outweigh the target's cannot be recommended however cheap it
                // is per token — a sparse MoE is genuinely fast and still has
                // to fit.
                None => {
                    if m.size_bytes >= target_size_bytes
                        && economics == DraftEconomics::Recommended
                    {
                        economics = DraftEconomics::Marginal;
                    }
                }
            }

            Some(DraftCandidate {
                path: m.path.clone(),
                label: m
                    .display_name
                    .clone()
                    .unwrap_or_else(|| m.file_name.clone()),
                size_bytes: m.size_bytes,
                parameter_count: meta.parameter_count,
                quant_label: meta.quant_label.clone(),
                verdict: check_draft(target, meta),
                active_params: cost,
                cost_ratio: ratio,
                economics,
                pair,
            })
        })
        .collect();

    out.sort_by(|a, b| {
        let rank = |c: &DraftCandidate| match c.verdict {
            DraftVerdict::Compatible => 0,
            DraftVerdict::Unverifiable { .. } => 1,
            DraftVerdict::Incompatible { .. } => 2,
        };
        let tier = |c: &DraftCandidate| match c.economics {
            DraftEconomics::Recommended => 0,
            DraftEconomics::Marginal => 1,
            DraftEconomics::Unknown => 2,
            DraftEconomics::Counterproductive => 3,
        };
        rank(a)
            .cmp(&rank(b))
            .then(tier(a).cmp(&tier(b)))
            // Ratios are finite by construction (guarded on t > 0), so an
            // unorderable comparison here is impossible; fall back on size.
            .then(
                a.cost_ratio
                    .partial_cmp(&b.cost_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.size_bytes.cmp(&b.size_bytes))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A metadata block with a complete, self-consistent tokenizer identity.
    fn meta(vocab_type: &str, vocab_size: u64) -> GgufMetadata {
        GgufMetadata {
            version: 3,
            tensor_count: 0,
            metadata_kv_count: 0,
            architecture: Some("qwen3".into()),
            name: None,
            quant_label: None,
            file_type: None,
            context_length: None,
            block_count: None,
            embedding_length: None,
            head_count: None,
            head_count_kv: None,
            key_length: None,
            value_length: None,
            parameter_count: None,
            size_label: None,
            split_count: None,
            split_no: None,
            expert_count: None,
            expert_used_count: None,
            has_norm_epsilon: true,
            vocab_type: Some(vocab_type.into()),
            vocab_size: Some(vocab_size),
            bos_token_id: Some(151643),
            eos_token_id: Some(151645),
            add_bos_token: Some(false),
            add_eos_token: Some(false),
        }
    }

    #[test]
    fn identical_tokenizers_are_compatible() {
        let t = meta("gpt2", 151936);
        let d = meta("gpt2", 151936);
        assert_eq!(check_draft(&t, &d), DraftVerdict::Compatible);
    }

    #[test]
    fn small_vocab_difference_is_tolerated() {
        // Same family, a handful of padding tokens apart. llama.cpp allows
        // this, so rejecting it would cost the user a working pair.
        let t = meta("gpt2", 151936);
        let d = meta("gpt2", 151936 - 64);
        assert_eq!(check_draft(&t, &d), DraftVerdict::Compatible);
        // Tolerance is inclusive at the boundary.
        let edge = meta("gpt2", 151936 - MAX_VOCAB_SIZE_DIFFERENCE);
        assert_eq!(check_draft(&t, &edge), DraftVerdict::Compatible);
    }

    #[test]
    fn large_vocab_difference_is_rejected() {
        let t = meta("gpt2", 151936);
        let d = meta("gpt2", 32000);
        match check_draft(&t, &d) {
            DraftVerdict::Incompatible { reasons } => {
                assert!(reasons.iter().any(|r| r.contains("vocab size differs")));
            }
            other => panic!("expected incompatible, got {other:?}"),
        }
    }

    #[test]
    fn vocab_type_mismatch_is_rejected() {
        let t = meta("gpt2", 32000);
        let d = meta("llama", 32000);
        assert!(matches!(
            check_draft(&t, &d),
            DraftVerdict::Incompatible { .. }
        ));
    }

    #[test]
    fn special_token_mismatch_is_rejected() {
        let t = meta("gpt2", 151936);
        let mut d = meta("gpt2", 151936);
        d.eos_token_id = Some(2);
        match check_draft(&t, &d) {
            DraftVerdict::Incompatible { reasons } => {
                assert!(reasons.iter().any(|r| r.contains("EOS token id differs")));
            }
            other => panic!("expected incompatible, got {other:?}"),
        }
    }

    #[test]
    fn add_flags_must_agree_when_both_present() {
        let t = meta("gpt2", 151936);
        let mut d = meta("gpt2", 151936);
        d.add_bos_token = Some(true);
        assert!(matches!(
            check_draft(&t, &d),
            DraftVerdict::Incompatible { .. }
        ));
    }

    #[test]
    fn absent_add_flags_on_both_sides_still_agree() {
        // llama.cpp derives a default from the vocab type, so two files that
        // both omit the key do not disagree.
        let mut t = meta("gpt2", 151936);
        let mut d = meta("gpt2", 151936);
        t.add_bos_token = None;
        d.add_bos_token = None;
        assert_eq!(check_draft(&t, &d), DraftVerdict::Compatible);
    }

    #[test]
    fn missing_metadata_is_unverifiable_not_compatible() {
        let t = meta("gpt2", 151936);
        let mut d = meta("gpt2", 151936);
        d.vocab_size = None;
        match check_draft(&t, &d) {
            DraftVerdict::Unverifiable { reasons } => {
                assert!(reasons.iter().any(|r| r.contains("vocab size")));
            }
            other => panic!("expected unverifiable, got {other:?}"),
        }
    }

    fn entry(path: &str, size: u64, meta: Option<GgufMetadata>) -> ModelEntry {
        ModelEntry {
            path: path.into(),
            file_name: path.into(),
            size_bytes: size,
            source: "folder".into(),
            display_name: None,
            is_shard_continuation: false,
            shard_total: None,
            is_mmproj: false,
            metadata: meta,
            parse_error: None,
            load_blocker: None,
        }
    }

    /// Same tokenizer as `meta`, with a parameter count so economics apply.
    fn dense(params: u64) -> GgufMetadata {
        let mut m = meta("gpt2", 151936);
        m.parameter_count = Some(params);
        m
    }

    /// A mixture-of-experts model: `used` of `total` experts run per token.
    fn moe(params: u64, used: u64, total: u64) -> GgufMetadata {
        let mut m = dense(params);
        m.expert_used_count = Some(used);
        m.expert_count = Some(total);
        m
    }

    #[test]
    fn active_params_discounts_unused_experts() {
        assert_eq!(active_params(&dense(14_000_000_000), 0), Some(14_000_000_000));
        // 8 of 256 experts => ~3% of the parameters run per token.
        let a = active_params(&moe(35_000_000_000, 8, 256), 0).unwrap();
        assert!(
            (1_000_000_000..=1_200_000_000).contains(&a),
            "expected ~1.09B active, got {a}"
        );
    }

    /// `general.parameter_count` is absent from most real GGUFs, so the size
    /// derivation is the path that actually runs. Pinned against a real file:
    /// Qwen3.6-27B Q4_K_M measures 15.41 GiB on disk.
    #[test]
    fn parameter_count_is_derived_from_file_size_when_absent() {
        let mut m = meta("gpt2", 248320);
        m.parameter_count = None;
        m.quant_label = Some("Q4_K_M".into());
        let bytes = (15.41 * 1024.0 * 1024.0 * 1024.0) as u64;
        let p = total_params(&m, bytes).expect("derivable from size + quant");
        assert!(
            (25_000_000_000..=28_000_000_000).contains(&p),
            "expected ~26-27B for a 15.41GiB Q4_K_M file, got {p}"
        );
    }

    #[test]
    fn a_stated_parameter_count_wins_over_the_derivation() {
        let mut m = meta("gpt2", 151936);
        m.parameter_count = Some(14_000_000_000);
        m.quant_label = Some("Q4_K_M".into());
        assert_eq!(total_params(&m, 999_999_999), Some(14_000_000_000));
    }

    #[test]
    fn params_are_unknown_without_a_count_or_a_quant_label() {
        let mut m = meta("gpt2", 151936);
        m.parameter_count = None;
        m.quant_label = None;
        assert_eq!(total_params(&m, 5_000_000_000), None);
    }

    #[test]
    fn ranking_puts_cheapest_compatible_draft_first() {
        let target = dense(14_000_000_000);
        let models = vec![
            entry("incompatible.gguf", 1_000, Some(meta("llama", 32000))),
            entry("mid.gguf", 9_000, Some(dense(4_000_000_000))),
            entry("tiny.gguf", 2_000, Some(dense(600_000_000))),
        ];
        let ranked = rank_drafts("target.gguf", &target, 30_000, &models, None);
        assert_eq!(ranked[0].path, "tiny.gguf");
        assert_eq!(ranked[0].economics, DraftEconomics::Recommended);
        assert_eq!(ranked[1].path, "mid.gguf");
        // 4B against 14B is 29% — cheaper, but not by enough to promise.
        assert_eq!(ranked[1].economics, DraftEconomics::Marginal);
        // The incompatible one is still listed: the user should see *why* it
        // is unavailable rather than have it silently vanish.
        assert_eq!(ranked[2].path, "incompatible.gguf");
        assert!(matches!(
            ranked[2].verdict,
            DraftVerdict::Incompatible { .. }
        ));
    }

    /// The case that a file-size comparison gets catastrophically wrong: a
    /// dense 27B "draft" is a third the size of a 35B MoE file, but runs ~25x
    /// the parameters per token, so speculating with it is far slower than not
    /// speculating at all.
    #[test]
    fn dense_draft_for_an_moe_target_is_counterproductive() {
        let target = moe(35_000_000_000, 8, 256); // ~1.1B active
        let models = vec![entry("dense-27b.gguf", 15_000, Some(dense(27_000_000_000)))];
        let ranked = rank_drafts("target.gguf", &target, 30_000, &models, None);
        assert_eq!(ranked[0].economics, DraftEconomics::Counterproductive);
        assert!(ranked[0].size_bytes < 30_000, "the file really is smaller");
    }

    /// A sparse MoE can be cheaper per token than a smaller dense model while
    /// carrying far more weight on disk. Cheap to run is not the same as
    /// affordable to load, and both models are resident at once.
    #[test]
    fn a_draft_larger_than_its_target_is_never_recommended() {
        let target = dense(27_000_000_000);
        let models = vec![entry(
            "sparse-but-huge.gguf",
            40_000,
            Some(moe(35_000_000_000, 8, 256)),
        )];
        let ranked = rank_drafts("target.gguf", &target, 30_000, &models, None);
        // ~1.1B active against 27B is 4% — cheap by compute alone.
        assert!(ranked[0].cost_ratio.unwrap() < RECOMMEND_MAX_RATIO);
        assert_eq!(ranked[0].economics, DraftEconomics::Marginal);
    }

    /// A draft can be cheap per token and still be the wrong call: if loading
    /// it pushes target layers onto the CPU, every token the target produces
    /// gets slower. With a GPU context supplied, that outranks the compute
    /// argument entirely.
    #[test]
    fn a_draft_that_evicts_target_layers_is_demoted_once_vram_is_known() {
        let mut target = dense(14_000_000_000);
        target.block_count = Some(40);
        target.head_count = Some(40);
        target.head_count_kv = Some(8);
        target.embedding_length = Some(5120);
        target.context_length = Some(32768);
        let mut draft_md = dense(600_000_000);
        draft_md.block_count = Some(28);
        draft_md.head_count = Some(16);
        draft_md.head_count_kv = Some(8);
        draft_md.embedding_length = Some(2048);
        draft_md.context_length = Some(32768);

        let models = vec![entry("draft.gguf", 500_000_000, Some(draft_md))];

        // On a roomy card the draft is free and stays recommended.
        let roomy = rank_drafts(
            "target.gguf",
            &target,
            9_000_000_000,
            &models,
            Some(PairContext {
                gpu_total_bytes: 24 * 1024 * 1024 * 1024,
                ctx: 8192,
                kv: KvType::F16,
            }),
        );
        assert_eq!(roomy[0].economics, DraftEconomics::Recommended);
        assert_eq!(
            roomy[0].pair.as_ref().unwrap().verdict,
            PairVerdict::Fits
        );

        // On a card where the target only just fits, the same draft is a loss.
        let tight = rank_drafts(
            "target.gguf",
            &target,
            9_000_000_000,
            &models,
            Some(PairContext {
                gpu_total_bytes: 10 * 1024 * 1024 * 1024,
                ctx: 8192,
                kv: KvType::F16,
            }),
        );
        let p = tight[0].pair.as_ref().unwrap();
        assert!(p.target_layers_evicted > 0, "expected evicted layers");
        assert_eq!(tight[0].economics, DraftEconomics::Counterproductive);
        // The compute argument is unchanged — only the VRAM reality differs.
        assert!(tight[0].cost_ratio.unwrap() < RECOMMEND_MAX_RATIO);
    }

    #[test]
    fn missing_parameter_counts_leave_economics_unknown() {
        let target = meta("gpt2", 151936); // no parameter_count
        let models = vec![entry("x.gguf", 500, Some(dense(600_000_000)))];
        let ranked = rank_drafts("target.gguf", &target, 30_000, &models, None);
        assert_eq!(ranked[0].economics, DraftEconomics::Unknown);
        assert_eq!(ranked[0].cost_ratio, None);
    }

    #[test]
    fn target_shards_and_projectors_are_not_offered() {
        let target = dense(14_000_000_000);
        let mut shard = entry("shard-2.gguf", 100, Some(dense(600_000_000)));
        shard.is_shard_continuation = true;
        let mut proj = entry("mmproj.gguf", 100, Some(dense(600_000_000)));
        proj.is_mmproj = true;
        let models = vec![
            entry("target.gguf", 30_000, Some(dense(14_000_000_000))),
            shard,
            proj,
            entry("ok.gguf", 500, Some(dense(600_000_000))),
        ];
        let ranked = rank_drafts("target.gguf", &target, 30_000, &models, None);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].path, "ok.gguf");
    }

    /// A model stock llama.cpp refuses to load is not a draft candidate,
    /// however compatible its tokenizer looks. Ollama's gemma3 blobs are
    /// exactly this: genuinely compatible with each other, and unloadable.
    #[test]
    fn models_the_runtime_rejects_are_not_offered() {
        let target = dense(14_000_000_000);
        let mut blocked = entry("unloadable.gguf", 500, Some(dense(600_000_000)));
        blocked.load_blocker = Some("missing gemma3.attention.layer_norm_rms_epsilon".into());
        let models = vec![blocked, entry("fine.gguf", 500, Some(dense(600_000_000)))];
        let ranked = rank_drafts("target.gguf", &target, 30_000, &models, None);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].path, "fine.gguf");
    }

    #[test]
    fn unparsed_models_are_skipped_entirely() {
        let target = dense(14_000_000_000);
        let models = vec![entry("broken.gguf", 500, None)];
        assert!(rank_drafts("target.gguf", &target, 30_000, &models, None).is_empty());
    }

    /// Rank the real library against every real model on this machine, so the
    /// metadata rules above are checked against files that actually exist
    /// rather than only against synthetic headers. Ignored by default
    /// (machine-dependent); run with:
    ///   cargo test -- --ignored --nocapture rank_real_library
    #[test]
    #[ignore]
    fn rank_real_library() {
        let models = crate::scanner::scan_models(&[]);
        let targets: Vec<_> = models
            .iter()
            .filter(|m| !m.is_shard_continuation && !m.is_mmproj && m.metadata.is_some())
            .collect();
        // Budget against a plausible 16 GiB card at a working context, so the
        // pair verdicts are exercised rather than skipped.
        let pair_ctx = Some(PairContext {
            gpu_total_bytes: 16 * 1024 * 1024 * 1024,
            ctx: 8192,
            kv: KvType::F16,
        });
        println!("\n--- {} candidate target(s), budgeted on 16GiB ---", targets.len());

        for t in &targets {
            let meta = t.metadata.as_ref().unwrap();
            let ranked = rank_drafts(&t.path, meta, t.size_bytes, &models, pair_ctx);
            // Show every tokenizer-compatible candidate with the reason it was
            // or was not recommended. Filtering silently here would hide
            // whether a "no usable draft" result is real arithmetic or a bug.
            let usable: Vec<_> = ranked
                .iter()
                .filter(|c| matches!(c.verdict, DraftVerdict::Compatible))
                .collect();
            println!(
                "\nTARGET {}  (vocab {} / {}, experts {}/{})",
                t.display_name.as_deref().unwrap_or(&t.file_name),
                meta.vocab_size
                    .map(|v| v.to_string())
                    .unwrap_or("?".into()),
                meta.vocab_type.as_deref().unwrap_or("?"),
                meta.expert_used_count
                    .map(|v| v.to_string())
                    .unwrap_or("-".into()),
                meta.expert_count.map(|v| v.to_string()).unwrap_or("-".into()),
            );
            if usable.is_empty() {
                println!("  no tokenizer-compatible model in the library");
            }
            for c in usable.iter().take(3) {
                let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
                println!(
                    "  DRAFT {:<44} {:.2}GB  cost {}  -> {:?}",
                    c.label,
                    gb(c.size_bytes),
                    c.cost_ratio
                        .map(|r| format!("{:.1}%", r * 100.0))
                        .unwrap_or("?".into()),
                    c.economics,
                );
                if let Some(p) = &c.pair {
                    println!(
                        "        pair {:?}: target {}/{} layers (-{}), draft {}/{}, {:.2}+{:.2}GB of {:.2}GB",
                        p.verdict,
                        p.target_layers_on_gpu,
                        p.target_layers_total,
                        p.target_layers_evicted,
                        p.draft_layers_on_gpu,
                        p.draft_layers_total,
                        gb(p.est_target_bytes),
                        gb(p.est_draft_bytes),
                        gb(p.budget_bytes),
                    );
                }
            }
        }
    }

    #[test]
    fn definite_mismatch_outranks_missing_metadata() {
        // A pair that is already disqualified should not be softened to
        // "unverifiable" just because some other key happens to be absent.
        let t = meta("gpt2", 151936);
        let mut d = meta("llama", 151936);
        d.bos_token_id = None;
        assert!(matches!(
            check_draft(&t, &d),
            DraftVerdict::Incompatible { .. }
        ));
    }
}
