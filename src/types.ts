/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

// Shared frontend types mirroring the Rust commands' serde output (snake_case).

export interface GgufMetadata {
  version: number;
  tensor_count: number;
  architecture: string | null;
  name: string | null;
  quant_label: string | null;
  context_length: number | null;
  block_count: number | null;
  embedding_length: number | null;
  parameter_count: number | null;
  size_label: string | null;
  split_count: number | null;
}

export interface ModelEntry {
  path: string;
  file_name: string;
  size_bytes: number;
  source: string;
  /// Set for Ollama models, whose on-disk name is a content hash — this is the
  /// `name:tag` to show instead.
  display_name: string | null;
  is_shard_continuation: boolean;
  shard_total: number | null;
  is_mmproj: boolean;
  metadata: GgufMetadata | null;
  parse_error: string | null;
  /// Set when the header parsed but stock llama.cpp will refuse to load the
  /// file. Readable, not runnable — distinct from `parse_error`.
  load_blocker: string | null;
}

export interface ScanRoot {
  path: string;
  source: string;
  exists: boolean;
}

export interface GpuSnapshot {
  index: number;
  name: string;
  vram_used_bytes: number;
  vram_total_bytes: number;
  gpu_util_pct: number;
  mem_util_pct: number;
  temperature_c: number | null;
  power_watts: number | null;
  power_limit_watts: number | null;
  clock_graphics_mhz: number | null;
  clock_mem_mhz: number | null;
  fan_pct: number | null;
}

export interface TelemetrySnapshot {
  nvml_available: boolean;
  error: string | null;
  gpus: GpuSnapshot[];
  ram_used_bytes: number;
  ram_total_bytes: number;
  cpu_util_pct: number;
  timestamp_ms: number;
}

export interface InferenceMetrics {
  prompt_tokens_total: number;
  predicted_tokens_total: number;
  prompt_tokens_per_sec: number;
  predicted_tokens_per_sec: number;
  kv_cache_usage_ratio: number;
  kv_cache_tokens: number;
  requests_processing: number;
}

export interface ServerStatus {
  running: boolean;
  health: string;
  pid: number | null;
  base_url: string | null;
  model_path: string | null;
  binary_label: string | null;
  uptime_ms: number | null;
  error: string | null;
}

export interface ContextOption {
  ctx: number;
  est_total_bytes: number;
  /// All layers fit on the GPU at this context.
  fits: boolean;
  /// Layers that fit at this context — non-zero rungs are usable even when
  /// `fits` is false (partial offload).
  n_gpu_layers: number;
}

export interface QuantOption {
  label: string;
  est_weights_bytes: number;
  headroom_bytes: number;
  fits: boolean;
  is_current: boolean;
}

export interface QuantAdvice {
  est_params_b: number;
  current_label: string | null;
  current_fits: boolean;
  recommended: string | null;
  options: QuantOption[];
}

export interface VramEstimate {
  fits: boolean;
  full_offload: boolean;
  n_gpu_layers: number;
  ctx_size: number;
  est_weights_bytes: number;
  est_kv_bytes: number;
  est_overhead_bytes: number;
  est_total_bytes: number;
  budget_bytes: number;
  gpu_total_bytes: number;
  gpu_free_bytes: number;
  context_options: ContextOption[];
  quant_advice: QuantAdvice | null;
  notes: string[];
}

export interface SuiteRow {
  model: string;
  quant: string | null;
  n_gpu_layers: number;
  ctx_size: number;
  load_ms: number;
  prefill_tok_s: number;
  decode_tok_s: number;
  peak_vram_bytes: number;
  skipped: string | null;
}

/// What speculative decoding actually bought, measured by running the same
/// config twice. `speedup` below 1.0 means the draft made generation slower,
/// which is a common and reportable outcome.
export interface SpecResult {
  draft_n: number;
  draft_n_accepted: number;
  accept_rate: number;
  baseline_decode_tok_s: number;
  decode_tok_s: number;
  speedup: number;
}

/// Live speculative-decoding counters for the generation in flight, pushed
/// from `chat.rs` as `spec-progress` events. Cumulative per generation, not
/// per delta. `draft_n` stays 0 when no draft model is loaded, and also when
/// the server is too old to report the counters at all — the cockpit tells
/// those two apart from the running server's config, not from this.
export interface SpecProgress {
  id: number;
  draft_n: number;
  draft_n_accepted: number;
  accept_rate: number;
}

export interface BenchResult {
  n_gpu_layers: number;
  ctx_size: number;
  loaded: boolean;
  load_ms: number;
  prefill_tok_s: number;
  decode_tok_s: number;
  peak_vram_bytes: number;
  speculative: SpecResult | null;
  error: string | null;
}

export interface LlamaBinary {
  path: string;
  label: string;
  backend: string;
  source: string;
  rank: number;
}

export interface Settings {
  extra_model_dirs: string[];
  preferred_binary: string | null;
  ui_scale?: number | null;
  agent_workspace?: string | null;
  /// KV cache element type: "f16" | "q8_0" | "q4_0".
  kv_cache_type?: string | null;
}

// ---- chat history (mirrors history.rs) ----

export interface SamplerSnap {
  temperature?: number | null;
  top_k?: number | null;
  top_p?: number | null;
  min_p?: number | null;
  max_tokens?: number | null;
  system?: string | null;
}

export interface StoredTurn {
  /// Stable node id. Absent only in files written before Rev G; the backend
  /// fills it on load, so the frontend never sees one missing.
  id?: string | null;
  /// The turn this one follows. `null` marks a root.
  parent?: string | null;
  role: string;
  kind?: string | null;
  tool_name?: string | null;
  content: string;
  thinking?: string | null;
  tokens?: number | null;
  decode_tok_s?: number | null;
  stopped?: boolean | null;
  /// finish_reason as recorded at generation time. Absent on turns saved before
  /// this was tracked — absence is "unknown", never "was truncated".
  finish?: string | null;
  error?: boolean | null;
  timestamp_ms: number;
  sampler?: SamplerSnap | null;
}

export interface StoredSession {
  id: string;
  kind: string;
  title: string;
  model_name?: string | null;
  model_path?: string | null;
  binary_label?: string | null;
  n_gpu_layers?: number | null;
  ctx_size?: number | null;
  workspace?: string | null;
  created_ms: number;
  updated_ms: number;
  /// Leaf of the branch last selected. `null` resolves to the last turn.
  head?: string | null;
  turns: StoredTurn[];
}

export interface SessionMeta {
  id: string;
  kind: string;
  title: string;
  model_name: string | null;
  n_gpu_layers: number | null;
  ctx_size: number | null;
  workspace: string | null;
  created_ms: number;
  updated_ms: number;
  /// Turns on the active branch, not the size of the pool.
  turn_count: number;
  /// Leaves in the pool. 1 means the session never forked.
  branch_count: number;
  /// Every token generated, abandoned branches included.
  total_tokens: number;
  avg_decode_tok_s: number;
}

// ---- formatting helpers used across components ----

/// Bytes as GiB, which is what GPU vendors, drivers and everyone else means by
/// "GB" for VRAM. NVML reports a 16 GB card as 16303 MiB = 15.92 GiB; dividing
/// by 1e9 turns that into "17.1", so every card looked ~7% larger than it is.
export const BYTES_PER_GB = 1024 * 1024 * 1024;

export function gb(bytes: number, digits = 1): string {
  return `${(bytes / BYTES_PER_GB).toFixed(digits)}`;
}

export function ctxLabel(n: number | null): string {
  if (!n) return "—";
  return n >= 1024 ? `${Math.round(n / 1024)}K` : String(n);
}

export function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

export function dirOf(path: string): string {
  return path.slice(0, Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/")));
}

export function modelLabel(m: ModelEntry): string {
  // Ollama models are shown by their `name:tag` so the library matches what
  // `ollama list` says; the GGUF's internal name would not.
  if (m.display_name) return m.display_name;
  return (m.metadata?.name ?? m.file_name).replace(/\.gguf$/i, "");
}

/// Why a model can or cannot draft for a target. Mirrors `DraftVerdict` in
/// speculative.rs: `unverifiable` means the metadata does not say, which is
/// deliberately distinct from "this works".
export type DraftVerdict =
  | { kind: "compatible" }
  | { kind: "incompatible"; reasons: string[] }
  | { kind: "unverifiable"; reasons: string[] };

/// Whether the draft is cheap enough relative to the target to be worth
/// running. This is about *fit and cost*, never a promise of speedup — only a
/// benchmark can tell you that, and it frequently says no.
export type DraftEconomics = "recommended" | "marginal" | "counterproductive" | "unknown";

export type PairVerdict = "fits" | "costs_target_layers" | "too_big";

export interface PairEstimate {
  verdict: PairVerdict;
  ctx_size: number;
  target_layers_on_gpu: number;
  target_layers_total: number;
  /// Target layers pushed off the GPU to make room. The real price.
  target_layers_evicted: number;
  draft_layers_on_gpu: number;
  draft_layers_total: number;
  est_target_bytes: number;
  est_draft_bytes: number;
  est_total_bytes: number;
  budget_bytes: number;
  notes: string[];
}

export interface DraftCandidate {
  path: string;
  label: string;
  size_bytes: number;
  parameter_count: number | null;
  quant_label: string | null;
  verdict: DraftVerdict;
  active_params: number | null;
  /// Draft cost as a fraction of the target's, by active parameters.
  cost_ratio: number | null;
  economics: DraftEconomics;
  pair: PairEstimate | null;
}
