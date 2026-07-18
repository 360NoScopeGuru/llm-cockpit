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
  is_shard_continuation: boolean;
  shard_total: number | null;
  is_mmproj: boolean;
  metadata: GgufMetadata | null;
  parse_error: string | null;
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
  fits: boolean;
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

export interface BenchResult {
  n_gpu_layers: number;
  ctx_size: number;
  loaded: boolean;
  load_ms: number;
  prefill_tok_s: number;
  decode_tok_s: number;
  peak_vram_bytes: number;
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
  role: string;
  kind?: string | null;
  tool_name?: string | null;
  content: string;
  thinking?: string | null;
  tokens?: number | null;
  decode_tok_s?: number | null;
  stopped?: boolean | null;
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
  turn_count: number;
  total_tokens: number;
  avg_decode_tok_s: number;
}

// ---- formatting helpers used across components ----

export function gb(bytes: number, digits = 1): string {
  return `${(bytes / 1e9).toFixed(digits)}`;
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
  return (m.metadata?.name ?? m.file_name).replace(/\.gguf$/i, "");
}
