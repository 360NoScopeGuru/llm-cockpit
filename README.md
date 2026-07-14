# llm-cockpit (working name)

A power-user desktop app for running local LLMs — built to actually exploit your
hardware and your downloaded models, not just wrap a chat box around them.

## Why

The incumbents (Ollama, LM Studio, GPT4All, …) stop at "chat window + model
downloader." This project's wedge is the **inference cockpit**: point it at the
models you already have, and it tells you the optimal way to run each one on *your*
hardware, then runs it under live telemetry.

## v1 pillars

1. **No-lock-in model library** — scan the HF cache, LM Studio, and any folders you
   add; read GGUF metadata directly (arch, quant, context length) with zero
   re-downloading. *(done: parser + scanner + library view)*
2. **Hardware-aware auto-config + benchmarking** — estimate then *measure* the
   Pareto-optimal quant/context/GPU-layer config for your GPU. *(done: VRAM-fit
   estimator + measured benchmark that launches candidate configs and records
   real tok/s + peak VRAM)*
3. **Live telemetry cockpit** — VRAM/util/temp/power + tokens/sec, prefill-vs-decode
   split, KV-cache occupancy while generating. *(done: GPU + system telemetry +
   inference-side metrics from llama-server)*

## Stack

- **Tauri 2** (Rust core + web frontend) — small footprint, native GPU/OS hooks.
- **React + TypeScript + Vite** frontend.
- Inference engine: **llama.cpp** (`llama-server`), behind a swappable backend
  abstraction so ExLlamaV2/vLLM can slot in later for multi-GPU tensor-parallel.
- v1 target: **Windows + NVIDIA** first.

## Current state

Rust backend (`src-tauri/src/`):
- `gguf.rs` — GGUF v2/v3 metadata parser (header + KV block only; skips large
  arrays in-place). Unit-tested against synthetic files.
- `scanner.rs` — cache scanner (HF / LM Studio / folders) with shard grouping and
  mmproj (multimodal projector) detection. Commands: `scan_models`, `scan_roots`.
  Verified against real local models.
- `settings.rs` — persisted settings (JSON in the OS config dir): extra model
  folders (merged into every scan) and the preferred llama-server binary.
  Commands: `add_model_dir`, `remove_model_dir`, `get_settings`,
  `set_preferred_binary`.
- `chat.rs` — built-in chat backend. Streams from the running server's
  OpenAI-compatible `/v1/chat/completions` (SSE) on a worker thread, emitting
  `chat-delta`/`chat-done` events with a measured decode rate. Commands:
  `chat_send`, `chat_cancel`. (The server's root URL intentionally serves no web
  page — LM Studio's llama-server build is API-only, so `GET /` is a JSON 404;
  the console is the UI, and the API base for external clients is `<url>/v1`.)
- `telemetry.rs` — live GPU telemetry via NVML (VRAM, util, temp, power, clocks)
  plus system RAM/CPU via sysinfo, held in Tauri managed state. Command:
  `gpu_telemetry`. Verified against a real RTX 5080.
- `llama.rs` — `llama-server` process manager. Resolves a llama.cpp binary
  (prefers CUDA; discovers LM Studio's bundled builds and injects their sibling
  `vendor/` DLL dirs into the child PATH), launches a model with configurable
  GPU layers / context, and tracks lifecycle + `/health`. Commands:
  `llama_binaries`, `llama_start`, `llama_stop`, `llama_status`, and
  `inference_metrics` (scrapes llama-server's Prometheus `/metrics` for decode/
  prefill tok/s + KV-cache usage). Verified by launching a real 4B model on the
  RTX 5080 and reading back live tok/s (load → generate → metrics → stop).
- `estimator.rs` — hardware-aware auto-config (tier 1). From a model's GGUF shape
  + the GPU's VRAM, computes the max GPU-offload layers + context that fit
  (weights + KV-cache + overhead vs a headroom budget). Command: `estimate_config`.
  Verified against real models on the RTX 5080.
- `benchmark.rs` — measured benchmark (the moat). Launches each candidate config
  for real on a dedicated port, generates a fixed token count (`ignore_eos`), and
  records real prefill/decode tok/s (token-count / time) + peak VRAM (NVML sampled
  during the run). Emits per-config progress. Command: `benchmark_model`. Verified
  on the RTX 5080 (full offload 193 tok/s vs partial 24 tok/s decode — an 8× cliff).

Frontend (`src/`) — **"The Core"**, a control-room instrument panel:
- `Core.tsx` — the centerpiece: a live canvas reactor rendering VRAM as a
  segmented outer ring (teal→violet), GPU utilization as a heat-tinted sweep,
  KV-cache as an amber band, and token generation as orbiting particles.
  Hovering a model in the hangar projects its estimated footprint onto the live
  ring as a violet ghost arc — you see whether it fits before launching.
  Launching is an "ignition" sequence; the center readout shows live decode
  tok/s while generating.
- `Hangar.tsx` — left bay: your models as craft cards with live fit verdicts
  (FITS / partial / CPU-only), vision + shard tags, IGNITE and BENCH actions,
  and the scan-source chips (add/remove folders via the native picker).
- `Rail.tsx` — right bay: system vitals (RAM/CPU/temp/power/clocks), plus a
  contextual panel — live bench results, or the selected model's config detail
  (recommendation, VRAM budget breakdown, full context ladder).
- `Console.tsx` — bottom drawer: terminal-style streaming chat with sampler
  controls (temp/top-k/top-p/min-p/max/system prompt), per-reply token counts
  and measured tok/s, STOP mid-generation, and a copyable OpenAI-compatible API
  chip (`<base>/v1`) for external clients.
- `App.tsx` — shell + orchestration; `styles.css` — the design system;
  `types.ts` — shared command types.

## Develop

```powershell
npm install
npm run tauri dev
```

Requires Rust (MSVC toolchain), Node, and WebView2. The hardware-dependent tests
are ignored by default:

```powershell
cd src-tauri
cargo test -- --ignored --nocapture   # real cache scan + live GPU snapshot
```
