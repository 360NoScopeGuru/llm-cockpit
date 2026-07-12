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
   Pareto-optimal quant/context/GPU-layer config for your GPU. *(tier 1 done:
   VRAM-fit estimator; measured benchmark pending)*
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
- `scanner.rs` — cache scanner (HF / LM Studio / folders) with shard grouping.
  Commands: `scan_models`, `scan_roots`. Verified against real local models.
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

Frontend (`src/`):
- `Telemetry.tsx` — cockpit panel polling `gpu_telemetry` + `inference_metrics`
  once a second: color-coded GPU/system meters plus an Inference tile (decode/
  prefill tok/s, KV-cache) that appears while a model runs.
- `ServerBar.tsx` — server status bar (health dot, model, binary, base URL, Stop)
  polling `llama_status`.
- `AutoConfig.tsx` — recommendation panel (full/partial offload, layers, context)
  with a stacked VRAM-breakdown bar and a launch-with-this-config button.
- `App.tsx` — model library view (arch, quant, context, size, source) with
  per-model Auto-config + Launch buttons.

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
