# Tokamak

**A reactor room for your local LLMs.**

Tokamak is a Windows desktop app for people who run large language models on their own hardware and want to actually see and control what is happening. It finds the GGUF models you already have, works out the best way to run each one on your GPU, launches them through llama.cpp, and turns the whole thing into a live instrument panel: VRAM, utilization, temperature, power, tokens per second, and KV cache pressure, all animated in real time.

A tokamak is the machine that holds a fusion reaction inside a magnetic ring. That is roughly the job here: contain a model that wants all of your VRAM, keep it stable, and get useful work out of it. The name also happens to start with "tok", which is the only unit anyone here cares about.

> Status: v0.1, Windows + NVIDIA first. Built with Tauri 2 (Rust) and React.

---

## Why this exists

Ollama and LM Studio are fine chat apps, but they treat your hardware as a black box:

- They will not tell you *why* a model runs at 4 tok/s instead of 190.
- They guess at GPU offload settings and never show you the math.
- They cannot tell you which quantization of a model you *should have downloaded* for your GPU.
- Their idea of telemetry is a spinner.

Tokamak is built around three ideas:

1. **No lock-in.** It reads the model caches you already have (Hugging Face, LM Studio, any folder you add). Plain GGUF files, no proprietary blob store, no re-downloading. It even drives the llama.cpp server binaries LM Studio already installed.
2. **Measure, don't guess.** The estimator predicts what fits, and the benchmark then *actually runs the model* and reports real numbers from your GPU. Recommendations are grounded in arithmetic you can inspect, then verified by measurement.
3. **Show everything.** If the KV cache is about to overflow, you should see it coming. If half the layers spilled to CPU, the decode-rate cliff should be visible, not mysterious.

---

## Features

### The Hangar: your model library
- Scans the standard Hugging Face hub cache (`HF_HOME` respected), LM Studio's model folders, and any custom folders you add through the native folder picker. Added folders persist across restarts.
- Parses GGUF headers directly (v2 and v3): architecture, quantization, native context length, layer count, attention geometry, parameter count.
- Understands multi-file models: split shards (`-00001-of-00003`) collapse into a single entry, and multimodal projector files (`mmproj-*`) show up as a VISION badge on their parent model instead of cluttering the list.
- Every model card shows a live fit verdict for your GPU: FITS with the max context, partial offload with the exact layer split, or CPU ONLY.

### Hardware-aware auto-config
For each model, Tokamak computes the best launch configuration for *your* GPU:

- Weights cost is derived from the file size and the offloaded layer fraction.
- KV cache cost is computed from the model's real attention shape: `layers x context x kv_heads x (key_len + value_len) x 2 bytes`.
- A fixed allowance covers llama.cpp compute buffers, and 10 percent of VRAM is held back for the desktop and driver.

The result is either full offload at the largest context that fits, or the maximum number of GPU layers at a working context. The config panel shows the VRAM budget breakdown (weights / KV / overhead) and a context ladder telling you exactly which context sizes fit at full offload.

### The quant advisor
The answer to "which file should I download?". For the selected model, Tokamak evaluates the whole GGUF quantization ladder (F16 down to Q2_K, using effective bits per weight) against your VRAM, including KV cache at a practical context, and tells you the sweet spot:

> F16 won't fully fit, get Q4_K_M (4.2 GB headroom)

Each rung shows the estimated weight size and remaining headroom, with your current file marked.

### Ignition and live telemetry: the Control Rod panel
Press IGNITE and Tokamak launches `llama-server` with the recommended (or your chosen) settings, then renders the machine as a warm-graphite instrument panel where a single plasma-amber hue always means energy and load:

- **Rod bank:** VRAM as fuel rods, one rod per GB. While a model runs the rods segment into weights, KV cache in use, and KV reserved. Hovering any model in the library projects its estimated footprint as dashed ghost rods, so you can see whether it fits *before* launching.
- **Flux trace:** the last 60 seconds of decode rate, GPU utilization, and temperature as scrolling heat strips; brightness is the value.
- **KV containment alert:** at 90 percent cache fill the third strip flips to a red KV pressure trace, a banner drops into the console, and the rail's KV box pulses. Your context is nearly full and you can see it happening.
- **Decode headline:** live tok/s in 42px numerals, with prefill rate and total tokens.
- **Session timeline:** every turn of the conversation as a block sized by tokens, laid against the context ceiling.
- The rail also carries RAM, CPU, GPU temperature, power draw against its limit, and core/memory clocks, polled at 1 Hz.

### Agent mode
Arm AGENT in the console and grant a workspace folder, and the running model gets Claude Code style abilities inside that folder:

- `list_dir` and `read_file` run automatically so the model can explore the project.
- `write_file` and `run_command` (PowerShell) stop at an approval card; nothing destructive happens without your click.
- Every tool is sandboxed to the workspace you granted. Path escapes (`..`, absolute paths, symlink tricks) are rejected in the Rust backend, not just hidden in the UI.
- The loop continues until the model answers without requesting a tool, with a hard cap per task so a confused model cannot spin forever.

### Console comforts
- Model output renders as markdown: headers, bold, lists, tables, fenced code with language tags. Rendering builds React elements directly, never raw HTML, so model output cannot inject anything into the app.
- Reasoning models stream their thinking into a collapsible section that folds away once the answer starts.
- UI scale control in the status bar, plus Ctrl+= / Ctrl+- / Ctrl+0 and Ctrl+scroll, persisted across restarts.

Binary resolution is automatic: LM Studio's bundled llama.cpp builds are discovered and ranked (CUDA 12 above CUDA above Vulkan above CPU), with `llama-server` on your PATH as a fallback. You can pin a specific binary in the header. The CUDA runtime DLLs that LM Studio keeps in a separate vendor folder are injected into the child process search path automatically.

### Measured benchmarks
Estimates are predictions; BENCH is proof. For any model, Tokamak launches real server instances on a dedicated port, loads the model at candidate configs (for example full offload versus a third of the layers), generates a fixed 96 token workload with `ignore_eos`, and reports:

- real prefill and decode tok/s, computed from token counts and elapsed time (llama.cpp's own per-second fields can overflow on near-zero timings, so Tokamak does the division itself),
- load time,
- peak VRAM, sampled from NVML at 10 Hz during the run.

On an RTX 5080 Laptop GPU this is what surfaced the offload cliff: the same 4B model at 137.8 tok/s fully offloaded versus 18.8 tok/s at partial offload. That cliff is exactly what the estimator exists to protect you from.

### The benchmark suite
SUITE benches *every* model in your hangar at its recommended config, one after another, and renders a comparative bar chart in the rail as results stream in. Models that will not fit are skipped with the reason shown. One click exports a Markdown report to `Documents\tokamak`, complete with a "vs best" column, ready to paste into a gist or a Reddit argument.

### The Console
A terminal-style chat drawer wired straight to the running server through the Rust backend (no CORS, no browser tab):

- Streaming output with a live token count and measured decode rate per reply.
- Reasoning models are handled properly: thinking deltas (`reasoning_content`) render dimmed and separate from the final answer.
- Sampler controls per message: temperature, top-k, top-p, min-p, max tokens, and a system prompt.
- STOP cancels generation mid-stream.
- The header shows the OpenAI-compatible endpoint (`http://127.0.0.1:8137/v1`) with a copy button, so you can point any other client at the same server. Note that the server root URL intentionally serves no web page; the API lives under `/v1`, and this console is the UI.

---

## Getting started

### Prerequisites
- Windows 10/11 with an NVIDIA GPU (telemetry and benchmarking use NVML).
- Rust (stable) and Node.js 20+.
- A `llama-server` binary. If LM Studio is installed, Tokamak finds its bundled builds automatically. Otherwise put llama.cpp's `llama-server` on your PATH.
- Some GGUF models on disk (LM Studio cache, HF cache, or any folder).

### Run in development
```
git clone https://github.com/360NoScopeGuru/tokamak
cd tokamak
npm install
npm run tauri dev
```

### Build a release
```
npm run tauri build
```

### Tests
```
cd src-tauri
cargo test                            # unit tests
cargo test -- --ignored --nocapture   # hardware integration tests (launch real models)
```

---

## Architecture

```
src-tauri/src/
  gguf.rs        GGUF v2/v3 header parser (metadata KV block, quant labels,
                 attention geometry, split-file fields)
  scanner.rs     cache discovery: HF hub, LM Studio, user folders; shard and
                 mmproj detection
  telemetry.rs   NVML GPU metrics + sysinfo RAM/CPU in long-lived managed state
  estimator.rs   VRAM fit arithmetic, context ladder, quant advisor
  llama.rs       llama-server process manager: binary discovery and ranking,
                 vendor DLL path injection, health probing, Prometheus /metrics
                 parsing, kill-on-drop process hygiene
  benchmark.rs   measured benchmark runner + Markdown report export
  chat.rs        SSE streaming chat client on a worker thread, reasoning aware
  tools.rs       agent tools (list/read/write/run), sandboxed to the workspace
  settings.rs    persisted JSON settings (folders, binary, UI scale, workspace)
  lib.rs         Tauri commands wiring it all together

src/
  App.tsx        orchestration, polling, launch/bench/suite flows, UI scaling
  Library.tsx    fuel library with fit verdicts
  Rail.tsx       telemetry stack: flux trace, rod bank, vitals, KV alert
  Flux.tsx       the 60 second canvas heat strips
  Dock.tsx       containment budget, context ladder, quant advisor, bench detail
  Console.tsx    streaming chat, markdown rendering, agent loop + approvals
  Markdown.tsx   safe markdown to React renderer for model output
  styles.css     "Control Rod" design system
```

Design notes:

- **One server at a time** in v1. Starting a model replaces the previous one; benchmarks run on their own port (8139) and never touch your session on 8137.
- **Process hygiene matters.** The server manager kills its child on drop, so a crash, a panicking test, or closing the app never leaves an orphaned `llama-server` squatting on your VRAM.
- **All HTTP lives in Rust.** The webview never talks to the model server directly, which avoids CORS entirely and keeps one code path for streaming, health, and metrics.

## Roadmap

- Sampler presets and per-message settings provenance
- Persistent chat transcripts
- Model downloads (grab the advisor's recommended quant straight from Hugging Face)
- Speculative decoding setup with a live accept-rate display
- KV cache quantization as a first-class toggle
- Multi-GPU and tensor-parallel backends (vLLM / ExLlamaV2) behind the same cockpit

## License

Not yet chosen (all rights reserved while in early development).
