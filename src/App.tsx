import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { TelemetryCockpit } from "./Telemetry";
import { ServerBar, ServerStatus } from "./ServerBar";
import { AutoConfigPanel, VramEstimate } from "./AutoConfig";
import { BenchmarkPanel, BenchResult } from "./Benchmark";
import "./App.css";

// Mirrors the serde output of the Rust `scanner`/`gguf` modules (snake_case).
interface GgufMetadata {
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

interface ModelEntry {
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

/** Directory portion of a model path (used to pair mmproj files with parents). */
function dirOf(path: string): string {
  return path.slice(0, Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/")));
}

interface ScanRoot {
  path: string;
  source: string;
  exists: boolean;
}

function formatBytes(n: number): string {
  if (n <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatCtx(n: number | null): string {
  if (!n) return "—";
  if (n >= 1024) return `${Math.round(n / 1024)}K`;
  return String(n);
}

function sourceLabel(source: string): string {
  switch (source) {
    case "huggingface":
      return "HF";
    case "lm-studio":
      return "LM Studio";
    default:
      return "Folder";
  }
}

function App() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [roots, setRoots] = useState<ScanRoot[]>([]);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [server, setServer] = useState<ServerStatus | null>(null);
  const [launching, setLaunching] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<
    { path: string; name: string; data: VramEstimate } | null
  >(null);
  const [estimating, setEstimating] = useState<string | null>(null);
  const [bench, setBench] = useState<{
    path: string;
    name: string;
    expected: number;
    results: BenchResult[];
  } | null>(null);
  const [benching, setBenching] = useState<string | null>(null);

  async function rescan() {
    setScanning(true);
    setError(null);
    try {
      const [rootList, modelList] = await Promise.all([
        invoke<ScanRoot[]>("scan_roots"),
        invoke<ModelEntry[]>("scan_models", { extraDirs: [] }),
      ]);
      setRoots(rootList);
      setModels(modelList);
    } catch (e) {
      setError(String(e));
    } finally {
      setScanning(false);
    }
  }

  async function launch(
    model: ModelEntry,
    opts?: { nGpuLayers?: number; ctxSize?: number }
  ) {
    setLaunching(model.path);
    try {
      const status = await invoke<ServerStatus>("llama_start", {
        config: {
          model_path: model.path,
          n_gpu_layers: opts?.nGpuLayers ?? 999,
          ctx_size: opts?.ctxSize ?? 4096,
          port: 8137,
        },
      });
      setServer(status);
      setEstimate(null);
    } catch (e) {
      setServer({
        running: false,
        health: "error",
        pid: null,
        base_url: null,
        model_path: model.path,
        binary_label: null,
        uptime_ms: null,
        error: String(e),
      });
    } finally {
      setLaunching(null);
    }
  }

  async function autoConfig(model: ModelEntry) {
    setEstimating(model.path);
    setError(null);
    try {
      const data = await invoke<VramEstimate>("estimate_config", {
        modelPath: model.path,
      });
      setEstimate({
        path: model.path,
        name: model.metadata?.name ?? model.file_name,
        data,
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setEstimating(null);
    }
  }

  async function stop() {
    try {
      await invoke("llama_stop");
    } catch {
      /* ignore */
    }
    setServer(await invoke<ServerStatus>("llama_status"));
  }

  async function benchmark(model: ModelEntry) {
    setBenching(model.path);
    setError(null);
    const name = model.metadata?.name ?? model.file_name;
    try {
      // Derive a 2-config sweep: the recommended config vs a reduced-offload one
      // to expose the GPU-offload speed cliff.
      const est = await invoke<VramEstimate>("estimate_config", {
        modelPath: model.path,
      });
      const rec = est.n_gpu_layers;
      const reduced = Math.max(1, Math.floor(rec / 3));
      const configs = [
        { n_gpu_layers: rec, ctx_size: est.ctx_size },
        { n_gpu_layers: reduced, ctx_size: est.ctx_size },
      ];
      setBench({ path: model.path, name, expected: configs.length, results: [] });

      const unlisten = await listen<BenchResult>("benchmark-progress", (e) => {
        setBench((prev) =>
          prev && prev.path === model.path
            ? { ...prev, results: [...prev.results, e.payload] }
            : prev
        );
      });
      const final = await invoke<BenchResult[]>("benchmark_model", {
        modelPath: model.path,
        configs,
      });
      unlisten();
      setBench((prev) =>
        prev && prev.path === model.path ? { ...prev, results: final } : prev
      );
    } catch (e) {
      setError(String(e));
      setBench(null);
    } finally {
      setBenching(null);
      setServer(await invoke<ServerStatus>("llama_status"));
    }
  }

  async function addFolder() {
    try {
      const dir = await open({ directory: true, title: "Add a model folder" });
      if (typeof dir !== "string" || !dir) return;
      await invoke("add_model_dir", { dir });
      await rescan();
    } catch (e) {
      setError(String(e));
    }
  }

  async function removeFolder(dir: string) {
    try {
      await invoke("remove_model_dir", { dir });
      await rescan();
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    rescan();
  }, []);

  // Poll server status so health transitions (loading → ok) show live.
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await invoke<ServerStatus>("llama_status");
        if (alive) setServer(s);
      } catch {
        /* ignore */
      }
    };
    poll();
    const id = setInterval(poll, 1500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // Primary list hides continuation shards and mmproj companion files; the
  // latter surface as a "vision" badge on models sharing their directory.
  const primary = models.filter((m) => !m.is_shard_continuation && !m.is_mmproj);
  const visionDirs = new Set(
    models.filter((m) => m.is_mmproj).map((m) => dirOf(m.path))
  );

  return (
    <main className="app">
      <TelemetryCockpit />

      <ServerBar status={server} onStop={stop} />

      {estimate && (
        <AutoConfigPanel
          name={estimate.name}
          estimate={estimate.data}
          launching={launching === estimate.path}
          onLaunch={(ngl, ctx) => {
            const model = models.find((m) => m.path === estimate.path);
            if (model) launch(model, { nGpuLayers: ngl, ctxSize: ctx });
          }}
          onDismiss={() => setEstimate(null)}
        />
      )}

      {bench && (
        <BenchmarkPanel
          name={bench.name}
          results={bench.results}
          running={benching === bench.path}
          expected={bench.expected}
          onDismiss={() => setBench(null)}
        />
      )}

      <header className="app-header">
        <div>
          <h1>Model Library</h1>
          <p className="subtitle">
            {scanning
              ? "Scanning caches…"
              : `${primary.length} model${primary.length === 1 ? "" : "s"} found`}
          </p>
        </div>
        <button onClick={rescan} disabled={scanning}>
          {scanning ? "Scanning…" : "Rescan"}
        </button>
      </header>

      <section className="roots">
        {roots.map((r) => (
          <span key={r.path} className={`root-chip ${r.exists ? "" : "missing"}`}>
            {sourceLabel(r.source)}
            <code>{r.path}</code>
            {!r.exists && <em>not found</em>}
            {r.source === "folder" && (
              <button
                className="chip-remove"
                title="Remove this folder"
                onClick={() => removeFolder(r.path)}
              >
                ✕
              </button>
            )}
          </span>
        ))}
        <button className="add-folder" onClick={addFolder}>
          + Add folder
        </button>
      </section>

      {error && <div className="error">Scan failed: {error}</div>}

      {!scanning && primary.length === 0 && !error && (
        <div className="empty">
          No GGUF models found in the default caches. Point the app at a folder to
          get started.
        </div>
      )}

      {primary.length > 0 && (
        <table className="model-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Arch</th>
              <th>Quant</th>
              <th>Ctx</th>
              <th>Params</th>
              <th>Size</th>
              <th>Source</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {primary.map((m) => {
              const md = m.metadata;
              return (
                <tr key={m.path} title={m.path}>
                  <td className="model-name">
                    {md?.name ?? m.file_name}
                    {m.shard_total && (
                      <span className="badge">{m.shard_total} shards</span>
                    )}
                    {visionDirs.has(dirOf(m.path)) && (
                      <span className="badge vision" title="Has a multimodal projector (mmproj) companion file">
                        vision
                      </span>
                    )}
                    {m.parse_error && (
                      <span className="badge warn" title={m.parse_error}>
                        unreadable
                      </span>
                    )}
                  </td>
                  <td>{md?.architecture ?? "—"}</td>
                  <td>{md?.quant_label ?? "—"}</td>
                  <td>{formatCtx(md?.context_length ?? null)}</td>
                  <td>{md?.size_label ?? "—"}</td>
                  <td>{formatBytes(m.size_bytes)}</td>
                  <td>
                    <span className="src">{sourceLabel(m.source)}</span>
                  </td>
                  <td>
                    {server?.running && server.model_path === m.path ? (
                      <span className="src">running</span>
                    ) : (
                      <div className="row-actions">
                        <button
                          className="auto-btn"
                          disabled={estimating !== null || !!m.parse_error}
                          onClick={() => autoConfig(m)}
                          title="Estimate optimal config for your GPU"
                        >
                          {estimating === m.path ? "…" : "⚙ Auto"}
                        </button>
                        <button
                          className="bench-btn"
                          disabled={benching !== null || !!m.parse_error}
                          onClick={() => benchmark(m)}
                          title="Measure real tok/s across configs on your GPU"
                        >
                          {benching === m.path ? "Benchmarking…" : "Bench"}
                        </button>
                        <button
                          className="launch-btn"
                          disabled={launching !== null || !!m.parse_error}
                          onClick={() => launch(m)}
                        >
                          {launching === m.path ? "Launching…" : "Launch"}
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </main>
  );
}

export default App;
