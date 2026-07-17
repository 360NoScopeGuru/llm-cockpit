import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { Core, CoreGhost } from "./Core";
import { Hangar } from "./Hangar";
import { Rail } from "./Rail";
import { Console } from "./Console";
import {
  BenchResult,
  InferenceMetrics,
  LlamaBinary,
  ModelEntry,
  ScanRoot,
  ServerStatus,
  Settings,
  SuiteRow,
  TelemetrySnapshot,
  VramEstimate,
  baseName,
  dirOf,
  modelLabel,
} from "./types";
import "./styles.css";

const PORT = 8137;

export default function App() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [roots, setRoots] = useState<ScanRoot[]>([]);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot | null>(null);
  const [metrics, setMetrics] = useState<InferenceMetrics | null>(null);
  const [server, setServer] = useState<ServerStatus | null>(null);
  const [estimates, setEstimates] = useState<Map<string, VramEstimate>>(new Map());
  const [hoverPath, setHoverPath] = useState<string | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const [bench, setBench] = useState<{
    path: string;
    name: string;
    expected: number;
    results: BenchResult[];
  } | null>(null);
  const [benching, setBenching] = useState(false);
  const [suite, setSuite] = useState<{
    running: boolean;
    current: string | null;
    total: number;
    rows: SuiteRow[];
    exportPath: string | null;
  } | null>(null);
  const [binaries, setBinaries] = useState<LlamaBinary[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);

  const estimatesRef = useRef(estimates);
  estimatesRef.current = estimates;

  // ---- scanning + estimates ----

  async function rescan() {
    setScanning(true);
    try {
      const [rootList, modelList] = await Promise.all([
        invoke<ScanRoot[]>("scan_roots"),
        invoke<ModelEntry[]>("scan_models", { extraDirs: [] }),
      ]);
      setRoots(rootList);
      setModels(modelList);
      // Prime fit estimates for every primary model (cheap: header + NVML).
      for (const m of modelList) {
        if (m.is_shard_continuation || m.is_mmproj || m.parse_error) continue;
        if (estimatesRef.current.has(m.path)) continue;
        invoke<VramEstimate>("estimate_config", { modelPath: m.path })
          .then((est) =>
            setEstimates((prev) => new Map(prev).set(m.path, est))
          )
          .catch(() => {});
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setScanning(false);
    }
  }

  useEffect(() => {
    rescan();
    invoke<LlamaBinary[]>("llama_binaries").then(setBinaries).catch(() => {});
    invoke<Settings>("get_settings").then(setSettings).catch(() => {});
  }, []);

  // WebView2 on Windows can lay the page out at physical-pixel width while
  // still rendering at the display's DPI scale, painting wider/taller than the
  // window and clipping the right/bottom of the UI. Self-correct by zooming so
  // rendered size == real client size (no-op on healthy setups; re-checked on
  // every resize).
  const zoomRef = useRef(1);
  useEffect(() => {
    let disposed = false;
    const correct = async () => {
      try {
        // Ground truth from Win32 GetClientRect — tao/WebView2 can agree on a
        // DPI belief that the real window contradicts.
        const truth = await invoke<[number, number] | null>("true_client_size");
        if (!truth) return;
        const ratio = truth[0] / (window.innerWidth * window.devicePixelRatio);
        if (!disposed && isFinite(ratio) && ratio > 0.3 && Math.abs(1 - ratio) > 0.02) {
          zoomRef.current *= ratio;
          await getCurrentWebview().setZoom(zoomRef.current);
        }
      } catch {
        /* not fatal */
      }
    };
    // The webview's DPI belief can settle (or flip) some time after load with
    // no resize event, so re-check on a short cadence at first, then hourly-ish
    // cheap: every 5s (a no-op comparison when healthy).
    correct();
    let ticks = 0;
    const iv = setInterval(() => {
      ticks += 1;
      correct();
      if (ticks > 6) {
        clearInterval(iv);
      }
    }, 1200);
    const slowIv = setInterval(correct, 5000);
    let unlisten: (() => void) | undefined;
    getCurrentWindow()
      .onResized(() => correct())
      .then((u) => (unlisten = u));
    return () => {
      disposed = true;
      clearInterval(iv);
      clearInterval(slowIv);
      unlisten?.();
    };
  }, []);

  // ---- polling ----

  useEffect(() => {
    let alive = true;
    const fast = setInterval(async () => {
      try {
        const [t, m] = await Promise.all([
          invoke<TelemetrySnapshot>("gpu_telemetry"),
          invoke<InferenceMetrics | null>("inference_metrics"),
        ]);
        if (alive) {
          setTelemetry(t);
          setMetrics(m);
        }
      } catch {
        /* transient */
      }
    }, 1000);
    const slow = setInterval(async () => {
      try {
        const s = await invoke<ServerStatus>("llama_status");
        if (alive) setServer(s);
      } catch {
        /* transient */
      }
    }, 1500);
    return () => {
      alive = false;
      clearInterval(fast);
      clearInterval(slow);
    };
  }, []);

  // ---- launch / stop ----

  async function ignite(m: ModelEntry, ngl?: number, ctx?: number) {
    setLaunching(true);
    setError(null);
    try {
      let est = estimates.get(m.path);
      if (!est && ngl === undefined) {
        est = await invoke<VramEstimate>("estimate_config", { modelPath: m.path });
        setEstimates((prev) => new Map(prev).set(m.path, est!));
      }
      const status = await invoke<ServerStatus>("llama_start", {
        config: {
          model_path: m.path,
          n_gpu_layers: ngl ?? est?.n_gpu_layers ?? 999,
          ctx_size: ctx ?? est?.ctx_size ?? 4096,
          port: PORT,
        },
      });
      setServer(status);
    } catch (e) {
      setError(String(e));
    } finally {
      setLaunching(false);
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

  // ---- benchmark ----

  async function runBench(m: ModelEntry) {
    setBenching(true);
    setError(null);
    const name = modelLabel(m);
    try {
      let est = estimates.get(m.path);
      if (!est) {
        est = await invoke<VramEstimate>("estimate_config", { modelPath: m.path });
      }
      const reduced = Math.max(1, Math.floor(est.n_gpu_layers / 3));
      const configs = [
        { n_gpu_layers: est.n_gpu_layers, ctx_size: est.ctx_size },
        { n_gpu_layers: reduced, ctx_size: est.ctx_size },
      ];
      setBench({ path: m.path, name, expected: configs.length, results: [] });
      const unlisten = await listen<BenchResult>("benchmark-progress", (e) => {
        setBench((prev) =>
          prev && prev.path === m.path
            ? { ...prev, results: [...prev.results, e.payload] }
            : prev
        );
      });
      try {
        const final = await invoke<BenchResult[]>("benchmark_model", {
          modelPath: m.path,
          configs,
        });
        setBench((prev) => (prev && prev.path === m.path ? { ...prev, results: final } : prev));
      } finally {
        unlisten();
      }
    } catch (e) {
      setError(String(e));
      setBench(null);
    } finally {
      setBenching(false);
      setServer(await invoke<ServerStatus>("llama_status"));
    }
  }

  // ---- benchmark suite (all models, recommended config each) ----

  async function runSuite() {
    const eligible = primary.filter((m) => !m.parse_error);
    if (eligible.length === 0) return;
    setBenching(true);
    setBench(null);
    setSuite({ running: true, current: null, total: eligible.length, rows: [], exportPath: null });
    try {
      for (const m of eligible) {
        const name = modelLabel(m);
        setSuite((prev) => (prev ? { ...prev, current: name } : prev));
        let est = estimates.get(m.path);
        if (!est) {
          try {
            est = await invoke<VramEstimate>("estimate_config", { modelPath: m.path });
            setEstimates((prev) => new Map(prev).set(m.path, est!));
          } catch {
            /* fall through to skip */
          }
        }
        const push = (row: SuiteRow) =>
          setSuite((prev) => (prev ? { ...prev, rows: [...prev.rows, row] } : prev));
        if (!est || !est.fits) {
          push({
            model: name,
            quant: m.metadata?.quant_label ?? null,
            n_gpu_layers: 0,
            ctx_size: 0,
            load_ms: 0,
            prefill_tok_s: 0,
            decode_tok_s: 0,
            peak_vram_bytes: 0,
            skipped: est ? "won't fit on GPU" : "no estimate",
          });
          continue;
        }
        try {
          const res = await invoke<BenchResult[]>("benchmark_model", {
            modelPath: m.path,
            configs: [{ n_gpu_layers: est.n_gpu_layers, ctx_size: est.ctx_size }],
          });
          const r = res[0];
          push({
            model: name,
            quant: m.metadata?.quant_label ?? null,
            n_gpu_layers: r?.n_gpu_layers ?? est.n_gpu_layers,
            ctx_size: r?.ctx_size ?? est.ctx_size,
            load_ms: r?.load_ms ?? 0,
            prefill_tok_s: r?.prefill_tok_s ?? 0,
            decode_tok_s: r?.decode_tok_s ?? 0,
            peak_vram_bytes: r?.peak_vram_bytes ?? 0,
            skipped: r?.loaded ? null : r?.error ?? "failed",
          });
        } catch (e) {
          push({
            model: name,
            quant: m.metadata?.quant_label ?? null,
            n_gpu_layers: 0,
            ctx_size: 0,
            load_ms: 0,
            prefill_tok_s: 0,
            decode_tok_s: 0,
            peak_vram_bytes: 0,
            skipped: String(e),
          });
        }
      }
    } finally {
      setSuite((prev) => (prev ? { ...prev, running: false, current: null } : prev));
      setBenching(false);
      setServer(await invoke<ServerStatus>("llama_status"));
    }
  }

  async function exportSuite() {
    if (!suite) return;
    try {
      const rows = suite.rows
        .filter((r) => !r.skipped)
        .map(({ skipped, ...rest }) => rest);
      const path = await invoke<string>("export_bench_report", { rows });
      setSuite((prev) => (prev ? { ...prev, exportPath: path } : prev));
    } catch (e) {
      setError(String(e));
    }
  }

  // ---- folders / binary ----

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

  async function pickBinary(path: string) {
    try {
      const s = await invoke<Settings>("set_preferred_binary", {
        path: path === "" ? null : path,
      });
      setSettings(s);
    } catch (e) {
      setError(String(e));
    }
  }

  // ---- derived ----

  const primary = models.filter((m) => !m.is_shard_continuation && !m.is_mmproj);
  const visionDirs = new Set(models.filter((m) => m.is_mmproj).map((m) => dirOf(m.path)));
  const busy = launching || benching;
  const runningPath = server?.running ? server.model_path : null;
  const selected = primary.find((m) => m.path === selectedPath) ?? null;
  const hoverEst = hoverPath && hoverPath !== runningPath ? estimates.get(hoverPath) : null;
  const ghost: CoreGhost | null = hoverEst
    ? {
        bytes: hoverEst.est_weights_bytes + hoverEst.est_kv_bytes + hoverEst.est_overhead_bytes,
        fits: hoverEst.fits,
      }
    : null;
  const gpu = telemetry?.gpus[0] ?? null;
  const health = server?.running ? server.health : "stopped";
  const runningModel = primary.find((m) => m.path === runningPath);

  return (
    <div className="shell">
      <header className="hdr">
        <span className="wordmark">
          <span className="hex">⬢</span> TOKA<span className="dim">·</span>MAK
          <span className="tagline">local llm reactor</span>
        </span>
        <span className="hdr-spacer" />
        <span className="status-chip">
          <span className={`dot ${health === "ok" ? "ok" : health === "loading" || health === "starting" ? "loading" : health === "error" ? "error" : ""}`} />
          {server?.running ? (
            <>
              <span className="model-name">{baseName(server.model_path ?? "")}</span>
              <span className="sub">{server.binary_label}</span>
            </>
          ) : benching ? (
            <span className="sub">benchmark in progress…</span>
          ) : (
            <span className="sub">core idle</span>
          )}
        </span>
        <select
          value={settings?.preferred_binary ?? ""}
          onChange={(e) => pickBinary(e.target.value)}
          title="llama-server binary (applies to the next ignition)"
        >
          <option value="">auto · {binaries[0]?.label ?? "no binary found"}</option>
          {binaries.map((b) => (
            <option key={b.path} value={b.path}>
              {b.label}
            </option>
          ))}
        </select>
        {server?.running && (
          <button className="stop-btn" onClick={stop}>
            SHUTDOWN
          </button>
        )}
      </header>

      <div className="deck">
        <Hangar
          models={primary}
          visionDirs={visionDirs}
          roots={roots}
          scanning={scanning}
          estimates={estimates}
          runningPath={runningPath ?? null}
          busy={busy}
          selectedPath={selectedPath}
          onHover={setHoverPath}
          onSelect={(path) => setSelectedPath(path === selectedPath ? null : path)}
          onIgnite={(m) => ignite(m)}
          onBench={runBench}
          onSuite={runSuite}
          onRescan={rescan}
          onAddFolder={addFolder}
          onRemoveFolder={removeFolder}
        />

        <div className="stage">
          <Core
            gpu={gpu}
            infer={metrics}
            server={server}
            ghost={ghost}
            modelName={runningModel ? modelLabel(runningModel) : server?.model_path ? baseName(server.model_path) : null}
          />
          {server?.error && <div className="fault">{server.error}</div>}
        </div>

        <Rail
          telemetry={telemetry}
          metrics={metrics}
          selected={selected}
          estimate={selected ? estimates.get(selected.path) ?? null : null}
          bench={bench}
          benching={benching}
          suite={suite}
          busy={busy}
          onIgniteWith={(m, ngl, ctx) => ignite(m, ngl, ctx)}
          onCloseBench={() => setBench(null)}
          onCloseSuite={() => setSuite(null)}
          onExportSuite={exportSuite}
        />
      </div>

      <Console server={server} />

      {error && (
        <div className="toast-error">
          <button onClick={() => setError(null)}>✕</button>
          {error}
        </div>
      )}
    </div>
  );
}
