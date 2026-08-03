import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { Library } from "./Library";
import { Rail, Ghost } from "./Rail";
import { Dock } from "./Dock";
import { Console, StagedIgnite } from "./Console";
import { FluxSample } from "./Flux";
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
  ctxLabel,
  gb,
  modelLabel,
} from "./types";
import "./styles.css";

const PORT = 8137;
const FLUX_WINDOW = 60;

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
  const [liveCfg, setLiveCfg] = useState<{ ngl: number; ctx: number } | null>(null);
  const [history, setHistory] = useState<FluxSample[]>([]);
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
  const [scale, setScale] = useState(1);

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
          .then((est) => setEstimates((prev) => new Map(prev).set(m.path, est)))
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
    invoke<Settings>("get_settings")
      .then((s) => {
        setSettings(s);
        if (s.ui_scale && s.ui_scale >= 0.5 && s.ui_scale <= 2.5) setScale(s.ui_scale);
      })
      .catch(() => {});
  }, []);

  // ---- UI scaling (independent of the DPI corrector below: this is user
  // preference, applied as CSS zoom; the corrector fixes webview DPI bugs) ----

  const scaleSaveTimer = useRef<number | undefined>(undefined);
  useEffect(() => {
    (document.documentElement.style as CSSStyleDeclaration & { zoom: string }).zoom =
      String(scale);
    window.clearTimeout(scaleSaveTimer.current);
    scaleSaveTimer.current = window.setTimeout(() => {
      invoke("set_ui_scale", { scale }).catch(() => {});
    }, 600);
  }, [scale]);

  const bumpScale = (d: number) =>
    setScale((s) => Math.min(2, Math.max(0.7, Math.round((s + d) * 20) / 20)));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!e.ctrlKey) return;
      if (e.key === "=" || e.key === "+") {
        e.preventDefault();
        bumpScale(0.1);
      } else if (e.key === "-") {
        e.preventDefault();
        bumpScale(-0.1);
      } else if (e.key === "0") {
        e.preventDefault();
        setScale(1);
      }
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      bumpScale(e.deltaY < 0 ? 0.05 : -0.05);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("wheel", onWheel);
    };
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
    // no resize event, so re-check on a short cadence at first, then keep a
    // cheap 5s no-op comparison running.
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

  // ---- polling (telemetry + inference at 1 Hz feeds the flux trace) ----

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
          const g = t.gpus[0];
          setHistory((prev) => [
            ...prev.slice(-(FLUX_WINDOW - 1)),
            {
              decode: m?.predicted_tokens_per_sec ?? 0,
              util: g?.gpu_util_pct ?? 0,
              temp: g?.temperature_c ?? 0,
              kv: m?.kv_cache_usage_ratio ?? 0,
            },
          ]);
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
      const cfg = {
        model_path: m.path,
        n_gpu_layers: ngl ?? est?.n_gpu_layers ?? 999,
        ctx_size: ctx ?? est?.ctx_size ?? 4096,
        port: PORT,
      };
      const status = await invoke<ServerStatus>("llama_start", { config: cfg });
      setLiveCfg({ ngl: cfg.n_gpu_layers, ctx: cfg.ctx_size });
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
    setLiveCfg(null);
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
      setLiveCfg(null);
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
      setLiveCfg(null);
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

  async function pickWorkspace() {
    try {
      const dir = await open({ directory: true, title: "Grant the agent a workspace folder" });
      if (typeof dir !== "string" || !dir) return;
      const s = await invoke<Settings>("set_agent_workspace", { dir });
      setSettings(s);
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
  const selectedEst = selected ? estimates.get(selected.path) ?? null : null;
  const runningModel = primary.find((m) => m.path === runningPath) ?? null;
  const liveEst = runningPath ? estimates.get(runningPath) ?? null : null;
  const gpu = telemetry?.gpus[0] ?? null;
  const health = server?.running ? server.health : "stopped";
  const ready = !!server?.running && health === "ok";
  const igniting = !!server?.running && (health === "starting" || health === "loading");
  const generating = ready && (metrics?.requests_processing ?? 0) > 0;
  const kvAlert = ready && (metrics?.kv_cache_usage_ratio ?? 0) >= 0.9;

  const hoverModel =
    hoverPath && hoverPath !== runningPath ? primary.find((m) => m.path === hoverPath) : null;
  const hoverEst = hoverModel ? estimates.get(hoverModel.path) : null;
  const ghost: Ghost | null =
    hoverModel && hoverEst
      ? {
          name: modelLabel(hoverModel),
          bytes:
            hoverEst.est_weights_bytes + hoverEst.est_kv_bytes + hoverEst.est_overhead_bytes,
          fits: hoverEst.fits,
          layers: hoverModel.metadata?.block_count
            ? `${hoverEst.n_gpu_layers}/${hoverModel.metadata.block_count}`
            : null,
        }
      : null;

  const liveName = runningModel
    ? modelLabel(runningModel)
    : server?.model_path
      ? baseName(server.model_path)
      : null;

  const staged: StagedIgnite | null =
    !server?.running && !benching && selected && selectedEst && selectedEst.fits
      ? {
          name: modelLabel(selected),
          ngl: selectedEst.n_gpu_layers,
          layers: selected.metadata?.block_count ?? null,
          ctx: selectedEst.ctx_size,
          busy,
          onIgnite: () => ignite(selected, selectedEst.n_gpu_layers, selectedEst.ctx_size),
        }
      : null;

  // Header state line.
  let stateText: string;
  let stateCls = "";
  let lampCls = "";
  if (kvAlert) {
    stateText = `GENERATING · CONTAINMENT ${Math.round((metrics?.kv_cache_usage_ratio ?? 0) * 100)}%`;
    stateCls = "alert";
    lampCls = "alert";
  } else if (server?.running && health === "error") {
    stateText = "FAULT";
    stateCls = "fault";
    lampCls = "alert";
  } else if (igniting || launching) {
    stateText = liveName ? `IGNITING · ${liveName}` : "IGNITING";
    stateCls = "live";
    lampCls = "igniting";
  } else if (generating) {
    stateText = `GENERATING · ${liveName ?? ""}`;
    stateCls = "live";
    lampCls = "live";
  } else if (ready) {
    stateText = `REACTOR LIVE · ${liveName ?? ""}`;
    stateCls = "live";
    lampCls = "live";
  } else if (benching) {
    stateText = suite ? "COLD · SUITE RUNNING" : "COLD · BENCH RUNNING";
    lampCls = "igniting";
  } else if (selected) {
    stateText = "COLD · FUEL SELECTED";
  } else {
    stateText = "COLD · NO REACTOR LIT";
  }

  const board = suite ? (
    <SuiteBoard
      suite={suite}
      onClose={() => setSuite(null)}
      onExport={exportSuite}
    />
  ) : null;

  return (
    <div className="shell">
      <header className="hdr">
        <span className="wordmark">TOKAMAK</span>
        <span className={`hdr-state ${stateCls}`}>
          <span className={`lamp ${lampCls}`} />
          {stateText}
        </span>
        <span className="spacer" />
        {gpu && (
          <span className="gpu-chip">
            {gpu.name} · {gb(gpu.vram_total_bytes, 0)} GB
          </span>
        )}
        {server?.running && (
          <button className="danger" onClick={stop}>
            Shutdown
          </button>
        )}
      </header>

      <div className="deck">
        <Library
          models={primary}
          visionDirs={visionDirs}
          roots={roots}
          scanning={scanning}
          estimates={estimates}
          runningPath={runningPath ?? null}
          busy={busy}
          selectedPath={selectedPath}
          hoverPath={hoverPath}
          onHover={setHoverPath}
          onSelect={(path) => setSelectedPath(path === selectedPath ? null : path)}
          onIgnite={(m) => ignite(m)}
          onBench={runBench}
          onSuite={runSuite}
          onRescan={rescan}
          onAddFolder={addFolder}
          onRemoveFolder={removeFolder}
        />

        <div className="center">
          <Console
            server={server}
            metrics={metrics}
            ctxSize={liveCfg?.ctx ?? null}
            modelName={liveName}
            cfgText={
              liveCfg
                ? `${liveCfg.ngl}${
                    runningModel?.metadata?.block_count
                      ? `/${runningModel.metadata.block_count}`
                      : ""
                  } layers · ${ctxLabel(liveCfg.ctx)} ctx`
                : null
            }
            staged={staged}
            board={board}
            kvAlert={kvAlert}
            workspace={settings?.agent_workspace ?? null}
            onPickWorkspace={pickWorkspace}
          />
          <Dock
            selected={selected}
            selectedEst={selectedEst}
            liveModel={runningModel}
            liveEst={liveEst}
            liveCfg={liveCfg}
            metrics={metrics}
            uptimeMs={server?.uptime_ms ?? null}
            benchDetail={
              bench
                ? {
                    name: bench.name,
                    expected: bench.expected,
                    results: bench.results,
                    running: benching && !suite,
                  }
                : null
            }
            onCloseBench={() => setBench(null)}
          />
        </div>

        <Rail
          telemetry={telemetry}
          metrics={metrics}
          server={server}
          liveEst={liveEst}
          ghost={ghost}
          ctxSize={liveCfg?.ctx ?? null}
          history={history}
          kvAlert={kvAlert}
        />
      </div>

      <footer className="statusbar">
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
        <span className="scale-ctl">
          UI {Math.round(scale * 100)}%
          <button onClick={() => bumpScale(-0.1)} title="Ctrl+- / Ctrl+wheel">
            −
          </button>
          <button onClick={() => bumpScale(0.1)} title="Ctrl+= / Ctrl+wheel">
            +
          </button>
        </span>
        <span>poll 1 Hz</span>
        {server?.base_url && <span className="api">api {server.base_url} · /v1</span>}
        <span className="spacer" />
        <span>{roots.length} scan dirs</span>
        {server?.uptime_ms != null && server.running && (
          <span>session {fmtUptime(server.uptime_ms)}</span>
        )}
      </footer>

      {error && (
        <div className="toast-error">
          <button onClick={() => setError(null)}>✕</button>
          {error}
        </div>
      )}
    </div>
  );
}

function SuiteBoard({
  suite,
  onClose,
  onExport,
}: {
  suite: {
    running: boolean;
    current: string | null;
    total: number;
    rows: SuiteRow[];
    exportPath: string | null;
  };
  onClose: () => void;
  onExport: () => void;
}) {
  const done = suite.rows.filter((r) => !r.skipped);
  const skipped = suite.rows.filter((r) => r.skipped);
  const ranked = [...done].sort((a, b) => b.decode_tok_s - a.decode_tok_s);
  const best = ranked[0]?.decode_tok_s ?? 0;
  return (
    <div className="board">
      <div className="board-head">
        <span className="lbl">Bench Board</span>
        <span style={{ font: "10.5px var(--mono)", color: "var(--faint)" }}>
          ranked by decode tok/s · recommended config per model
        </span>
        <span className="spacer" />
        {!suite.running && done.length > 0 && <button onClick={onExport}>Export Report ⇣</button>}
        {!suite.running && <button onClick={onClose}>✕</button>}
      </div>
      <div className="board-cols">
        <span className="c-rank">#</span>
        <span className="c-name">Model · Config</span>
        <span className="c-bar">Decode tok/s</span>
        <span className="c-num">Prefill</span>
        <span className="c-num small">Load</span>
        <span className="c-num">Peak VRAM</span>
      </div>
      {ranked.map((r, i) => {
        const ratio = best > 0 ? r.decode_tok_s / best : 0;
        const isBest = i === 0 && r.decode_tok_s > 0;
        const fillCls = isBest ? "best" : ratio >= 0.5 ? "" : ratio >= 0.1 ? "slow" : "bad";
        return (
          <div key={r.model} className="board-row">
            <span className="c-rank" style={{ color: isBest ? "var(--plasma)" : "var(--low)" }}>
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="c-name" title={r.model}>
              {r.model}{" "}
              <span className="sub">
                {r.quant ?? ""} · {r.n_gpu_layers}L · {ctxLabel(r.ctx_size)}
              </span>
            </span>
            <span className="c-bar">
              <span className="track">
                <span className={`fill ${fillCls}`} style={{ width: `${Math.max(2, ratio * 100)}%` }} />
              </span>
              <span className={`c-val ${isBest ? "best" : ""}`}>{r.decode_tok_s.toFixed(1)}</span>
            </span>
            <span className="c-num">{Math.round(r.prefill_tok_s).toLocaleString()}</span>
            <span className="c-num small">{(r.load_ms / 1000).toFixed(1)}s</span>
            <span className="c-num">{gb(r.peak_vram_bytes, 1)} GB</span>
          </div>
        );
      })}
      {skipped.map((r) => (
        <div key={r.model} className="board-row skipped" title={r.skipped ?? undefined}>
          <span className="c-rank">—</span>
          <span className="c-name">{r.model}</span>
          <span className="c-bar" style={{ color: "var(--low)", font: "11px var(--mono)" }}>
            skipped · {r.skipped}
          </span>
        </div>
      ))}
      {suite.running && (
        <div className="board-row pending">
          measuring {suite.current ?? "…"} — loading + generating on the bench port…{" "}
          {suite.rows.length}/{suite.total}
        </div>
      )}
      <div className="board-foot">
        {suite.exportPath
          ? `report saved: ${suite.exportPath}`
          : !suite.running && done.length > 0
            ? "measured on your GPU — real generation, not estimated"
            : ""}
      </div>
    </div>
  );
}

function fmtUptime(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function dirOf(path: string): string {
  return path.slice(0, Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/")));
}
