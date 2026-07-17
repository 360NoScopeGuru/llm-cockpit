import {
  ModelEntry,
  ScanRoot,
  VramEstimate,
  ctxLabel,
  gb,
} from "./types";

// Fuel library (left panel): every GGUF on disk with a live fit verdict
// against the GPU. Hovering a row projects its footprint onto the rod bank;
// selecting exposes IGNITE / BENCH and fills the containment dock.

interface LibraryProps {
  models: ModelEntry[];
  visionDirs: Set<string>;
  roots: ScanRoot[];
  scanning: boolean;
  estimates: Map<string, VramEstimate>;
  runningPath: string | null;
  busy: boolean;
  selectedPath: string | null;
  hoverPath: string | null;
  onHover: (path: string | null) => void;
  onSelect: (path: string) => void;
  onIgnite: (m: ModelEntry) => void;
  onBench: (m: ModelEntry) => void;
  onSuite: () => void;
  onRescan: () => void;
  onAddFolder: () => void;
  onRemoveFolder: (dir: string) => void;
}

function verdict(est: VramEstimate | undefined, layers: number | null) {
  if (!est) return <span className="verdict pending">…</span>;
  if (est.full_offload) return <span className="verdict full">● FITS</span>;
  if (est.fits)
    return (
      <span className="verdict partial">
        ◐ {est.n_gpu_layers}
        {layers ? `/${layers}` : ""}
      </span>
    );
  return <span className="verdict none">○ CPU</span>;
}

function srcShort(r: ScanRoot): string {
  if (r.source === "huggingface") return "HF";
  if (r.source === "lm-studio") return "LM Studio";
  const parts = r.path.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? r.path;
}

function dirKey(path: string): string {
  return path.slice(0, Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/")));
}

export function Library(p: LibraryProps) {
  const totalBytes = p.models.reduce((s, m) => s + m.size_bytes, 0);
  return (
    <div className="library">
      <div className="lib-head">
        <span className="lbl">Fuel Library</span>
        <span className="lib-count">
          {p.models.length} GGUF · {gb(totalBytes, 1)} GB{p.scanning ? " · scanning…" : ""}
        </span>
      </div>
      <div className="lib-actions">
        <button onClick={p.onRescan} disabled={p.scanning}>
          ↻ Rescan
        </button>
        <button
          onClick={p.onSuite}
          disabled={p.busy || p.models.length === 0}
          title="Benchmark every model at its recommended config and rank them"
        >
          Bench All
        </button>
        <button onClick={p.onAddFolder}>+ Dir</button>
      </div>

      <div className="lib-scroll" onMouseLeave={() => p.onHover(null)}>
        {p.models.length === 0 && !p.scanning && (
          <div className="dock-empty" style={{ padding: "24px 20px" }}>
            <div className="inner">
              <span className="sub">
                No GGUF models found. Add a folder with + DIR to begin.
              </span>
            </div>
          </div>
        )}
        {p.models.map((m) => {
          const md = m.metadata;
          const est = p.estimates.get(m.path);
          const live = p.runningPath === m.path;
          const sel = p.selectedPath === m.path;
          const hovered = p.hoverPath === m.path;
          return (
            <div
              key={m.path}
              className={`fuel-row ${sel ? "selected" : ""}`}
              title={m.path}
              onMouseEnter={() => p.onHover(m.path)}
              onClick={() => p.onSelect(m.path)}
            >
              <span className="fuel-name">{m.file_name}</span>
              <div className="fuel-meta">
                <span>{(md?.architecture ?? "?").toUpperCase()}</span>
                <span className="chip">{md?.quant_label ?? "?"}</span>
                <span>{gb(m.size_bytes)} GB</span>
                <span>{ctxLabel(md?.context_length ?? null)}</span>
                {p.visionDirs.has(dirKey(m.path)) && <span className="chip vision">VISION</span>}
                {m.shard_total && <span className="chip">×{m.shard_total}</span>}
                {live && <span className="chip running">▶ RUNNING</span>}
                {verdict(est, md?.block_count ?? null)}
              </div>
              {sel && (
                <div className="fuel-cta">
                  <button
                    className="primary"
                    onClick={(e) => {
                      e.stopPropagation();
                      p.onIgnite(m);
                    }}
                    disabled={p.busy || !!m.parse_error || live}
                  >
                    {live ? "Live" : "Ignite"}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      p.onBench(m);
                    }}
                    disabled={p.busy || !!m.parse_error}
                  >
                    Bench
                  </button>
                </div>
              )}
              {!sel && hovered && est && !live && (
                <span className="fuel-hint">
                  hover → projecting{" "}
                  {gb(est.est_weights_bytes + est.est_kv_bytes + est.est_overhead_bytes)} /{" "}
                  {gb(est.gpu_total_bytes)} GB on rod bank ▸
                </span>
              )}
              {m.parse_error && <span className="fuel-hint">parse error: {m.parse_error}</span>}
            </div>
          );
        })}
      </div>

      <div className="lib-foot">
        {p.roots.map((r) => (
          <span key={r.path} className={`src ${r.exists ? "" : "missing"}`} title={r.path}>
            {srcShort(r)}
            {r.source === "folder" && (
              <button onClick={() => p.onRemoveFolder(r.path)} title="Remove folder">
                ✕
              </button>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}
