import {
  ModelEntry,
  ScanRoot,
  VramEstimate,
  ctxLabel,
  gb,
  modelLabel,
} from "./types";

// Left bay: the models you already have, each with a live fit verdict against
// your GPU. Hovering a card projects its footprint onto the core's VRAM ring.

interface HangarProps {
  models: ModelEntry[];
  visionDirs: Set<string>;
  roots: ScanRoot[];
  scanning: boolean;
  estimates: Map<string, VramEstimate>;
  runningPath: string | null;
  busy: boolean;
  selectedPath: string | null;
  onHover: (path: string | null) => void;
  onSelect: (path: string) => void;
  onIgnite: (m: ModelEntry) => void;
  onBench: (m: ModelEntry) => void;
  onSuite: () => void;
  onRescan: () => void;
  onAddFolder: () => void;
  onRemoveFolder: (dir: string) => void;
}

function fitChip(est: VramEstimate | undefined, layers: number | null) {
  if (!est) return <span className="fit pending">measuring…</span>;
  if (est.full_offload)
    return <span className="fit full">FITS · {ctxLabel(est.ctx_size)} ctx</span>;
  if (est.fits)
    return (
      <span className="fit partial">
        GPU {est.n_gpu_layers}
        {layers ? `/${layers}` : ""} · {ctxLabel(est.ctx_size)}
      </span>
    );
  return <span className="fit none">CPU ONLY</span>;
}

function srcShort(r: ScanRoot): string {
  if (r.source === "huggingface") return "HF";
  if (r.source === "lm-studio") return "LM STUDIO";
  const parts = r.path.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? r.path;
}

export function Hangar(p: HangarProps) {
  return (
    <div className="bay">
      <div className="bay-head">
        <span className="microlabel">
          hangar · {p.models.length} craft{p.scanning ? " · scanning" : ""}
        </span>
        <div className="actions">
          <button
            onClick={p.onSuite}
            disabled={p.busy || p.models.length === 0}
            title="Benchmark every model at its recommended config and compare"
          >
            SUITE
          </button>
          <button onClick={p.onRescan} disabled={p.scanning}>
            RESCAN
          </button>
        </div>
      </div>

      <div className="bay-scroll" onMouseLeave={() => p.onHover(null)}>
        {p.models.length === 0 && !p.scanning && (
          <div className="empty-pad">
            No GGUF models found.
            <br />
            Add a folder below to begin.
          </div>
        )}
        {p.models.map((m) => {
          const md = m.metadata;
          const est = p.estimates.get(m.path);
          const live = p.runningPath === m.path;
          return (
            <div
              key={m.path}
              className={`craft ${live ? "live" : ""} ${
                p.selectedPath === m.path ? "selected" : ""
              }`}
              title={m.path}
              onMouseEnter={() => p.onHover(m.path)}
              onClick={() => p.onSelect(m.path)}
            >
              <div className="craft-name">
                {modelLabel(m)}
                {live && <span className="tag live">LIVE</span>}
                {p.visionDirs.has(dirKey(m.path)) && (
                  <span className="tag vision">VISION</span>
                )}
                {m.shard_total && (
                  <span className="tag shards">{m.shard_total} SHARDS</span>
                )}
              </div>
              <div className="craft-specs">
                <span>
                  <b>{md?.architecture ?? "?"}</b>
                </span>
                <span>
                  <b>{md?.quant_label ?? "?"}</b>
                </span>
                <span>
                  ctx <b>{ctxLabel(md?.context_length ?? null)}</b>
                </span>
                <span>
                  <b>{gb(m.size_bytes)}</b> gb
                </span>
              </div>
              <div className="craft-foot">
                {fitChip(est, md?.block_count ?? null)}
                <div className="craft-actions">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      p.onBench(m);
                    }}
                    disabled={p.busy || !!m.parse_error}
                  >
                    BENCH
                  </button>
                  <button
                    className="ignite"
                    onClick={(e) => {
                      e.stopPropagation();
                      p.onIgnite(m);
                    }}
                    disabled={p.busy || !!m.parse_error || live}
                  >
                    {live ? "LIVE" : "IGNITE"}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="sources">
        <span className="microlabel">sources</span>
        <div className="chips">
          {p.roots.map((r) => (
            <span
              key={r.path}
              className={`src-chip ${r.exists ? "" : "missing"}`}
              title={r.path}
            >
              {srcShort(r)}
              {r.source === "folder" && (
                <button onClick={() => p.onRemoveFolder(r.path)} title="Remove">
                  ✕
                </button>
              )}
            </span>
          ))}
          <span className="src-chip add" onClick={p.onAddFolder}>
            + ADD
          </span>
        </div>
      </div>
    </div>
  );
}

function dirKey(path: string): string {
  return path.slice(0, Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/")));
}
