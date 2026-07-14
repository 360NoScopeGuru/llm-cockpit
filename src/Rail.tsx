import {
  BenchResult,
  InferenceMetrics,
  ModelEntry,
  TelemetrySnapshot,
  VramEstimate,
  ctxLabel,
  gb,
  modelLabel,
} from "./types";

// Right bay: system vitals always on top; below, a contextual panel — bench
// results while a benchmark runs, otherwise the selected model's config detail
// (recommendation, VRAM budget breakdown, context ladder).

interface RailProps {
  telemetry: TelemetrySnapshot | null;
  metrics: InferenceMetrics | null;
  selected: ModelEntry | null;
  estimate: VramEstimate | null;
  bench: { name: string; expected: number; results: BenchResult[] } | null;
  benching: boolean;
  busy: boolean;
  onIgniteWith: (m: ModelEntry, ngl: number, ctx: number) => void;
  onCloseBench: () => void;
}

function Slim({
  label,
  pct,
  text,
}: {
  label: string;
  pct: number;
  text: string;
}) {
  const cls = pct >= 85 ? "hot" : pct >= 60 ? "warm" : "";
  return (
    <div className="slim">
      <div className="slim-head">
        <span>{label}</span>
        <b>{text}</b>
      </div>
      <div className="slim-track">
        <div
          className={`slim-fill ${cls}`}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
    </div>
  );
}

export function Rail(p: RailProps) {
  const t = p.telemetry;
  const g = t?.gpus[0] ?? null;
  const ramPct = t && t.ram_total_bytes > 0 ? (t.ram_used_bytes / t.ram_total_bytes) * 100 : 0;

  return (
    <div className="bay right">
      <div className="rail-section">
        <h4 className="microlabel">system</h4>
        <Slim
          label="RAM"
          pct={ramPct}
          text={t ? `${gb(t.ram_used_bytes)} / ${gb(t.ram_total_bytes)} gb` : "—"}
        />
        <Slim label="CPU" pct={t?.cpu_util_pct ?? 0} text={`${(t?.cpu_util_pct ?? 0).toFixed(0)}%`} />
        {g && (
          <div className="grid2">
            <div className="stat">
              <span className="microlabel">gpu temp</span>
              <span className="num">
                {g.temperature_c ?? "—"}
                <small> °c</small>
              </span>
            </div>
            <div className="stat">
              <span className="microlabel">power</span>
              <span className="num">
                {g.power_watts?.toFixed(0) ?? "—"}
                <small> / {g.power_limit_watts?.toFixed(0) ?? "—"} w</small>
              </span>
            </div>
            <div className="stat">
              <span className="microlabel">core clock</span>
              <span className="num">
                {g.clock_graphics_mhz ?? "—"}
                <small> mhz</small>
              </span>
            </div>
            <div className="stat">
              <span className="microlabel">mem clock</span>
              <span className="num">
                {g.clock_mem_mhz ?? "—"}
                <small> mhz</small>
              </span>
            </div>
          </div>
        )}
        {p.metrics && (
          <div className="grid2" style={{ marginTop: 10 }}>
            <div className="stat">
              <span className="microlabel">tokens out</span>
              <span className="num">{p.metrics.predicted_tokens_total.toFixed(0)}</span>
            </div>
            <div className="stat">
              <span className="microlabel">kv cache</span>
              <span className="num">
                {(p.metrics.kv_cache_usage_ratio * 100).toFixed(0)}
                <small> %</small>
              </span>
            </div>
          </div>
        )}
      </div>

      {p.bench ? (
        <div className="rail-section">
          <h4 className="microlabel">
            bench · {p.bench.name}
            {!p.benching && (
              <button style={{ float: "right", padding: "0 6px", fontSize: 10 }} onClick={p.onCloseBench}>
                ✕
              </button>
            )}
          </h4>
          <BenchRows bench={p.bench} benching={p.benching} />
        </div>
      ) : p.selected && p.estimate ? (
        <ConfigPanel
          model={p.selected}
          est={p.estimate}
          busy={p.busy}
          onIgniteWith={p.onIgniteWith}
        />
      ) : (
        <div className="rail-section">
          <h4 className="microlabel">config</h4>
          <div className="empty-pad">Select a craft in the hangar to see its optimal configuration.</div>
        </div>
      )}
    </div>
  );
}

function BenchRows({
  bench,
  benching,
}: {
  bench: { expected: number; results: BenchResult[] };
  benching: boolean;
}) {
  const best = bench.results.reduce(
    (m, r) => (r.loaded && r.decode_tok_s > m ? r.decode_tok_s : m),
    0
  );
  return (
    <div className="bench-rows">
      {bench.results.map((r, i) => (
        <div key={i} className={`bench-row ${r.loaded && r.decode_tok_s === best && best > 0 ? "best" : ""}`}>
          <div className="rowhead">
            <span>
              ngl {r.n_gpu_layers} · ctx {ctxLabel(r.ctx_size)}
            </span>
            <span>{r.loaded ? `${(r.load_ms / 1000).toFixed(1)}s load` : ""}</span>
          </div>
          {r.loaded ? (
            <div className="rowmain">
              <span className="decode">
                {r.decode_tok_s.toFixed(1)} <small>TOK/S</small>
              </span>
              <span className="aux">
                prefill {r.prefill_tok_s.toFixed(0)}
                <br />
                peak {gb(r.peak_vram_bytes, 2)} gb
              </span>
            </div>
          ) : (
            <div className="err">{r.error}</div>
          )}
        </div>
      ))}
      {benching &&
        Array.from({ length: Math.max(0, bench.expected - bench.results.length) }).map((_, i) => (
          <div key={`p${i}`} className="bench-row pending">
            measuring{i === 0 && bench.results.length === 0 ? " — loading model" : ""}…
          </div>
        ))}
      {!benching && best > 0 && (
        <div className="microlabel" style={{ marginTop: 4 }}>
          measured on your gpu — not estimated
        </div>
      )}
    </div>
  );
}

function ConfigPanel({
  model,
  est,
  busy,
  onIgniteWith,
}: {
  model: ModelEntry;
  est: VramEstimate;
  busy: boolean;
  onIgniteWith: (m: ModelEntry, ngl: number, ctx: number) => void;
}) {
  const total = est.est_total_bytes || 1;
  const pct = (b: number) => `${(b / total) * 100}%`;
  return (
    <div className="rail-section">
      <h4 className="microlabel">config · {modelLabel(model)}</h4>
      <div className="reco-line">
        <span className="big">
          {est.full_offload ? "FULL GPU" : est.fits ? `${est.n_gpu_layers} LAYERS` : "CPU ONLY"}
        </span>
        <span className="num" style={{ color: "var(--muted)", fontSize: 11 }}>
          @ {ctxLabel(est.ctx_size)} ctx
        </span>
      </div>
      <div className="stackbar">
        <div className="w" style={{ width: pct(est.est_weights_bytes) }} />
        <div className="k" style={{ width: pct(est.est_kv_bytes) }} />
        <div className="o" style={{ width: pct(est.est_overhead_bytes) }} />
      </div>
      <div className="legend">
        <span>
          <i className="iw" />
          weights {gb(est.est_weights_bytes)}g
        </span>
        <span>
          <i className="ik" />
          kv {gb(est.est_kv_bytes)}g
        </span>
        <span>
          <i className="io" />
          sys {gb(est.est_overhead_bytes)}g
        </span>
      </div>
      <span className="microlabel">context ladder (full offload)</span>
      <div className="ladder">
        {est.context_options.map((o) => (
          <div key={o.ctx} className="rung">
            <b>{ctxLabel(o.ctx)}</b>
            <span>{gb(o.est_total_bytes)} gb</span>
            <span className={o.fits ? "yes" : "no"}>{o.fits ? "✓" : "✗"}</span>
          </div>
        ))}
      </div>
      {est.notes.length > 0 && (
        <ul className="rail-notes">
          {est.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
      <button
        className="ignite wide-btn"
        disabled={busy}
        onClick={() => onIgniteWith(model, est.n_gpu_layers, est.ctx_size)}
      >
        IGNITE · ngl {est.n_gpu_layers} · ctx {ctxLabel(est.ctx_size)}
      </button>
    </div>
  );
}
