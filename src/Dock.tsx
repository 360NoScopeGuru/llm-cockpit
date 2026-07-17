import {
  BenchResult,
  InferenceMetrics,
  ModelEntry,
  QuantAdvice,
  VramEstimate,
  ctxLabel,
  gb,
  modelLabel,
} from "./types";

// Containment dock (232px under the console): the arithmetic behind every
// recommendation. Cold + selected → VRAM budget | context ladder | quant
// advisor. Live → live budget | session stats | context ladder. After a
// single-model bench → the measured per-config table.

interface DockProps {
  selected: ModelEntry | null;
  selectedEst: VramEstimate | null;
  liveModel: ModelEntry | null;
  liveEst: VramEstimate | null;
  liveCfg: { ngl: number; ctx: number } | null;
  metrics: InferenceMetrics | null;
  uptimeMs: number | null;
  benchDetail: { name: string; expected: number; results: BenchResult[]; running: boolean } | null;
  onCloseBench: () => void;
}

function uptime(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function BudgetCol({
  est,
  kvRatio,
}: {
  est: VramEstimate;
  kvRatio: number | null; // null = cold (single kv segment)
}) {
  const budget = est.budget_bytes || 1;
  const pct = (b: number) => `${Math.min(100, (b / budget) * 100).toFixed(1)}%`;
  const headroom = est.budget_bytes - est.est_total_bytes;
  const kvUsed = kvRatio != null ? est.est_kv_bytes * kvRatio : 0;
  const kvRest = est.est_kv_bytes - kvUsed;
  return (
    <div className="dock-col">
      <div className="budget-bar">
        <span className="seg-w" style={{ width: pct(est.est_weights_bytes) }} />
        {kvRatio != null && <span className="seg-ku" style={{ width: pct(kvUsed) }} />}
        <span className="seg-k" style={{ width: pct(kvRatio != null ? kvRest : est.est_kv_bytes) }} />
        <span className="seg-o" style={{ width: pct(est.est_overhead_bytes) }} />
      </div>
      <div className="budget-legend">
        <span className="li">
          <span className="sw w" />
          weights
          <span className="val">{gb(est.est_weights_bytes, 2)} GB</span>
        </span>
        {kvRatio != null ? (
          <>
            <span className="li">
              <span className="sw ku" />
              kv used
              <span className="val">{gb(kvUsed, 2)} GB</span>
            </span>
            <span className="li">
              <span className="sw k" />
              kv reserved
              <span className="val">{gb(kvRest, 2)} GB</span>
            </span>
          </>
        ) : (
          <span className="li">
            <span className="sw k" />
            kv cache · {ctxLabel(est.ctx_size)}
            <span className="val">{gb(est.est_kv_bytes, 2)} GB</span>
          </span>
        )}
        <span className="li">
          <span className="sw o" />
          overhead
          <span className="val">{gb(est.est_overhead_bytes, 2)} GB</span>
        </span>
        <span className="li head-room">
          <span className="sw h" />
          headroom
          <span className={`val ${headroom >= 0 ? "good" : "bad"}`}>
            {headroom >= 0 ? "" : "−"}
            {gb(Math.abs(headroom), 2)} GB
          </span>
        </span>
      </div>
    </div>
  );
}

function ContextCol({ est }: { est: VramEstimate }) {
  return (
    <div className="dock-col">
      <div className="lbl faint">Context Ladder</div>
      <div className="ladder">
        {est.context_options.map((o) => {
          const rec = o.ctx === est.ctx_size;
          const tight = o.fits && o.est_total_bytes > est.budget_bytes * 0.92;
          return (
            <span key={o.ctx} className="rung">
              <span className={o.fits ? "ok" : "no"}>{o.fits ? "✓" : "✗"}</span>
              <span className={`k ${rec ? "hot" : ""}`} style={{ width: 36 }}>
                {ctxLabel(o.ctx)}
              </span>
              <span className="v">{gb(o.est_total_bytes)} GB</span>
              {rec ? (
                <span className="tag rec">● rec</span>
              ) : tight ? (
                <span className="tag tight">tight</span>
              ) : null}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function AdvisorCol({ advice }: { advice: QuantAdvice }) {
  const rec = advice.recommended;
  return (
    <div className="dock-col">
      <div className="lbl faint">Quant Advisor · ~{advice.est_params_b.toFixed(0)}B · this GPU</div>
      <div className="ladder">
        {advice.options.map((o) => {
          const isRec = !!rec && o.label === rec;
          return (
            <span key={o.label} className={`rung ${o.is_current ? "current" : ""}`}>
              <span className={o.fits ? "ok" : "no"}>{o.fits ? "✓" : "✗"}</span>
              <span className={`k ${o.is_current || isRec ? "hot" : ""}`}>{o.label}</span>
              <span className="v">{gb(o.est_weights_bytes)} GB</span>
              {isRec ? (
                <span className="tag rec">● sweet spot</span>
              ) : o.fits ? (
                <span className="tag dim">+{gb(o.headroom_bytes)} GB</span>
              ) : (
                <span className="tag bad">over w/ kv</span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function SessionCol({
  metrics,
  uptimeMs,
}: {
  metrics: InferenceMetrics | null;
  uptimeMs: number | null;
}) {
  return (
    <div className="dock-col">
      <div className="lbl faint">This Session</div>
      <div className="kv-list">
        <span className="row">
          <span className="k">decode</span>
          <span className="v">{metrics ? `${metrics.predicted_tokens_per_sec.toFixed(1)} tok/s` : "—"}</span>
        </span>
        <span className="row">
          <span className="k">prefill</span>
          <span className="v">{metrics ? `${Math.round(metrics.prompt_tokens_per_sec).toLocaleString()} tok/s` : "—"}</span>
        </span>
        <span className="row">
          <span className="k">total tok</span>
          <span className="v">{metrics ? metrics.predicted_tokens_total.toLocaleString() : "—"}</span>
        </span>
        <span className="row">
          <span className="k">kv tokens</span>
          <span className="v">{metrics ? metrics.kv_cache_tokens.toLocaleString() : "—"}</span>
        </span>
        <span className="row">
          <span className="k">in flight</span>
          <span className="v">{metrics ? metrics.requests_processing : "—"}</span>
        </span>
        <span className="row">
          <span className="k">uptime</span>
          <span className="v">{uptimeMs != null ? uptime(uptimeMs) : "—"}</span>
        </span>
      </div>
    </div>
  );
}

function BenchDetail({
  detail,
  onClose,
}: {
  detail: { name: string; expected: number; results: BenchResult[]; running: boolean };
  onClose: () => void;
}) {
  const best = detail.results.reduce((m, r) => (r.loaded ? Math.max(m, r.decode_tok_s) : m), 0);
  return (
    <>
      <div className="dock-head">
        <span className="lbl">Bench Detail</span>
        <span className="name">{detail.name}</span>
        <span className="right">
          {detail.running
            ? `measuring ${detail.results.length + 1} / ${detail.expected}…`
            : `${detail.results.length} configs measured`}
        </span>
        {!detail.running && <button onClick={onClose}>✕</button>}
      </div>
      <div style={{ flex: 1, padding: "0 14px 10px", overflowY: "auto" }}>
        <div className="board-cols" style={{ borderBottom: "1px solid var(--hair2)", padding: "4px 0" }}>
          <span style={{ width: 130 }}>Config</span>
          <span style={{ width: 90, textAlign: "right" }}>Decode</span>
          <span style={{ width: 90, textAlign: "right" }}>Prefill</span>
          <span style={{ width: 70, textAlign: "right" }}>Load</span>
          <span style={{ width: 90, textAlign: "right" }}>Peak VRAM</span>
          <span style={{ flex: 1 }} />
        </div>
        {detail.results.map((r, i) => (
          <div
            key={i}
            className="board-row"
            style={{ padding: "5px 0", borderBottom: "1px solid var(--hair3)", fontSize: 11 }}
          >
            <span style={{ width: 130, color: "var(--hi)" }}>
              {r.n_gpu_layers}L · {ctxLabel(r.ctx_size)}
            </span>
            {r.loaded ? (
              <>
                <span
                  style={{
                    width: 90,
                    textAlign: "right",
                    color: r.decode_tok_s === best && best > 0 ? "var(--plasma)" : "var(--hi)",
                    fontWeight: r.decode_tok_s === best && best > 0 ? 600 : 400,
                  }}
                >
                  {r.decode_tok_s.toFixed(1)}
                </span>
                <span style={{ width: 90, textAlign: "right", color: "var(--mid)" }}>
                  {Math.round(r.prefill_tok_s).toLocaleString()}
                </span>
                <span style={{ width: 70, textAlign: "right", color: "var(--mid)" }}>
                  {(r.load_ms / 1000).toFixed(1)}s
                </span>
                <span style={{ width: 90, textAlign: "right", color: "var(--mid)" }}>
                  {gb(r.peak_vram_bytes, 1)} GB
                </span>
                <span style={{ flex: 1, color: "var(--faint)", paddingLeft: 8 }}>
                  {r.decode_tok_s === best && best > 0 ? "fastest" : ""}
                </span>
              </>
            ) : (
              <span style={{ flex: 1, color: "var(--danger)" }}>{r.error ?? "failed"}</span>
            )}
          </div>
        ))}
        {detail.running && (
          <div className="board-row pending" style={{ padding: "6px 0" }}>
            measuring — loading model on the bench port…
          </div>
        )}
        {!detail.running && detail.results.length > 0 && (
          <div className="board-foot">measured on your GPU — real generation, not estimated</div>
        )}
      </div>
    </>
  );
}

export function Dock(p: DockProps) {
  if (p.benchDetail) {
    return (
      <div className="dock">
        <BenchDetail detail={p.benchDetail} onClose={p.onCloseBench} />
      </div>
    );
  }

  // Selecting a different model than the running one wins over the live view.
  const showSelected =
    p.selected && p.selectedEst && (!p.liveModel || p.selected.path !== p.liveModel.path);

  if (showSelected && p.selected && p.selectedEst) {
    const est = p.selectedEst;
    const fitTag = est.full_offload ? (
      <span className="state-tag" style={{ color: "var(--good)" }}>● FITS FULLY</span>
    ) : est.fits ? (
      <span className="state-tag" style={{ color: "var(--warn)" }}>◐ PARTIAL OFFLOAD</span>
    ) : (
      <span className="state-tag" style={{ color: "var(--danger)" }}>○ CPU ONLY</span>
    );
    return (
      <div className="dock">
        <div className="dock-head">
          <span className="lbl">Containment Budget</span>
          <span className="name">{modelLabel(p.selected)}</span>
          {fitTag}
          <span className="right">
            recommended{" "}
            <b>
              {est.n_gpu_layers}
              {p.selected.metadata?.block_count ? `/${p.selected.metadata.block_count}` : ""} layers ·{" "}
              {ctxLabel(est.ctx_size)} ctx
            </b>
          </span>
        </div>
        <div className="dock-grid">
          <BudgetCol est={est} kvRatio={null} />
          <ContextCol est={est} />
          {est.quant_advice ? <AdvisorCol advice={est.quant_advice} /> : <div className="dock-col" />}
        </div>
      </div>
    );
  }

  if (p.liveModel && p.liveEst) {
    return (
      <div className="dock">
        <div className="dock-head">
          <span className="lbl">Containment Budget</span>
          <span className="name">{modelLabel(p.liveModel)}</span>
          <span className="state-tag" style={{ color: "var(--plasma)" }}>▶ LIVE</span>
          <span className="right">
            {p.liveCfg ? (
              <b>
                {p.liveCfg.ngl}
                {p.liveModel.metadata?.block_count ? `/${p.liveModel.metadata.block_count}` : ""} layers ·{" "}
                {ctxLabel(p.liveCfg.ctx)} ctx
              </b>
            ) : null}
          </span>
        </div>
        <div className="dock-grid">
          <BudgetCol est={p.liveEst} kvRatio={p.metrics?.kv_cache_usage_ratio ?? 0} />
          <SessionCol metrics={p.metrics} uptimeMs={p.uptimeMs} />
          <ContextCol est={p.liveEst} />
        </div>
      </div>
    );
  }

  return (
    <div className="dock">
      <div className="dock-head">
        <span className="lbl">Containment Budget</span>
      </div>
      <div className="dock-empty">
        <div className="inner">
          <span className="sub">
            Select a model to see recommended layers &amp; context, the VRAM budget, the context
            ladder and the quant advisor.
          </span>
        </div>
      </div>
    </div>
  );
}
