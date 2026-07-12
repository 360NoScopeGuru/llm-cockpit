import "./AutoConfig.css";

export interface ContextOption {
  ctx: number;
  est_total_bytes: number;
  fits: boolean;
}

export interface VramEstimate {
  fits: boolean;
  full_offload: boolean;
  n_gpu_layers: number;
  ctx_size: number;
  est_weights_bytes: number;
  est_kv_bytes: number;
  est_overhead_bytes: number;
  est_total_bytes: number;
  budget_bytes: number;
  gpu_total_bytes: number;
  gpu_free_bytes: number;
  context_options: ContextOption[];
  notes: string[];
}

function gb(bytes: number): string {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

function ctxLabel(n: number): string {
  return n >= 1024 ? `${Math.round(n / 1024)}K` : String(n);
}

export function AutoConfigPanel({
  name,
  estimate,
  launching,
  onLaunch,
  onDismiss,
}: {
  name: string;
  estimate: VramEstimate;
  launching: boolean;
  onLaunch: (nGpuLayers: number, ctxSize: number) => void;
  onDismiss: () => void;
}) {
  const total = estimate.gpu_total_bytes || 1;
  const pct = (b: number) => `${Math.min(100, (b / total) * 100)}%`;
  const budgetPct = `${Math.min(100, (estimate.budget_bytes / total) * 100)}%`;

  return (
    <div className="autoconfig">
      <div className="ac-head">
        <div>
          <span className="ac-title">Recommended config</span>
          <span className="ac-sub">{name}</span>
        </div>
        <button className="ac-dismiss" onClick={onDismiss}>
          ✕
        </button>
      </div>

      <div className="ac-reco">
        <span className={`ac-badge ${estimate.full_offload ? "full" : "partial"}`}>
          {estimate.full_offload ? "Full GPU offload" : "Partial offload"}
        </span>
        <span className="ac-detail">
          {estimate.n_gpu_layers} layers on GPU · {ctxLabel(estimate.ctx_size)} context
        </span>
        {!estimate.fits && <span className="ac-warn">won't fit — CPU fallback</span>}
      </div>

      <div className="ac-bar" title="Estimated VRAM usage vs GPU total">
        <div
          className="ac-seg weights"
          style={{ width: pct(estimate.est_weights_bytes) }}
        />
        <div className="ac-seg kv" style={{ width: pct(estimate.est_kv_bytes) }} />
        <div
          className="ac-seg overhead"
          style={{ width: pct(estimate.est_overhead_bytes) }}
        />
        <div className="ac-budget" style={{ left: budgetPct }} />
      </div>

      <div className="ac-legend">
        <span>
          <i className="dot weights" /> weights {gb(estimate.est_weights_bytes)}
        </span>
        <span>
          <i className="dot kv" /> KV cache {gb(estimate.est_kv_bytes)}
        </span>
        <span>
          <i className="dot overhead" /> overhead {gb(estimate.est_overhead_bytes)}
        </span>
        <span className="ac-total">
          = {gb(estimate.est_total_bytes)} / {gb(estimate.gpu_total_bytes)} total
        </span>
      </div>

      {estimate.notes.length > 0 && (
        <ul className="ac-notes">
          {estimate.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}

      <button
        className="ac-launch"
        disabled={launching}
        onClick={() => onLaunch(estimate.n_gpu_layers, estimate.ctx_size)}
      >
        {launching ? "Launching…" : "Launch with this config"}
      </button>
    </div>
  );
}
