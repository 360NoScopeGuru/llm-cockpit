import { Flux, FluxSample } from "./Flux";
import {
  InferenceMetrics,
  ServerStatus,
  TelemetrySnapshot,
  VramEstimate,
  gb,
} from "./types";

// Telemetry stack (right panel): flux trace on top, decode headline, the
// VRAM rod bank (1 rod ≈ 1 GB; hovering a model in the library projects its
// footprint as dashed ghost rods), the vitals grid, and KV containment at
// the bottom — which goes into a pulsing alert at 90%.

export interface Ghost {
  name: string;
  bytes: number;
  fits: boolean;
  layers: string | null;
}

interface RailProps {
  telemetry: TelemetrySnapshot | null;
  metrics: InferenceMetrics | null;
  server: ServerStatus | null;
  liveEst: VramEstimate | null;
  ghost: Ghost | null;
  ctxSize: number | null;
  history: FluxSample[];
  kvAlert: boolean;
}

interface Seg {
  bytes: number;
  color: string;
}

function RodBank({
  totalBytes,
  segments,
  ghostBytes,
  ghostFits,
}: {
  totalBytes: number;
  segments: Seg[];
  ghostBytes: number;
  ghostFits: boolean;
}) {
  const GB = 1e9;
  const rods = Math.min(32, Math.max(8, Math.round(totalBytes / GB)));
  const perRod = totalBytes / rods;
  const usedBytes = segments.reduce((s, x) => s + x.bytes, 0);

  const els = [];
  for (let i = 0; i < rods; i++) {
    const lo = i * perRod;
    const hi = lo + perRod;
    if (lo >= usedBytes && ghostBytes > 0 && lo < usedBytes + ghostBytes) {
      els.push(<div key={i} className={`rod ghost ${ghostFits ? "" : "bad"}`} />);
      continue;
    }
    // Build this rod's fill as a bottom-up gradient across the segments that
    // overlap [lo, hi).
    const stops: string[] = [];
    let cum = 0;
    let prevPct = 0;
    for (const s of segments) {
      const segLo = cum;
      const segHi = cum + s.bytes;
      cum = segHi;
      const overlap = Math.min(hi, segHi) - Math.max(lo, segLo);
      if (overlap <= 0) continue;
      const pct = prevPct + (overlap / perRod) * 100;
      stops.push(`${s.color} ${prevPct.toFixed(1)}% ${pct.toFixed(1)}%`);
      prevPct = pct;
    }
    stops.push(`transparent ${prevPct.toFixed(1)}% 100%`);
    els.push(
      <div key={i} className="rod" style={{ background: `linear-gradient(0deg, ${stops.join(", ")})` }} />
    );
  }

  const ticks = [];
  for (let i = 0; i < rods; i++) {
    const n = i + 1;
    ticks.push(
      <span key={i} style={{ flex: 1, textAlign: "center" }}>
        {n === 1 || n % 4 === 0 ? n : ""}
      </span>
    );
  }

  return (
    <>
      <div className="rodbank">{els}</div>
      <div className="rod-ticks">{ticks}</div>
    </>
  );
}

function Vital({
  label,
  value,
  unit,
  tone,
}: {
  label: string;
  value: string;
  unit: string;
  tone?: "hot" | "warm" | "bad";
}) {
  return (
    <div className="vital">
      <div className="lbl faint">{label}</div>
      <div className={`num ${tone ?? ""}`}>
        {value}
        <small> {unit}</small>
      </div>
    </div>
  );
}

export function Rail(p: RailProps) {
  const t = p.telemetry;
  const g = t?.gpus[0] ?? null;
  const live = !!p.server?.running && p.server.health === "ok";
  const generating = live && (p.metrics?.requests_processing ?? 0) > 0;

  const vramTotal = g?.vram_total_bytes ?? 0;
  const vramUsed = g?.vram_used_bytes ?? 0;

  // Segment the rod bank: while our reactor is live, break its footprint
  // into weights / KV used / KV reserved + overhead from the estimate, with
  // anything else on the card (desktop, other apps) as a leading dim band.
  let segments: Seg[];
  const kvRatio = p.metrics?.kv_cache_usage_ratio ?? 0;
  if (live && p.liveEst) {
    const e = p.liveEst;
    const kvUsed = e.est_kv_bytes * kvRatio;
    const kvRest = e.est_kv_bytes - kvUsed + e.est_overhead_bytes;
    const other = Math.max(0, vramUsed - (e.est_weights_bytes + e.est_kv_bytes + e.est_overhead_bytes));
    segments = [
      { bytes: other, color: "#3b3229" },
      { bytes: e.est_weights_bytes, color: "#eda03f" },
      { bytes: kvUsed, color: "#c07f2e" },
      { bytes: kvRest, color: "#8a5f2a" },
    ];
  } else {
    segments = [{ bytes: vramUsed, color: "#8a5f2a" }];
  }

  const ghostBytes = p.ghost ? Math.max(0, p.ghost.bytes) : 0;
  const ghostOver = p.ghost ? vramUsed + p.ghost.bytes > vramTotal : false;

  const kvTokens = p.metrics?.kv_cache_tokens ?? 0;
  const kvPct = Math.round(kvRatio * 100);

  const tempTone = g?.temperature_c == null ? undefined : g.temperature_c >= 85 ? "bad" : g.temperature_c >= 72 ? "warm" : live ? "hot" : undefined;
  const activeTone = live ? ("hot" as const) : undefined;

  return (
    <div className="rail">
      <div className="rail-block">
        <div className="rail-head">
          <span className="lbl">Flux Trace</span>
          <span className="note">60 s · 1 Hz</span>
        </div>
        <Flux history={p.history} alert={p.kvAlert} />
      </div>

      <div className="rail-block headline">
        <div className="lbl">Decode</div>
        <div className={`big ${generating ? "" : "idle"}`}>
          {p.metrics ? p.metrics.predicted_tokens_per_sec.toFixed(1) : "0.0"}
          <small> tok/s</small>
        </div>
        <div className="sub">
          {p.metrics
            ? `prefill ${Math.round(p.metrics.prompt_tokens_per_sec).toLocaleString()} tok/s · total ${p.metrics.predicted_tokens_total.toLocaleString()} tok`
            : "no reactor lit"}
        </div>
      </div>

      <div className="rail-block">
        <div className="rail-head">
          <span className="lbl">Rod Bank · VRAM</span>
          <span className={`val ${p.ghost ? "ghosted" : ""}`}>
            {p.ghost ? `◐ ${gb(vramUsed + ghostBytes)} ` : `${gb(vramUsed)} `}
            <small>/ {gb(vramTotal)} GB</small>
          </span>
        </div>
        {vramTotal > 0 && (
          <RodBank
            totalBytes={vramTotal}
            segments={segments}
            ghostBytes={ghostBytes}
            ghostFits={p.ghost?.fits ?? true}
          />
        )}
        {p.ghost ? (
          <div className={`rod-caption ${ghostOver ? "bad" : "warn"}`}>
            ghost = {p.ghost.name}
            {p.ghost.layers ? ` · ${p.ghost.layers} layers` : ""}
            {ghostOver ? " · over capacity" : ""}
          </div>
        ) : live && p.liveEst ? (
          <div className="rod-caption">
            weights {gb(p.liveEst.est_weights_bytes)} · kv {gb(p.liveEst.est_kv_bytes)} · ovh{" "}
            {gb(p.liveEst.est_overhead_bytes)} GB
          </div>
        ) : (
          <div className="rod-caption">1 rod ≈ 1 GB · hover a model to project its footprint</div>
        )}
      </div>

      <div className="rail-block vitals">
        <Vital label="GPU Util" value={`${Math.round(g?.gpu_util_pct ?? 0)}`} unit="%" tone={activeTone} />
        <Vital label="Temp" value={g?.temperature_c != null ? `${g.temperature_c}` : "—"} unit="°C" tone={tempTone} />
        <Vital
          label="Power"
          value={g?.power_watts != null ? g.power_watts.toFixed(0) : "—"}
          unit={`/ ${g?.power_limit_watts?.toFixed(0) ?? "—"} W`}
          tone={activeTone}
        />
        <Vital
          label="Core · Mem"
          value={g?.clock_graphics_mhz != null ? `${g.clock_graphics_mhz}` : "—"}
          unit={`· ${g?.clock_mem_mhz ?? "—"} MHz`}
          tone={activeTone}
        />
        <Vital
          label="Sys RAM"
          value={t ? gb(t.ram_used_bytes) : "—"}
          unit={`/ ${t ? gb(t.ram_total_bytes, 0) : "—"} GB`}
          tone={activeTone}
        />
        <Vital label="CPU" value={`${Math.round(t?.cpu_util_pct ?? 0)}`} unit="%" tone={activeTone} />
      </div>

      <div className={`kv-block ${p.kvAlert ? "alert" : ""}`}>
        <div className="kv-head">
          <span className="lbl">{p.kvAlert ? "KV Cache · Alert" : "KV Cache"}</span>
          <span className={`kv-val ${p.metrics ? "" : "empty"}`}>
            {p.metrics
              ? p.ctxSize
                ? `${kvTokens.toLocaleString()} / ${p.ctxSize.toLocaleString()} · ${kvPct}%`
                : `${kvPct}%`
              : "—"}
          </span>
        </div>
        <div className="kv-bar">
          <div className="fill" style={{ width: `${Math.min(100, kvPct)}%` }} />
        </div>
      </div>
    </div>
  );
}
