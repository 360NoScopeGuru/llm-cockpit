import { useEffect, useRef } from "react";

// Flux trace: 60 s of telemetry drawn as heat strips — brightness is the
// value. Newest sample lands at the right edge; history slides left. The
// third strip swaps from TEMP to KV pressure while containment is in alert.

export interface FluxSample {
  decode: number;
  util: number;
  temp: number;
  kv: number;
}

const WINDOW = 60;

// Piecewise-linear color ramp over hex stops.
function ramp(stops: string[], v: number): string {
  const t = Math.min(1, Math.max(0, v)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const f = t - i;
  const a = hex(stops[i]);
  const b = hex(stops[i + 1]);
  const c = a.map((x, j) => Math.round(x + (b[j] - x) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function hex(s: string): number[] {
  return [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
}

const DECODE_STOPS = ["#1a1614", "#5a3d1a", "#8a5a20", "#a86e26", "#c07f2e", "#eda03f", "#f5c069"];
const UTIL_STOPS = ["#1a1614", "#6b4a1e", "#9a6524", "#c07f2e", "#e0972f"];
const TEMP_STOPS = ["#1d1712", "#33251a", "#5a3d22", "#7d5226", "#96622a"];
const KV_STOPS = ["#1a1614", "#4a2a1a", "#7d3a28", "#b04a36", "#d95c45"];

function Strip({
  values,
  stops,
}: {
  values: number[]; // normalized 0..1, oldest first
  stops: string[];
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth * dpr;
    const h = cv.clientHeight * dpr;
    if (cv.width !== w) cv.width = w;
    if (cv.height !== h) cv.height = h;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#1a1614";
    ctx.fillRect(0, 0, w, h);
    const cell = w / WINDOW;
    const start = WINDOW - values.length;
    for (let i = 0; i < values.length; i++) {
      ctx.fillStyle = ramp(stops, values[i]);
      // +1px overlap hides seams from fractional cell widths.
      ctx.fillRect((start + i) * cell, 0, cell + 1, h);
    }
  }, [values, stops]);
  return <canvas ref={ref} />;
}

export function Flux({ history, alert }: { history: FluxSample[]; alert: boolean }) {
  const maxDecode = Math.max(10, ...history.map((s) => s.decode));
  const decode = history.map((s) => s.decode / maxDecode);
  const util = history.map((s) => s.util / 100);
  const temp = history.map((s) => (s.temp - 35) / 55);
  const kv = history.map((s) => s.kv);
  const active = history.some((s) => s.decode > 0 || s.util > 20);
  return (
    <div className="flux">
      <div className="flux-row">
        <span className={`tag ${active ? "hot" : ""}`}>DECODE</span>
        <Strip values={decode} stops={DECODE_STOPS} />
      </div>
      <div className="flux-row">
        <span className={`tag ${active ? "hot" : ""}`}>UTIL</span>
        <Strip values={util} stops={UTIL_STOPS} />
      </div>
      <div className="flux-row">
        <span className={`tag ${alert ? "alert" : active ? "hot" : ""}`}>
          {alert ? "KV" : "TEMP"}
        </span>
        <Strip values={alert ? kv : temp} stops={alert ? KV_STOPS : TEMP_STOPS} />
      </div>
    </div>
  );
}
