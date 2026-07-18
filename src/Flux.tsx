// Flux trace: 60 s of telemetry drawn as heat strips — brightness is the
// value. Newest sample lands at the right edge; history slides left. The
// third strip swaps from TEMP to KV pressure while containment is in alert.
//
// Deliberately NOT a <canvas>: accelerated canvases live in GPU memory, and
// this app's whole job is to fill GPU memory with model weights. Under VRAM
// pressure WebView2 evicts the canvas backing store and Chromium replaces it
// with a "content lost" sad-face placeholder. CSS gradients are rasterized
// per paint and cannot be lost.

export interface FluxSample {
  decode: number;
  util: number;
  temp: number;
  kv: number;
}

const WINDOW = 60;
const BASE = "#1a1614";

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

function stripGradient(values: number[], stops: string[]): string {
  if (values.length === 0) return BASE;
  const start = WINDOW - values.length;
  const segs: string[] = [];
  if (start > 0) segs.push(`${BASE} 0% ${((start / WINDOW) * 100).toFixed(2)}%`);
  for (let i = 0; i < values.length; i++) {
    const a = (((start + i) / WINDOW) * 100).toFixed(2);
    const b = (((start + i + 1) / WINDOW) * 100).toFixed(2);
    segs.push(`${ramp(stops, values[i])} ${a}% ${b}%`);
  }
  return `linear-gradient(90deg, ${segs.join(", ")})`;
}

function Strip({ values, stops }: { values: number[]; stops: string[] }) {
  return <div className="strip" style={{ background: stripGradient(values, stops) }} />;
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
