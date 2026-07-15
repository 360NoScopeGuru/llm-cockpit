import { useEffect, useRef } from "react";
import { GpuSnapshot, InferenceMetrics, ServerStatus } from "./types";

// The reactor core: a live canvas instrument rendering VRAM as the outer ring,
// GPU utilization as a heat-tinted inner sweep, KV-cache as an amber band, and
// token generation as orbiting particles. Hovering a model in the hangar
// projects its estimated footprint onto the VRAM ring as a violet ghost arc.

export interface CoreGhost {
  bytes: number;
  fits: boolean;
}

interface CoreProps {
  gpu: GpuSnapshot | null;
  infer: InferenceMetrics | null;
  server: ServerStatus | null;
  ghost: CoreGhost | null;
  modelName: string | null;
}

interface Particle {
  a: number;
  r: number;
  va: number;
  vr: number;
  life: number;
}

// Arc geometry: 270° sweep opening downward.
const A0 = Math.PI * 0.75;
const SWEEP = Math.PI * 1.5;

const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

function tempColor(t: number): string {
  if (t < 60) return "#36f1b6";
  if (t < 78) return "#ffb454";
  return "#ff4d6d";
}

export function Core(props: CoreProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef<CoreProps>(props);
  propsRef.current = props;

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let last = performance.now();
    let particles: Particle[] = [];
    let spawnCarry = 0;
    // Display values eased toward targets each frame.
    const d = { vram: 0, ghost: 0, util: 0, temp: 40, kv: 0, decode: 0, spin: 0 };

    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const p = propsRef.current;

      // Resize to element * dpr.
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const gpu = p.gpu;
      const total = gpu?.vram_total_bytes || 1;
      const busy = (p.infer?.requests_processing ?? 0) > 0;
      const health = p.server?.running ? p.server.health : "stopped";

      // Ease displayed values.
      d.vram = lerp(d.vram, gpu ? gpu.vram_used_bytes / total : 0, 0.08);
      d.ghost = lerp(d.ghost, p.ghost ? p.ghost.bytes / total : 0, 0.14);
      d.util = lerp(d.util, (gpu?.gpu_util_pct ?? 0) / 100, 0.1);
      d.temp = lerp(d.temp, gpu?.temperature_c ?? 40, 0.05);
      d.kv = lerp(d.kv, p.infer?.kv_cache_usage_ratio ?? 0, 0.1);
      d.decode = lerp(d.decode, busy ? p.infer?.predicted_tokens_per_sec ?? 0 : d.decode, 0.15);
      d.spin += dt * 2.2;

      const cx = w / 2;
      const cy = h / 2;
      const R = Math.max(60, Math.min(w, h) / 2 - 46);

      const arc = (r: number, from: number, frac: number, width: number, style: string, glow = 0) => {
        ctx.beginPath();
        ctx.arc(cx, cy, r, A0 + SWEEP * from, A0 + SWEEP * Math.min(1, from + frac));
        ctx.lineWidth = width;
        ctx.lineCap = "butt";
        ctx.strokeStyle = style;
        ctx.shadowBlur = glow;
        ctx.shadowColor = style;
        ctx.stroke();
        ctx.shadowBlur = 0;
      };

      // --- VRAM ring (outer) ---
      arc(R, 0, 1, 16, "rgba(215,229,224,0.055)");
      // GB tick marks every 4 GB.
      const gbTotal = total / 1e9;
      ctx.strokeStyle = "rgba(5,7,10,0.9)";
      ctx.lineWidth = 2;
      for (let g = 4; g < gbTotal; g += 4) {
        const a = A0 + SWEEP * (g / gbTotal);
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * (R - 9), cy + Math.sin(a) * (R - 9));
        ctx.lineTo(cx + Math.cos(a) * (R + 9), cy + Math.sin(a) * (R + 9));
        ctx.stroke();
      }
      // Used VRAM — teal→violet conic gradient.
      if (d.vram > 0.004) {
        let grad: CanvasGradient | string;
        try {
          const cg = ctx.createConicGradient(A0, cx, cy);
          cg.addColorStop(0, "#36f1b6");
          cg.addColorStop(0.75, "#8b7bff");
          grad = cg;
        } catch {
          grad = "#36f1b6";
        }
        ctx.save();
        ctx.strokeStyle = grad as string;
        ctx.lineWidth = 16;
        ctx.shadowBlur = 14;
        ctx.shadowColor = "rgba(54,241,182,0.4)";
        ctx.beginPath();
        ctx.arc(cx, cy, R, A0, A0 + SWEEP * Math.min(1, d.vram));
        ctx.stroke();
        ctx.restore();
      }
      // Ghost projection — hovering a model previews its footprint.
      if (d.ghost > 0.004) {
        const from = Math.min(1, d.vram);
        const fit = from + d.ghost <= 1;
        const pulse = 0.45 + 0.25 * Math.sin(now / 300);
        arc(
          R,
          from,
          Math.min(d.ghost, 1 - from),
          16,
          fit ? `rgba(139,123,255,${pulse})` : `rgba(139,123,255,${pulse})`,
          10
        );
        if (!fit) {
          // Overflow tail flashes danger at the end of the ring.
          arc(R, 0.985, 0.015, 16, `rgba(255,77,109,${0.5 + 0.4 * Math.sin(now / 160)})`, 16);
        }
      }

      // --- GPU util ring ---
      const heat = tempColor(d.temp);
      arc(R - 30, 0, 1, 7, "rgba(215,229,224,0.045)");
      if (health === "loading" || health === "starting") {
        // Ignition spinner: three rotating dashes.
        for (let i = 0; i < 3; i++) {
          const start = (d.spin + (i * Math.PI * 2) / 3) % (Math.PI * 2);
          ctx.beginPath();
          ctx.arc(cx, cy, R - 30, start, start + 0.5);
          ctx.lineWidth = 7;
          ctx.strokeStyle = "#ffb454";
          ctx.shadowBlur = 12;
          ctx.shadowColor = "#ffb454";
          ctx.stroke();
          ctx.shadowBlur = 0;
        }
      } else if (d.util > 0.004) {
        arc(R - 30, 0, d.util, 7, heat, 14);
      }

      // --- KV cache band ---
      // Above 90% the context is nearly exhausted (next stop: truncation or a
      // spill), so the band goes into a pulsing red alert and generation
      // particles drag to telegraph the latency cliff.
      const kvHot = d.kv >= 0.9;
      if (p.server?.running && health === "ok") {
        arc(R - 44, 0, 1, 4, "rgba(255,180,84,0.08)");
        if (d.kv > 0.004) {
          if (kvHot) {
            const pulse = 0.55 + 0.4 * Math.sin(now / 150);
            // Glitch echo: a faint offset twin ring while in overflow alert.
            arc(R - 46.5, 0, d.kv, 2, `rgba(255,77,109,${pulse * 0.35})`, 0);
            arc(R - 44, 0, d.kv, 4, `rgba(255,77,109,${pulse})`, 16);
          } else {
            arc(R - 44, 0, d.kv, 4, "#ffb454", 8);
          }
        }
      }

      // --- Particles while generating ---
      // In KV-overflow alert they crawl (and tint red): the visual analog of
      // the latency hit when the cache is saturated.
      const drag = kvHot ? 0.3 : 1;
      if (busy) {
        spawnCarry += dt * drag * Math.min(60, 6 + d.decode / 4);
        while (spawnCarry >= 1 && particles.length < 110) {
          spawnCarry -= 1;
          particles.push({
            a: A0 + Math.random() * SWEEP,
            r: R - 30,
            va: (Math.random() - 0.5) * 1.6,
            vr: 26 + Math.random() * 44,
            life: 1,
          });
        }
      }
      particles = particles.filter((pt) => pt.life > 0);
      for (const pt of particles) {
        pt.r += pt.vr * dt * drag;
        pt.a += pt.va * dt * drag;
        pt.life -= dt * 0.75;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(pt.a) * pt.r, cy + Math.sin(pt.a) * pt.r, 1.6, 0, Math.PI * 2);
        ctx.fillStyle = kvHot
          ? `rgba(255,77,109,${Math.max(0, pt.life) * 0.8})`
          : `rgba(54,241,182,${Math.max(0, pt.life) * 0.8})`;
        ctx.fill();
      }

      // --- Center readout ---
      ctx.textAlign = "center";
      const mono = (size: number) => `${size}px "Cascadia Mono", Consolas, monospace`;
      const micro = (text: string, y: number, color = "#5d7470") => {
        ctx.font = mono(9.5);
        ctx.fillStyle = color;
        ctx.save();
        ctx.letterSpacing = "3px";
        ctx.fillText(text.toUpperCase(), cx, y);
        ctx.restore();
      };

      if (busy) {
        ctx.font = mono(46);
        ctx.fillStyle = "#36f1b6";
        ctx.shadowBlur = 22;
        ctx.shadowColor = "rgba(54,241,182,0.55)";
        ctx.fillText(d.decode.toFixed(1), cx, cy + 6);
        ctx.shadowBlur = 0;
        micro("tok/s decode", cy + 26);
        ctx.font = mono(11);
        ctx.fillStyle = "#5d7470";
        ctx.fillText(`prefill ${(p.infer?.prompt_tokens_per_sec ?? 0).toFixed(0)} tok/s`, cx, cy + 46);
        if (kvHot) {
          ctx.fillStyle = `rgba(255,77,109,${0.6 + 0.4 * Math.sin(now / 150)})`;
          ctx.fillText("⚠ kv cache saturated — context nearly full", cx, cy + 66);
        }
      } else if (health === "ok") {
        ctx.font = mono(26);
        ctx.fillStyle = "#36f1b6";
        ctx.fillText("READY", cx, cy + 2);
        micro(p.modelName ?? "", cy + 24, "#d7e5e0");
        micro("awaiting prompt — open the console", cy + 42);
      } else if (health === "loading" || health === "starting") {
        const dots = ".".repeat(1 + (Math.floor(now / 400) % 3));
        ctx.font = mono(24);
        ctx.fillStyle = "#ffb454";
        ctx.fillText(`IGNITION${dots}`, cx, cy + 2);
        micro(p.modelName ?? "loading model", cy + 24, "#d7e5e0");
      } else if (health === "error") {
        ctx.font = mono(26);
        ctx.fillStyle = "#ff4d6d";
        ctx.fillText("FAULT", cx, cy + 2);
        micro("see diagnostics below", cy + 24);
      } else {
        ctx.font = mono(40);
        ctx.fillStyle = "#d7e5e0";
        ctx.fillText(`${Math.round(d.temp)}°`, cx, cy + 4);
        micro("core idle", cy + 26);
        micro(gpu?.name ?? "no gpu detected", cy + 44);
      }

      // VRAM caption above center.
      if (gpu) {
        ctx.font = mono(11);
        ctx.fillStyle = "#5d7470";
        ctx.fillText(
          `vram ${(gpu.vram_used_bytes / 1e9).toFixed(1)} / ${(total / 1e9).toFixed(1)} gb`,
          cx,
          cy - 52
        );
        // Ghost verdict readout while hovering.
        if (p.ghost) {
          ctx.fillStyle = p.ghost.fits ? "#8b7bff" : "#ff4d6d";
          ctx.fillText(
            p.ghost.fits
              ? `+${(p.ghost.bytes / 1e9).toFixed(1)} gb projected — fits`
              : `+${(p.ghost.bytes / 1e9).toFixed(1)} gb projected — exceeds vram`,
            cx,
            cy - 34
          );
        }
        // Power + clock along the bottom gap of the ring.
        const pw = gpu.power_watts != null ? `${gpu.power_watts.toFixed(0)}w` : "—";
        const pl = gpu.power_limit_watts != null ? `/${gpu.power_limit_watts.toFixed(0)}w` : "";
        const ck = gpu.clock_graphics_mhz != null ? `${gpu.clock_graphics_mhz}mhz` : "";
        ctx.font = mono(10);
        ctx.fillStyle = "#5d7470";
        ctx.fillText(`${pw}${pl}   ${ck}   util ${gpu.gpu_util_pct}%`, cx, cy + R - 6);
      }

      raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={canvasRef} />;
}
