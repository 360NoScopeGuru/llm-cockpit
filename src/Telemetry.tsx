import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./Telemetry.css";

// Mirrors the serde output of the Rust `telemetry` module (snake_case).
interface GpuSnapshot {
  index: number;
  name: string;
  vram_used_bytes: number;
  vram_total_bytes: number;
  gpu_util_pct: number;
  mem_util_pct: number;
  temperature_c: number | null;
  power_watts: number | null;
  power_limit_watts: number | null;
  clock_graphics_mhz: number | null;
  clock_mem_mhz: number | null;
  fan_pct: number | null;
}

interface TelemetrySnapshot {
  nvml_available: boolean;
  error: string | null;
  gpus: GpuSnapshot[];
  ram_used_bytes: number;
  ram_total_bytes: number;
  cpu_util_pct: number;
  timestamp_ms: number;
}

const POLL_MS = 1000;

function gb(bytes: number): number {
  return bytes / 1e9;
}

function pctClass(pct: number): string {
  if (pct >= 85) return "hot";
  if (pct >= 60) return "warm";
  return "cool";
}

function Meter({
  label,
  pct,
  valueText,
}: {
  label: string;
  pct: number;
  valueText: string;
}) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="meter">
      <div className="meter-head">
        <span className="meter-label">{label}</span>
        <span className="meter-value">{valueText}</span>
      </div>
      <div className="meter-track">
        <div
          className={`meter-fill ${pctClass(clamped)}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

export function TelemetryCockpit() {
  const [snap, setSnap] = useState<TelemetrySnapshot | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await invoke<TelemetrySnapshot>("gpu_telemetry");
        if (alive) setSnap(s);
      } catch {
        /* transient; keep last snapshot */
      }
    };
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  if (!snap) {
    return <div className="cockpit loading">Reading hardware…</div>;
  }

  const ramPct =
    snap.ram_total_bytes > 0
      ? (snap.ram_used_bytes / snap.ram_total_bytes) * 100
      : 0;

  return (
    <div className="cockpit">
      {snap.gpus.map((g) => {
        const vramPct =
          g.vram_total_bytes > 0
            ? (g.vram_used_bytes / g.vram_total_bytes) * 100
            : 0;
        return (
          <div className="tile" key={g.index}>
            <div className="tile-title">
              {g.name}
              <span className="tile-stat">
                {g.temperature_c != null && <span>{g.temperature_c}°C</span>}
                {g.power_watts != null && (
                  <span>
                    {g.power_watts.toFixed(0)}
                    {g.power_limit_watts != null
                      ? `/${g.power_limit_watts.toFixed(0)}`
                      : ""}{" "}
                    W
                  </span>
                )}
                {g.clock_graphics_mhz != null && (
                  <span>{g.clock_graphics_mhz} MHz</span>
                )}
              </span>
            </div>
            <Meter
              label="VRAM"
              pct={vramPct}
              valueText={`${gb(g.vram_used_bytes).toFixed(1)} / ${gb(
                g.vram_total_bytes
              ).toFixed(1)} GB`}
            />
            <Meter
              label="GPU"
              pct={g.gpu_util_pct}
              valueText={`${g.gpu_util_pct}%`}
            />
          </div>
        );
      })}

      <div className="tile">
        <div className="tile-title">
          System
          {!snap.nvml_available && (
            <span className="tile-stat">
              <span className="warn" title={snap.error ?? ""}>
                no NVIDIA GPU
              </span>
            </span>
          )}
        </div>
        <Meter
          label="RAM"
          pct={ramPct}
          valueText={`${gb(snap.ram_used_bytes).toFixed(1)} / ${gb(
            snap.ram_total_bytes
          ).toFixed(1)} GB`}
        />
        <Meter
          label="CPU"
          pct={snap.cpu_util_pct}
          valueText={`${snap.cpu_util_pct.toFixed(0)}%`}
        />
      </div>
    </div>
  );
}
