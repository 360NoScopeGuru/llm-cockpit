//! Live hardware telemetry — the cockpit's "cheap wow".
//!
//! Polls NVIDIA GPU metrics via NVML (VRAM, utilization, temperature, power,
//! clocks) plus system RAM/CPU via sysinfo. NVML and the sysinfo `System` are
//! held in long-lived Tauri managed state so each poll is cheap and CPU usage
//! deltas are meaningful across calls.
//!
//! NVIDIA-first by design (v1 target). If NVML can't initialize (no NVIDIA
//! driver), snapshots still return system RAM/CPU with `nvml_available: false`.

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::Nvml;
use serde::Serialize;
use sysinfo::System;

#[derive(Debug, Clone, Serialize)]
pub struct GpuSnapshot {
    pub index: u32,
    pub name: String,
    pub vram_used_bytes: u64,
    pub vram_total_bytes: u64,
    pub gpu_util_pct: u32,
    pub mem_util_pct: u32,
    pub temperature_c: Option<u32>,
    pub power_watts: Option<f64>,
    pub power_limit_watts: Option<f64>,
    pub clock_graphics_mhz: Option<u32>,
    pub clock_mem_mhz: Option<u32>,
    pub fan_pct: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TelemetrySnapshot {
    pub nvml_available: bool,
    pub error: Option<String>,
    pub gpus: Vec<GpuSnapshot>,
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub cpu_util_pct: f32,
    pub timestamp_ms: u128,
}

/// Long-lived telemetry sources, initialized once and reused for every poll.
pub struct TelemetryState {
    nvml: Option<Nvml>,
    init_error: Option<String>,
    system: Mutex<System>,
}

impl TelemetryState {
    pub fn new() -> Self {
        let (nvml, init_error) = match Nvml::init() {
            Ok(n) => (Some(n), None),
            Err(e) => (None, Some(e.to_string())),
        };
        TelemetryState {
            nvml,
            init_error,
            system: Mutex::new(System::new()),
        }
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        let (ram_used_bytes, ram_total_bytes, cpu_util_pct) = self.system_metrics();
        let (gpus, error) = self.gpu_metrics();

        TelemetrySnapshot {
            nvml_available: self.nvml.is_some(),
            error,
            gpus,
            ram_used_bytes,
            ram_total_bytes,
            cpu_util_pct,
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
        }
    }

    fn system_metrics(&self) -> (u64, u64, f32) {
        let mut sys = self.system.lock().unwrap();
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        (
            sys.used_memory(),
            sys.total_memory(),
            sys.global_cpu_usage(),
        )
    }

    fn gpu_metrics(&self) -> (Vec<GpuSnapshot>, Option<String>) {
        let Some(nvml) = self.nvml.as_ref() else {
            return (Vec::new(), self.init_error.clone());
        };

        let count = match nvml.device_count() {
            Ok(c) => c,
            Err(e) => return (Vec::new(), Some(e.to_string())),
        };

        let mut gpus = Vec::new();
        for index in 0..count {
            let device = match nvml.device_by_index(index) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let mem = device.memory_info().ok();
            let util = device.utilization_rates().ok();
            gpus.push(GpuSnapshot {
                index,
                name: device.name().unwrap_or_else(|_| format!("GPU {index}")),
                vram_used_bytes: mem.as_ref().map(|m| m.used).unwrap_or(0),
                vram_total_bytes: mem.as_ref().map(|m| m.total).unwrap_or(0),
                gpu_util_pct: util.as_ref().map(|u| u.gpu).unwrap_or(0),
                mem_util_pct: util.as_ref().map(|u| u.memory).unwrap_or(0),
                temperature_c: device.temperature(TemperatureSensor::Gpu).ok(),
                // NVML reports power in milliwatts.
                power_watts: device.power_usage().ok().map(|mw| mw as f64 / 1000.0),
                power_limit_watts: device
                    .enforced_power_limit()
                    .ok()
                    .map(|mw| mw as f64 / 1000.0),
                clock_graphics_mhz: device.clock_info(Clock::Graphics).ok(),
                clock_mem_mhz: device.clock_info(Clock::Memory).ok(),
                fan_pct: device.fan_speed(0).ok(),
            });
        }
        (gpus, None)
    }
}

impl Default for TelemetryState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real live snapshot of this machine's GPU + system. Ignored by default
    /// (hardware-dependent); run with:
    ///   cargo test -- --ignored --nocapture live_snapshot
    #[test]
    #[ignore]
    fn live_snapshot() {
        let state = TelemetryState::new();
        // CPU usage needs two samples spaced apart to be meaningful.
        let _ = state.snapshot();
        std::thread::sleep(std::time::Duration::from_millis(400));
        let snap = state.snapshot();

        println!("\nnvml_available: {}", snap.nvml_available);
        if let Some(e) = &snap.error {
            println!("nvml error: {e}");
        }
        println!(
            "system: RAM {:.1}/{:.1} GB, CPU {:.1}%",
            snap.ram_used_bytes as f64 / 1e9,
            snap.ram_total_bytes as f64 / 1e9,
            snap.cpu_util_pct,
        );
        for g in &snap.gpus {
            println!(
                "GPU{} {}: VRAM {:.2}/{:.2} GB, util {}%, {}C, {} W (limit {} W), gfx {} MHz",
                g.index,
                g.name,
                g.vram_used_bytes as f64 / 1e9,
                g.vram_total_bytes as f64 / 1e9,
                g.gpu_util_pct,
                g.temperature_c.map(|t| t.to_string()).unwrap_or("?".into()),
                g.power_watts.map(|p| format!("{p:.1}")).unwrap_or("?".into()),
                g.power_limit_watts
                    .map(|p| format!("{p:.0}"))
                    .unwrap_or("?".into()),
                g.clock_graphics_mhz
                    .map(|c| c.to_string())
                    .unwrap_or("?".into()),
            );
        }
        assert!(snap.ram_total_bytes > 0, "should read system RAM");
    }
}
