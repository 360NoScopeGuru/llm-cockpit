import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { TelemetryCockpit } from "./Telemetry";
import "./App.css";

// Mirrors the serde output of the Rust `scanner`/`gguf` modules (snake_case).
interface GgufMetadata {
  version: number;
  tensor_count: number;
  architecture: string | null;
  name: string | null;
  quant_label: string | null;
  context_length: number | null;
  block_count: number | null;
  embedding_length: number | null;
  parameter_count: number | null;
  size_label: string | null;
  split_count: number | null;
}

interface ModelEntry {
  path: string;
  file_name: string;
  size_bytes: number;
  source: string;
  is_shard_continuation: boolean;
  shard_total: number | null;
  metadata: GgufMetadata | null;
  parse_error: string | null;
}

interface ScanRoot {
  path: string;
  source: string;
  exists: boolean;
}

function formatBytes(n: number): string {
  if (n <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatCtx(n: number | null): string {
  if (!n) return "—";
  if (n >= 1024) return `${Math.round(n / 1024)}K`;
  return String(n);
}

function sourceLabel(source: string): string {
  switch (source) {
    case "huggingface":
      return "HF";
    case "lm-studio":
      return "LM Studio";
    default:
      return "Folder";
  }
}

function App() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [roots, setRoots] = useState<ScanRoot[]>([]);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function rescan() {
    setScanning(true);
    setError(null);
    try {
      const [rootList, modelList] = await Promise.all([
        invoke<ScanRoot[]>("scan_roots"),
        invoke<ModelEntry[]>("scan_models", { extraDirs: [] }),
      ]);
      setRoots(rootList);
      setModels(modelList);
    } catch (e) {
      setError(String(e));
    } finally {
      setScanning(false);
    }
  }

  useEffect(() => {
    rescan();
  }, []);

  // Primary list hides continuation shards of split models.
  const primary = models.filter((m) => !m.is_shard_continuation);

  return (
    <main className="app">
      <TelemetryCockpit />

      <header className="app-header">
        <div>
          <h1>Model Library</h1>
          <p className="subtitle">
            {scanning
              ? "Scanning caches…"
              : `${primary.length} model${primary.length === 1 ? "" : "s"} found`}
          </p>
        </div>
        <button onClick={rescan} disabled={scanning}>
          {scanning ? "Scanning…" : "Rescan"}
        </button>
      </header>

      <section className="roots">
        {roots.map((r) => (
          <span key={r.path} className={`root-chip ${r.exists ? "" : "missing"}`}>
            {sourceLabel(r.source)}
            <code>{r.path}</code>
            {!r.exists && <em>not found</em>}
          </span>
        ))}
      </section>

      {error && <div className="error">Scan failed: {error}</div>}

      {!scanning && primary.length === 0 && !error && (
        <div className="empty">
          No GGUF models found in the default caches. Point the app at a folder to
          get started.
        </div>
      )}

      {primary.length > 0 && (
        <table className="model-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Arch</th>
              <th>Quant</th>
              <th>Ctx</th>
              <th>Params</th>
              <th>Size</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {primary.map((m) => {
              const md = m.metadata;
              return (
                <tr key={m.path} title={m.path}>
                  <td className="model-name">
                    {md?.name ?? m.file_name}
                    {m.shard_total && (
                      <span className="badge">{m.shard_total} shards</span>
                    )}
                    {m.parse_error && (
                      <span className="badge warn" title={m.parse_error}>
                        unreadable
                      </span>
                    )}
                  </td>
                  <td>{md?.architecture ?? "—"}</td>
                  <td>{md?.quant_label ?? "—"}</td>
                  <td>{formatCtx(md?.context_length ?? null)}</td>
                  <td>{md?.size_label ?? "—"}</td>
                  <td>{formatBytes(m.size_bytes)}</td>
                  <td>
                    <span className="src">{sourceLabel(m.source)}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </main>
  );
}

export default App;
