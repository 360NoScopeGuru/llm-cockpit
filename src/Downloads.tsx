import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { BYTES_PER_GB, gb } from "./types";

// Model acquisition, in-app. Three sources so nobody has to install LM Studio
// or Ollama just to get weights:
//   HUB    — search Hugging Face, see every quant with a fit verdict, download
//   URL    — paste a direct .gguf link
//   OLLAMA — drive `ollama pull` (only shown when Ollama is actually installed)
//
// Everything lands as a plain .gguf in a folder the scanner already watches.

interface HfModel {
  id: string;
  downloads: number;
  likes: number;
  gated: boolean;
}

interface HfFile {
  name: string;
  size_bytes: number;
  quant: string | null;
}

interface ProgressEvent {
  id: number;
  received: number;
  total: number;
  bytes_per_sec: number;
  done: boolean;
  cancelled: boolean;
  path: string | null;
  error: string | null;
}

interface Job {
  id: number;
  label: string;
  received: number;
  total: number;
  rate: number;
  done: boolean;
  cancelled: boolean;
  error: string | null;
}

type Source = "huggingface" | "url" | "ollama";

interface DownloadsProps {
  /// Total VRAM in bytes, for the pre-download fit verdict.
  vramTotal: number | null;
  onClose: () => void;
  /// Called when a download finishes so the library can rescan.
  onDownloaded: () => void;
}

/// Overhead beyond raw weights: llama.cpp compute buffers plus a modest KV
/// cache. The real estimator reads the model's attention shape, but that needs
/// the file — before downloading, this is the honest approximation.
const RESERVE_BYTES = 1.5 * BYTES_PER_GB;
/// The estimator holds back 10% of VRAM for the desktop and driver; match it.
const VRAM_BUDGET = 0.9;

function verdict(size: number, vramTotal: number | null) {
  if (!vramTotal) return null;
  const budget = vramTotal * VRAM_BUDGET;
  if (size + RESERVE_BYTES <= budget) {
    return { cls: "full", text: `● FITS · ${gb(budget - size - RESERVE_BYTES)} GB spare` };
  }
  if (size <= budget) return { cls: "partial", text: "◐ TIGHT · partial offload" };
  return { cls: "none", text: "○ TOO BIG for VRAM" };
}

function rate(bps: number) {
  if (bps <= 0) return "";
  return bps >= 1e6 ? `${(bps / 1e6).toFixed(1)} MB/s` : `${Math.round(bps / 1e3)} kB/s`;
}

function eta(j: Job) {
  if (!j.total || j.rate <= 0) return "";
  const s = Math.max(0, (j.total - j.received) / j.rate);
  if (s > 3600) return `${(s / 3600).toFixed(1)} h left`;
  if (s > 60) return `${Math.round(s / 60)} min left`;
  return `${Math.round(s)} s left`;
}

export function Downloads(p: DownloadsProps) {
  const [sources, setSources] = useState<Record<string, boolean>>({
    huggingface: true,
    url: true,
    ollama: false,
  });
  const [source, setSource] = useState<Source>("huggingface");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HfModel[] | null>(null);
  const [openRepo, setOpenRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<HfFile[] | null>(null);
  const [urlText, setUrlText] = useState("");
  const [ollamaText, setOllamaText] = useState("");
  const [dir, setDir] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const nextId = useRef(1);

  useEffect(() => {
    invoke<Record<string, boolean>>("download_sources").then(setSources).catch(() => {});
    invoke<string>("downloads_dir").then(setDir).catch(() => {});
  }, []);

  // The close button advertises Esc, so Esc has to work.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") p.onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [p.onClose]);

  // StrictMode-safe: the effect can run twice, so ignore a listener that
  // resolves after teardown.
  useEffect(() => {
    let disposed = false;
    let un: (() => void) | undefined;
    listen<ProgressEvent>("download-progress", (e) => {
      const ev = e.payload;
      setJobs((prev) =>
        prev.map((j) =>
          j.id !== ev.id
            ? j
            : {
                ...j,
                received: ev.received || j.received,
                total: ev.total || j.total,
                rate: ev.bytes_per_sec || j.rate,
                done: ev.done,
                cancelled: ev.cancelled,
                error: ev.error,
              }
        )
      );
      if (ev.done && !ev.cancelled && !ev.error) p.onDownloaded();
    }).then((u) => (disposed ? u() : (un = u)));
    return () => {
      disposed = true;
      un?.();
    };
  }, []);

  async function runSearch() {
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    setOpenRepo(null);
    setFiles(null);
    try {
      setResults(await invoke<HfModel[]>("hf_search", { query: query.trim() }));
    } catch (e) {
      setError(String(e));
      setResults([]);
    } finally {
      setBusy(false);
    }
  }

  async function openFiles(repo: string) {
    if (openRepo === repo) {
      setOpenRepo(null);
      return;
    }
    setOpenRepo(repo);
    setFiles(null);
    setError(null);
    try {
      setFiles(await invoke<HfFile[]>("hf_files", { repo }));
    } catch (e) {
      setError(String(e));
      setFiles([]);
    }
  }

  function track(label: string, total = 0) {
    const id = nextId.current++;
    setJobs((prev) => [
      { id, label, received: 0, total, rate: 0, done: false, cancelled: false, error: null },
      ...prev,
    ]);
    return id;
  }

  async function start(args: Record<string, unknown>, label: string, total = 0) {
    const id = track(label, total);
    try {
      await invoke("download_start", { id, ...args });
    } catch (e) {
      setJobs((prev) =>
        prev.map((j) => (j.id === id ? { ...j, done: true, error: String(e) } : j))
      );
    }
  }

  const cancel = (id: number) => invoke("download_cancel", { id }).catch(() => {});

  const tab = (id: Source, label: string, enabled: boolean) =>
    enabled ? (
      <button
        key={id}
        className={`tab-btn ${source === id ? "active" : ""}`}
        onClick={() => setSource(id)}
      >
        {label}
      </button>
    ) : null;

  return (
    <div className="board downloads">
      <div className="board-head">
        <span className="lbl">Get Models</span>
        <span className="tab-bar" style={{ marginLeft: 12 }}>
          {tab("huggingface", "HUGGING FACE", sources.huggingface)}
          {tab("url", "URL", sources.url)}
          {tab("ollama", "OLLAMA", sources.ollama)}
        </span>
        <span className="spacer" />
        {dir && (
          <span className="dl-dir" title={dir}>
            → {dir}
          </span>
        )}
        <button className="dl-close" onClick={p.onClose} title="close (Esc)">
          ✕ Close
        </button>
      </div>

      {error && <div className="dl-error">⚠ {error}</div>}

      {source === "huggingface" && (
        <div className="dl-body">
          <div className="dl-search">
            <input
              value={query}
              placeholder="search Hugging Face for GGUF models — e.g. qwen3 coder"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runSearch()}
            />
            <button onClick={runSearch} disabled={busy || !query.trim()}>
              {busy ? "…" : "Search"}
            </button>
          </div>

          {results?.length === 0 && !busy && (
            <div className="dl-empty">No GGUF repos matched that.</div>
          )}

          {(results ?? []).map((m) => (
            <div key={m.id}>
              <div className="dl-repo" onClick={() => openFiles(m.id)}>
                <span className="dl-caret">{openRepo === m.id ? "▾" : "▸"}</span>
                <span className="dl-repo-id">{m.id}</span>
                {m.gated && <span className="chip gated">GATED</span>}
                <span className="spacer" />
                <span className="dl-meta">
                  ↓ {m.downloads.toLocaleString()} · ♥ {m.likes.toLocaleString()}
                </span>
              </div>
              {openRepo === m.id && (
                <div className="dl-files">
                  {files === null && <div className="dl-empty">reading files…</div>}
                  {files?.length === 0 && (
                    <div className="dl-empty">No .gguf files in this repo.</div>
                  )}
                  {(files ?? []).map((f) => {
                    const v = verdict(f.size_bytes, p.vramTotal);
                    return (
                      <div key={f.name} className="dl-file">
                        <span className="dl-quant">{f.quant ?? "—"}</span>
                        <span className="dl-name" title={f.name}>
                          {f.name}
                        </span>
                        <span className="dl-size">{gb(f.size_bytes, 2)} GB</span>
                        {v && <span className={`verdict ${v.cls}`}>{v.text}</span>}
                        <button
                          onClick={() =>
                            start(
                              { source: "huggingface", repo: m.id, file: f.name },
                              f.name,
                              f.size_bytes
                            )
                          }
                        >
                          Download
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {source === "url" && (
        <div className="dl-body">
          <div className="dl-search">
            <input
              value={urlText}
              placeholder="https://…/model-Q4_K_M.gguf"
              onChange={(e) => setUrlText(e.target.value)}
            />
            <button
              disabled={!urlText.trim().startsWith("https://")}
              onClick={() => {
                const u = urlText.trim();
                start({ source: "url", url: u }, u.split("/").pop() ?? u);
                setUrlText("");
              }}
            >
              Download
            </button>
          </div>
          <div className="dl-hint">
            Direct link to a .gguf file. Downloads resume if interrupted.
          </div>
        </div>
      )}

      {source === "ollama" && (
        <div className="dl-body">
          <div className="dl-search">
            <input
              value={ollamaText}
              placeholder="model name — e.g. qwen3:14b"
              onChange={(e) => setOllamaText(e.target.value)}
            />
            <button
              disabled={!ollamaText.trim()}
              onClick={() => {
                const m = ollamaText.trim();
                start({ source: "ollama", model: m }, `ollama pull ${m}`);
                setOllamaText("");
              }}
            >
              Pull
            </button>
          </div>
          <div className="dl-hint">
            Runs <code>ollama pull</code>. The blobs it writes are already scanned,
            so the model appears in the library when it finishes. Progress is
            reported as running/finished only — Ollama does not expose byte counts here.
          </div>
        </div>
      )}

      {jobs.length > 0 && (
        <div className="dl-jobs">
          <div className="lbl faint">Transfers</div>
          {jobs.map((j) => {
            const pct = j.total ? Math.min(100, (j.received / j.total) * 100) : 0;
            return (
              <div key={j.id} className="dl-job">
                <div className="dl-job-head">
                  <span className="dl-name">{j.label}</span>
                  <span className="spacer" />
                  {j.error ? (
                    <span className="dl-fail">⚠ {j.error}</span>
                  ) : j.cancelled ? (
                    <span className="dl-meta">cancelled · partial kept, retry resumes</span>
                  ) : j.done ? (
                    <span className="dl-ok">✓ done</span>
                  ) : (
                    <span className="dl-meta">
                      {j.total
                        ? `${gb(j.received, 2)} / ${gb(j.total, 2)} GB`
                        : "working…"}{" "}
                      {rate(j.rate)} {eta(j)}
                    </span>
                  )}
                  {!j.done && <button onClick={() => cancel(j.id)}>Stop</button>}
                </div>
                {!j.done && (
                  <div className="dl-bar">
                    <span style={{ width: `${pct}%` }} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
