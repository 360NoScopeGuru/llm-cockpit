import "./Benchmark.css";

export interface BenchResult {
  n_gpu_layers: number;
  ctx_size: number;
  loaded: boolean;
  load_ms: number;
  prefill_tok_s: number;
  decode_tok_s: number;
  peak_vram_bytes: number;
  error: string | null;
}

function gb(bytes: number): string {
  return bytes > 0 ? `${(bytes / 1e9).toFixed(2)} GB` : "—";
}

export function BenchmarkPanel({
  name,
  results,
  running,
  expected,
  onDismiss,
}: {
  name: string;
  results: BenchResult[];
  running: boolean;
  expected: number;
  onDismiss: () => void;
}) {
  const loaded = results.filter((r) => r.loaded && r.decode_tok_s > 0);
  const bestDecode = loaded.reduce((m, r) => Math.max(m, r.decode_tok_s), 0);

  return (
    <div className="benchmark">
      <div className="bm-head">
        <div>
          <span className="bm-title">Measured benchmark</span>
          <span className="bm-sub">{name}</span>
        </div>
        <button className="bm-dismiss" onClick={onDismiss} disabled={running}>
          ✕
        </button>
      </div>

      <table className="bm-table">
        <thead>
          <tr>
            <th>GPU layers</th>
            <th>Context</th>
            <th>Load</th>
            <th>Prefill</th>
            <th>Decode</th>
            <th>Peak VRAM</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr
              key={i}
              className={
                r.decode_tok_s === bestDecode && bestDecode > 0 ? "best" : ""
              }
            >
              <td>{r.n_gpu_layers}</td>
              <td>{r.ctx_size >= 1024 ? `${Math.round(r.ctx_size / 1024)}K` : r.ctx_size}</td>
              {r.loaded ? (
                <>
                  <td>{(r.load_ms / 1000).toFixed(1)}s</td>
                  <td>{r.prefill_tok_s.toFixed(0)} tok/s</td>
                  <td className="decode">{r.decode_tok_s.toFixed(1)} tok/s</td>
                  <td>{gb(r.peak_vram_bytes)}</td>
                </>
              ) : (
                <td colSpan={4} className="bm-error" title={r.error ?? ""}>
                  failed: {r.error ?? "unknown"}
                </td>
              )}
            </tr>
          ))}
          {running &&
            Array.from({ length: Math.max(0, expected - results.length) }).map(
              (_, i) => (
                <tr key={`pending-${i}`} className="pending">
                  <td colSpan={6}>running…</td>
                </tr>
              )
            )}
        </tbody>
      </table>

      {loaded.length > 1 && (
        <p className="bm-note">
          Best decode: <strong>{bestDecode.toFixed(1)} tok/s</strong> — measured on
          your GPU, not estimated.
        </p>
      )}
    </div>
  );
}
