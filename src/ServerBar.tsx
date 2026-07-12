import "./ServerBar.css";

export interface ServerStatus {
  running: boolean;
  health: string;
  pid: number | null;
  base_url: string | null;
  model_path: string | null;
  binary_label: string | null;
  uptime_ms: number | null;
  error: string | null;
}

function baseName(path: string | null): string {
  if (!path) return "—";
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function healthClass(health: string): string {
  switch (health) {
    case "ok":
      return "ok";
    case "loading":
    case "starting":
      return "loading";
    case "error":
      return "error";
    default:
      return "idle";
  }
}

export function ServerBar({
  status,
  onStop,
}: {
  status: ServerStatus | null;
  onStop: () => void;
}) {
  const running = status?.running ?? false;
  const health = status?.health ?? "stopped";

  return (
    <div className="server-bar">
      <span className={`health-dot ${healthClass(health)}`} />
      <div className="server-info">
        {running ? (
          <>
            <span className="server-model">{baseName(status!.model_path)}</span>
            <span className="server-meta">
              {status!.binary_label} · {health}
              {status!.base_url && ` · ${status!.base_url}`}
            </span>
          </>
        ) : status?.error ? (
          <>
            <span className="server-model">Launch failed</span>
            <span className="server-meta error" title={status.error}>
              {status.error.split("\n")[0]}
            </span>
          </>
        ) : (
          <span className="server-meta">No model loaded</span>
        )}
      </div>
      {running && (
        <button className="stop-btn" onClick={onStop}>
          Stop
        </button>
      )}
    </div>
  );
}
