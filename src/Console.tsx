import { ReactNode, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { InferenceMetrics, ServerStatus, ctxLabel } from "./types";

// The console: streaming chat wired straight to the running llama-server
// through the Rust backend (no CORS, no browser tab — the server's root URL
// intentionally serves no page; this is the UI). Owns the whole center pane:
// cold / staged / ignition / fault states, the KV containment alert banner,
// the session timeline, sampler chips and the composer. While a bench or
// suite is running, `board` replaces the transcript.

interface Turn {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  meta?: string;
  tokens?: number;
  error?: boolean;
}

interface DeltaEvent {
  id: number;
  content: string;
  reasoning: boolean;
}

interface DoneEvent {
  id: number;
  tokens: number;
  decode_tok_s: number;
  stopped: boolean;
  error: string | null;
}

export interface StagedIgnite {
  name: string;
  ngl: number;
  layers: number | null;
  ctx: number;
  busy: boolean;
  onIgnite: () => void;
}

interface ConsoleProps {
  server: ServerStatus | null;
  metrics: InferenceMetrics | null;
  ctxSize: number | null;
  modelName: string | null;
  cfgText: string | null;
  staged: StagedIgnite | null;
  board: ReactNode | null;
  kvAlert: boolean;
}

const estTokens = (s: string) => Math.max(1, Math.ceil(s.length / 4));

export function Console(p: ConsoleProps) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [system, setSystem] = useState("");
  const [temp, setTemp] = useState("0.7");
  const [topK, setTopK] = useState("40");
  const [topP, setTopP] = useState("0.95");
  const [minP, setMinP] = useState("0.05");
  const [maxTok, setMaxTok] = useState("1024");
  const [copied, setCopied] = useState(false);

  const genId = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  const server = p.server;
  const health = server?.running ? server.health : "stopped";
  const ready = !!server?.running && health === "ok";
  const igniting = !!server?.running && (health === "starting" || health === "loading");
  const fault = !!server?.error && !ready && !igniting;

  useEffect(() => {
    // listen() resolves asynchronously; if this effect is torn down before the
    // promise settles (StrictMode does exactly that on mount), the handler must
    // still be unregistered — otherwise a second live listener doubles every
    // streamed token.
    let disposed = false;
    const unlistens: Array<() => void> = [];
    const track = (pr: Promise<() => void>) =>
      pr.then((u) => {
        if (disposed) u();
        else unlistens.push(u);
      });
    track(
      listen<DeltaEvent>("chat-delta", (e) => {
        if (e.payload.id !== genId.current) return;
        setTurns((prev) => {
          const next = [...prev];
          const lastTurn = next[next.length - 1];
          if (lastTurn?.role === "assistant" && !lastTurn.meta) {
            next[next.length - 1] = e.payload.reasoning
              ? { ...lastTurn, thinking: (lastTurn.thinking ?? "") + e.payload.content }
              : { ...lastTurn, content: lastTurn.content + e.payload.content };
          }
          return next;
        });
      })
    );
    track(
      listen<DoneEvent>("chat-done", (e) => {
        if (e.payload.id !== genId.current) return;
        setStreaming(false);
        setTurns((prev) => {
          const next = [...prev];
          const lastTurn = next[next.length - 1];
          if (lastTurn?.role === "assistant") {
            next[next.length - 1] = {
              ...lastTurn,
              content: e.payload.error
                ? lastTurn.content || `⚠ ${e.payload.error}`
                : lastTurn.content,
              error: !!e.payload.error,
              tokens: e.payload.tokens,
              meta: e.payload.error
                ? undefined
                : `${e.payload.tokens} tok · ${e.payload.decode_tok_s.toFixed(1)} tok/s${
                    e.payload.stopped ? " · stopped" : ""
                  }`,
            };
          }
          return next;
        });
      })
    );
    return () => {
      disposed = true;
      unlistens.forEach((u) => u());
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns]);

  async function send() {
    const text = input.trim();
    if (!text || !ready || streaming) return;
    setInput("");
    const id = ++genId.current;
    const history = [...turns.filter((t) => !t.error), { role: "user" as const, content: text }];
    setTurns([...turns, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setStreaming(true);
    const messages = [
      ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
      ...history.map((t) => ({ role: t.role, content: t.content })),
    ];
    // NaN → undefined, but keep legitimate zeros (temp 0 = greedy decoding).
    const num = (s: string) => {
      const v = parseFloat(s);
      return Number.isFinite(v) ? v : undefined;
    };
    const int = (s: string) => {
      const v = parseInt(s, 10);
      return Number.isFinite(v) ? v : undefined;
    };
    try {
      await invoke("chat_send", {
        id,
        messages,
        params: {
          temperature: num(temp),
          top_k: int(topK),
          top_p: num(topP),
          min_p: num(minP),
          max_tokens: int(maxTok),
        },
      });
    } catch (e) {
      setStreaming(false);
      setTurns((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: `⚠ ${e}`, error: true },
      ]);
    }
  }

  async function stopGen() {
    try {
      await invoke("chat_cancel");
    } catch {
      /* ignore */
    }
  }

  async function copyApi() {
    if (!server?.base_url) return;
    await navigator.clipboard.writeText(`${server.base_url}/v1`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  const kvTokens = p.metrics?.kv_cache_tokens ?? 0;
  const kvPct = Math.round((p.metrics?.kv_cache_usage_ratio ?? 0) * 100);

  const sampler = (
    label: string,
    value: string,
    set: (v: string) => void,
    wide = false
  ) => (
    <span className="sampler">
      {label}
      <input
        className={wide ? "wide" : ""}
        value={value}
        placeholder={wide ? "(none)" : ""}
        onChange={(e) => set(e.target.value)}
      />
    </span>
  );

  return (
    <div className="console">
      <div className="console-head">
        <span className="lbl">Console</span>
        {p.modelName && <span className="console-model">{p.modelName}</span>}
        {p.cfgText && <span className="console-cfg">{p.cfgText}</span>}
        <span className="spacer" />
        {server?.base_url ? (
          <span
            className="api-chip"
            title="OpenAI-compatible API — point any client here. The root URL serves no page; endpoints live under /v1."
          >
            {server.base_url}/v1
            <button onClick={copyApi}>{copied ? "✓" : "⧉"}</button>
          </span>
        ) : (
          <span className="api-chip offline">api offline</span>
        )}
        {turns.length > 0 && (
          <button onClick={() => setTurns([])} disabled={streaming}>
            Clear
          </button>
        )}
      </div>

      {p.kvAlert && ready && (
        <div className="alert-banner">
          <span className="alert-dot" />
          <span className="alert-title">CONTAINMENT NEAR CAPACITY — KV CACHE {kvPct}%</span>
          <span className="alert-sub">
            {kvTokens.toLocaleString()}
            {p.ctxSize ? ` / ${p.ctxSize.toLocaleString()}` : ""} tok · context is nearly full
          </span>
          <span className="spacer" />
          <button onClick={() => setTurns([])} disabled={streaming}>
            Clear Session
          </button>
          {streaming && (
            <button className="danger" onClick={stopGen}>
              ■ Stop
            </button>
          )}
        </div>
      )}

      {p.board ? (
        p.board
      ) : fault ? (
        <div className="console-state">
          <div className="state-box" style={{ maxWidth: 680 }}>
            <div className="state-title" style={{ color: "var(--danger)" }}>
              FAULT
            </div>
            <div className="fault-box">{server?.error}</div>
            <div className="state-hint">the reactor scrammed — fix the cause and ignite again</div>
          </div>
        </div>
      ) : igniting ? (
        <div className="console-state">
          <div className="state-box">
            <div className="state-title ignite">IGNITION</div>
            {p.modelName && <div className="state-sub">{p.modelName}</div>}
            <div className="ignition-steps">
              <span className="step done">
                <span className="mark">✓</span>
                <span>spawn llama-server</span>
              </span>
              <span className={`step ${health === "loading" ? "done" : "active"}`}>
                <span className="mark">{health === "loading" ? "✓" : "▶"}</span>
                <span>probe /health</span>
              </span>
              <span className={`step ${health === "loading" ? "active" : "todo"}`}>
                <span className="mark">{health === "loading" ? "▶" : "·"}</span>
                <span>load weights → VRAM · rod bank filling →</span>
              </span>
              <span className="step todo">
                <span className="mark">·</span>
                <span>warmup pass + bind api</span>
              </span>
            </div>
            <div className="state-hint">
              elapsed {((server?.uptime_ms ?? 0) / 1000).toFixed(0)} s · big models can take a
              minute or two
            </div>
          </div>
        </div>
      ) : ready ? (
        <div className="transcript" ref={scrollRef}>
          {turns.length === 0 && (
            <div style={{ color: "var(--low)", paddingTop: 8 }}>
              Reactor live. Message it below — or point any OpenAI client at the api endpoint
              above.
            </div>
          )}
          {turns.map((t, i) => (
            <div key={i} className={`turn ${t.role} ${t.error ? "error-turn" : ""}`}>
              <div className={`turn-label ${t.role}`}>
                {t.role === "user" ? "You" : p.modelName ?? "Model"}
              </div>
              {t.thinking && <span className="thinking">{t.thinking}</span>}
              <div className="turn-body">
                {t.content}
                {t.role === "assistant" && streaming && i === turns.length - 1 && (
                  <span className="caret-blink" />
                )}
              </div>
              {t.role === "assistant" && streaming && i === turns.length - 1 && p.metrics && (
                <div className="turn-meta">
                  streaming · {p.metrics.predicted_tokens_per_sec.toFixed(1)} tok/s
                </div>
              )}
              {t.meta && <div className="turn-meta">{t.meta}</div>}
            </div>
          ))}
        </div>
      ) : p.staged ? (
        <div className="console-state">
          <div className="state-box">
            <div className="state-sub mono">{p.staged.name} staged</div>
            <button className="ignite-cta" disabled={p.staged.busy} onClick={p.staged.onIgnite}>
              IGNITE · {p.staged.ngl}
              {p.staged.layers ? `/${p.staged.layers}` : ""} LAYERS · {ctxLabel(p.staged.ctx)} CTX
            </button>
            <div className="state-hint">launches llama-server at the recommended config</div>
          </div>
        </div>
      ) : (
        <div className="console-state">
          <div className="state-box">
            <div className="state-title">CONTAINMENT COLD</div>
            <div className="state-sub">
              No model loaded. Select a fuel rod from the library, review its fit, then{" "}
              <span style={{ color: "var(--plasma)" }}>IGNITE</span>.
            </div>
            <div className="state-hint">hover a model to preview its VRAM footprint</div>
          </div>
        </div>
      )}

      {!p.board && ready && p.ctxSize && (turns.length > 0 || kvTokens > 0) && (
        <div className="timeline">
          <div className="timeline-track">
            {turns.map((t, i) => {
              const tok = t.tokens ?? estTokens(t.content + (t.thinking ?? ""));
              const w = Math.max(0.6, (tok / p.ctxSize!) * 100);
              const live = streaming && i === turns.length - 1 && t.role === "assistant";
              return (
                <span
                  key={i}
                  className={`blk ${t.role} ${live ? "live" : ""}`}
                  style={{ width: `${w}%` }}
                  title={`${t.role} · ~${tok} tok`}
                />
              );
            })}
            <span className="free" />
            <span className="ceiling" />
          </div>
          <div className="timeline-foot">
            <span>session timeline · block width = tokens</span>
            <span className="spacer" />
            <span>
              {kvTokens.toLocaleString()} / {p.ctxSize.toLocaleString()} tok
              {p.kvAlert && <span style={{ color: "var(--danger)" }}> · ceiling</span>}
            </span>
          </div>
        </div>
      )}

      {!p.board && (
        <>
          <div className="sampler-row">
            {sampler("temp", temp, setTemp)}
            {sampler("top-k", topK, setTopK)}
            {sampler("top-p", topP, setTopP)}
            {sampler("min-p", minP, setMinP)}
            {sampler("max", maxTok, setMaxTok)}
            {sampler("sys", system, setSystem, true)}
          </div>
          <div className="composer">
            <textarea
              placeholder={ready ? "message the reactor…" : "ignite a model first"}
              value={input}
              disabled={!ready || streaming}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            {streaming ? (
              <button className="send danger" onClick={stopGen}>
                ■ Stop
              </button>
            ) : (
              <button className="send primary" onClick={send} disabled={!ready || !input.trim()}>
                Send
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
