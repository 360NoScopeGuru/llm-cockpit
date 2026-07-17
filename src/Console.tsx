import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ServerStatus, baseName } from "./types";

// Bottom drawer: a terminal-style chat console streaming straight from the
// running llama-server via the Rust backend (no CORS, no external browser —
// the server's root URL intentionally serves no page; this console is the UI).

interface Turn {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  meta?: string;
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

export function Console({ server }: { server: ServerStatus | null }) {
  const [open, setOpen] = useState(true);
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

  const ready = server?.running && server.health === "ok";

  useEffect(() => {
    // listen() resolves asynchronously; if this effect is torn down before the
    // promise settles (StrictMode does exactly that on mount), the handler must
    // still be unregistered — otherwise a second live listener doubles every
    // streamed token.
    let disposed = false;
    const unlistens: Array<() => void> = [];
    const track = (p: Promise<() => void>) =>
      p.then((u) => {
        if (disposed) u();
        else unlistens.push(u);
      });
    track(listen<DeltaEvent>("chat-delta", (e) => {
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
    }));
    track(listen<DoneEvent>("chat-done", (e) => {
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
            meta: e.payload.error
              ? undefined
              : `· ${e.payload.tokens} tok · ${e.payload.decode_tok_s.toFixed(1)} tok/s${
                  e.payload.stopped ? " · stopped" : ""
                }`,
          };
        }
        return next;
      });
    }));
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

  return (
    <div className={`console ${open ? "" : "closed"}`}>
      <div className="console-head" onClick={() => setOpen(!open)}>
        <span className="caret">{open ? "▼" : "▲"}</span>
        <span className="microlabel">console</span>
        {server?.running && (
          <span className="microlabel" style={{ color: "var(--teal)" }}>
            {baseName(server.model_path ?? "")}
          </span>
        )}
        <div className="right" onClick={(e) => e.stopPropagation()}>
          {server?.base_url && (
            <span className="api-chip" title="OpenAI-compatible API — point any client here. The root URL serves no page; endpoints live under /v1.">
              api {server.base_url}/v1
              <button onClick={copyApi}>{copied ? "✓" : "copy"}</button>
            </span>
          )}
          {turns.length > 0 && (
            <button onClick={() => setTurns([])} disabled={streaming}>
              CLEAR
            </button>
          )}
        </div>
      </div>

      {open && (
        <>
          <div className="params-row">
            <div className="param">
              <label>temp</label>
              <input value={temp} onChange={(e) => setTemp(e.target.value)} />
            </div>
            <div className="param">
              <label>top-k</label>
              <input value={topK} onChange={(e) => setTopK(e.target.value)} />
            </div>
            <div className="param">
              <label>top-p</label>
              <input value={topP} onChange={(e) => setTopP(e.target.value)} />
            </div>
            <div className="param">
              <label>min-p</label>
              <input value={minP} onChange={(e) => setMinP(e.target.value)} />
            </div>
            <div className="param">
              <label>max</label>
              <input value={maxTok} onChange={(e) => setMaxTok(e.target.value)} />
            </div>
            <div className="param">
              <label>system</label>
              <input
                className="wide"
                placeholder="(optional system prompt)"
                value={system}
                onChange={(e) => setSystem(e.target.value)}
              />
            </div>
          </div>

          <div className="transcript" ref={scrollRef}>
            {turns.length === 0 && (
              <div style={{ color: "var(--muted)", paddingTop: 8 }}>
                {ready
                  ? "Model ready. Say something."
                  : "No model running — ignite one in the hangar to open a channel."}
              </div>
            )}
            {turns.map((t, i) => (
              <div key={i} className={`turn ${t.role} ${t.error ? "error-turn" : ""}`}>
                {t.thinking && <span className="thinking">{t.thinking}</span>}
                {t.content}
                {t.role === "assistant" && streaming && i === turns.length - 1 && (
                  <span className="caret-blink" />
                )}
                {t.meta && <span className="meta">{t.meta}</span>}
              </div>
            ))}
          </div>

          <div className="console-input">
            <textarea
              placeholder={ready ? "prompt…" : "ignite a model first"}
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
              <button className="stop-btn" onClick={stopGen}>
                STOP
              </button>
            ) : (
              <button className="ignite" onClick={send} disabled={!ready || !input.trim()}>
                SEND
              </button>
            )}
            <span className="hint">
              enter ↵ send
              <br />
              shift+↵ newline
            </span>
          </div>
        </>
      )}
    </div>
  );
}
