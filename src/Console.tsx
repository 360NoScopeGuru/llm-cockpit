import { ReactNode, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Markdown } from "./Markdown";
import { InferenceMetrics, ServerStatus, baseName, ctxLabel } from "./types";

// The console: streaming chat wired straight to the running llama-server
// through the Rust backend. Owns the whole center pane: cold / staged /
// ignition / fault states, the KV containment alert banner, the session
// timeline, sampler chips and the composer. While a bench or suite is
// running, `board` replaces the transcript.
//
// AGENT MODE: when armed (and a workspace folder is granted), a system
// prompt teaches the model a one-tool-per-reply protocol. Replies ending in
// a ```tool fenced JSON block are parsed here; reads (list_dir/read_file)
// execute immediately, writes and commands wait for an explicit APPROVE
// click. Results are fed back as the next message and the loop continues
// until the model answers without a tool block.

interface Turn {
  role: "user" | "assistant";
  kind?: "chat" | "tool-result";
  toolName?: string;
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
  workspace: string | null;
  onPickWorkspace: () => void;
}

interface ToolCall {
  tool: "list_dir" | "read_file" | "write_file" | "run_command";
  args: Record<string, string>;
}

const TOOL_NAMES = new Set(["list_dir", "read_file", "write_file", "run_command"]);
const MAX_TOOL_ROUNDS = 24;

const estTokens = (s: string) => Math.max(1, Math.ceil(s.length / 4));

const agentPrompt = (root: string) => `You are Tokamak Agent, a coding agent with tool access to the user's workspace folder on their Windows machine: ${root}

To use a tool, end your reply with exactly one fenced block in this format:
\`\`\`tool
{"tool": "list_dir", "args": {"path": "."}}
\`\`\`

Tools:
- list_dir {"path": "."} — list files and folders
- read_file {"path": "relative\\file.txt"} — read a text file
- write_file {"path": "relative\\file.txt", "content": "..."} — create or overwrite a file (the user must approve)
- run_command {"command": "..."} — run a PowerShell command in the workspace (the user must approve)

Rules:
- Paths are relative to the workspace; you cannot access anything outside it.
- Make at most ONE tool call per reply. After the tool block, stop and wait — the result arrives in the next message tagged [tool result].
- Work step by step: inspect before you edit, verify after you change.
- When the task is done (or no tool is needed), reply normally with no tool block.`;

function parseToolCall(content: string): ToolCall | null {
  const matches = [...content.matchAll(/```tool\s*\n?([\s\S]*?)```/g)];
  if (matches.length === 0) return null;
  try {
    const obj = JSON.parse(matches[matches.length - 1][1]);
    if (typeof obj?.tool !== "string" || !TOOL_NAMES.has(obj.tool)) return null;
    return { tool: obj.tool, args: obj.args ?? {} };
  } catch {
    return null;
  }
}

function stripToolBlock(content: string): string {
  return content.replace(/```tool\s*\n?[\s\S]*?```\s*$/, "").trimEnd();
}

export function Console(p: ConsoleProps) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [system, setSystem] = useState("");
  const [temp, setTemp] = useState("0.7");
  const [topK, setTopK] = useState("40");
  const [topP, setTopP] = useState("0.95");
  const [minP, setMinP] = useState("0.05");
  const [maxTok, setMaxTok] = useState("2048");
  const [copied, setCopied] = useState(false);
  const [agentOn, setAgentOn] = useState(false);
  const [pendingTool, setPendingTool] = useState<ToolCall | null>(null);
  const [toolBusy, setToolBusy] = useState(false);

  const genId = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const turnsRef = useRef<Turn[]>([]);
  const agentRef = useRef(false);
  const wsRef = useRef<string | null>(null);
  const systemRef = useRef("");
  const samplerRef = useRef({ temp: "0.7", topK: "40", topP: "0.95", minP: "0.05", maxTok: "2048" });
  const roundsRef = useRef(0);

  turnsRef.current = turns;
  agentRef.current = agentOn;
  wsRef.current = p.workspace;
  systemRef.current = system;
  samplerRef.current = { temp, topK, topP, minP, maxTok };

  const server = p.server;
  const health = server?.running ? server.health : "stopped";
  const ready = !!server?.running && health === "ok";
  const igniting = !!server?.running && (health === "starting" || health === "loading");
  const fault = !!server?.error && !ready && !igniting;

  // ---- message assembly ----

  function buildMessages(allTurns: Turn[]) {
    const sysParts: string[] = [];
    if (agentRef.current && wsRef.current) sysParts.push(agentPrompt(wsRef.current));
    if (systemRef.current.trim()) sysParts.push(systemRef.current.trim());
    const messages: { role: string; content: string }[] = sysParts.length
      ? [{ role: "system", content: sysParts.join("\n\n") }]
      : [];
    for (const t of allTurns) {
      if (t.error || (t.role === "assistant" && !t.content && !t.meta)) continue;
      messages.push({
        role: t.role,
        content:
          t.kind === "tool-result" ? `[tool result: ${t.toolName}]\n${t.content}` : t.content,
      });
    }
    return messages;
  }

  async function dispatch(allTurns: Turn[]) {
    const id = ++genId.current;
    setStreaming(true);
    const s = samplerRef.current;
    const num = (v: string) => {
      const n = parseFloat(v);
      return Number.isFinite(n) ? n : undefined;
    };
    const int = (v: string) => {
      const n = parseInt(v, 10);
      return Number.isFinite(n) ? n : undefined;
    };
    try {
      await invoke("chat_send", {
        id,
        messages: buildMessages(allTurns),
        params: {
          temperature: num(s.temp),
          top_k: int(s.topK),
          top_p: num(s.topP),
          min_p: num(s.minP),
          max_tokens: int(s.maxTok),
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

  // ---- agent tool loop ----

  async function execTool(call: ToolCall) {
    const root = wsRef.current;
    if (!root) return;
    setPendingTool(null);
    setToolBusy(true);
    let result: string;
    try {
      if (call.tool === "list_dir") {
        const entries = await invoke<{ name: string; is_dir: boolean; size_bytes: number }[]>(
          "agent_list_dir",
          { root, path: call.args.path ?? "." }
        );
        result =
          entries
            .map(
              (e) =>
                `${e.is_dir ? "dir " : "file"}  ${e.name}${
                  e.is_dir ? "" : `  (${e.size_bytes.toLocaleString()} B)`
                }`
            )
            .join("\n") || "(empty directory)";
      } else if (call.tool === "read_file") {
        const r = await invoke<{ content: string; size_bytes: number; truncated: boolean }>(
          "agent_read_file",
          { root, path: call.args.path ?? "" }
        );
        result = r.truncated
          ? `${r.content}\n…[truncated — file is ${r.size_bytes.toLocaleString()} bytes]`
          : r.content;
      } else if (call.tool === "write_file") {
        result = await invoke<string>("agent_write_file", {
          root,
          path: call.args.path ?? "",
          content: call.args.content ?? "",
        });
      } else {
        const r = await invoke<{
          stdout: string;
          stderr: string;
          exit_code: number | null;
          timed_out: boolean;
        }>("agent_run_command", { root, command: call.args.command ?? "" });
        result = [
          `exit: ${r.timed_out ? "timed out (120 s)" : (r.exit_code ?? "unknown")}`,
          r.stdout ? `stdout:\n${r.stdout}` : "stdout: (empty)",
          r.stderr ? `stderr:\n${r.stderr}` : "",
        ]
          .filter(Boolean)
          .join("\n");
      }
    } catch (e) {
      result = `[tool error] ${e}`;
    }
    setToolBusy(false);
    continueWith({
      role: "user",
      kind: "tool-result",
      toolName: call.tool,
      content: result,
    });
  }

  function denyTool(call: ToolCall) {
    setPendingTool(null);
    continueWith({
      role: "user",
      kind: "tool-result",
      toolName: call.tool,
      content: "The user DENIED this tool call. Do not retry it; ask them or take another path.",
    });
  }

  function continueWith(resultTurn: Turn) {
    const next: Turn[] = [...turnsRef.current, resultTurn, { role: "assistant", content: "" }];
    setTurns(next);
    dispatch(next);
  }

  function maybeRunToolLoop() {
    if (!agentRef.current || !wsRef.current) return;
    const t = turnsRef.current;
    const last = t[t.length - 1];
    if (!last || last.role !== "assistant" || last.error || !last.content) return;
    const call = parseToolCall(last.content);
    if (!call) {
      roundsRef.current = 0;
      return;
    }
    if (roundsRef.current >= MAX_TOOL_ROUNDS) {
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠ agent stopped after ${MAX_TOOL_ROUNDS} tool rounds — send a message to continue`,
          error: true,
        },
      ]);
      roundsRef.current = 0;
      return;
    }
    roundsRef.current += 1;
    if (call.tool === "list_dir" || call.tool === "read_file") {
      execTool(call);
    } else {
      setPendingTool(call);
    }
  }

  // ---- stream listeners ----

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
        // Agent loop: inspect the finished reply for a tool call. Deferred a
        // tick so turnsRef reflects the update above.
        if (!e.payload.error && !e.payload.stopped) {
          setTimeout(maybeRunToolLoop, 30);
        } else {
          roundsRef.current = 0;
        }
      })
    );
    return () => {
      disposed = true;
      unlistens.forEach((u) => u());
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns, pendingTool]);

  // ---- actions ----

  async function send() {
    const text = input.trim();
    if (!text || !ready || streaming || toolBusy || pendingTool) return;
    setInput("");
    roundsRef.current = 0;
    const next: Turn[] = [
      ...turns,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ];
    setTurns(next);
    dispatch(next);
  }

  async function stopGen() {
    try {
      await invoke("chat_cancel");
    } catch {
      /* ignore */
    }
    roundsRef.current = 0;
    setPendingTool(null);
  }

  function clearSession() {
    setTurns([]);
    setPendingTool(null);
    roundsRef.current = 0;
  }

  async function copyApi() {
    if (!server?.base_url) return;
    await navigator.clipboard.writeText(`${server.base_url}/v1`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  function toggleAgent() {
    if (!agentOn && !p.workspace) {
      p.onPickWorkspace();
    }
    setAgentOn(!agentOn);
  }

  const kvTokens = p.metrics?.kv_cache_tokens ?? 0;
  const kvPct = Math.round((p.metrics?.kv_cache_usage_ratio ?? 0) * 100);
  const busyLoop = streaming || toolBusy || !!pendingTool;

  const sampler = (label: string, value: string, set: (v: string) => void, wide = false) => (
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

  // ---- turn rendering ----

  function renderTurn(t: Turn, i: number) {
    const isLast = i === turns.length - 1;
    if (t.kind === "tool-result") {
      return (
        <details key={i} className="tool-card result">
          <summary>
            <span className="tool-tag">⚙ {t.toolName}</span> result ·{" "}
            {t.content.split("\n").length} lines
          </summary>
          <pre>{t.content}</pre>
        </details>
      );
    }
    const call = t.role === "assistant" && !streaming ? parseToolCall(t.content) : null;
    const body = call ? stripToolBlock(t.content) : t.content;
    return (
      <div key={i} className={`turn ${t.role} ${t.error ? "error-turn" : ""}`}>
        <div className={`turn-label ${t.role}`}>
          {t.role === "user" ? "You" : p.modelName ?? "Model"}
        </div>
        {t.thinking && (
          <details className="thinking-box" open={isLast && streaming && !t.content}>
            <summary>thinking · ~{estTokens(t.thinking)} tok</summary>
            <div className="thinking">{t.thinking}</div>
          </details>
        )}
        <div className="turn-body">
          {t.role === "assistant" ? <Markdown text={body} /> : body}
          {t.role === "assistant" && streaming && isLast && <span className="caret-blink" />}
        </div>
        {call && (
          <div className="tool-card">
            <div className="tool-head">
              <span className="tool-tag">⚙ {call.tool}</span>
              <span className="tool-args">
                {call.tool === "run_command" ? call.args.command : call.args.path}
              </span>
            </div>
          </div>
        )}
        {t.role === "assistant" && streaming && isLast && p.metrics && (
          <div className="turn-meta">
            streaming · {p.metrics.predicted_tokens_per_sec.toFixed(1)} tok/s
          </div>
        )}
        {t.meta && <div className="turn-meta">{t.meta}</div>}
      </div>
    );
  }

  return (
    <div className="console">
      <div className="console-head">
        <span className="lbl">Console</span>
        {p.modelName && <span className="console-model">{p.modelName}</span>}
        {p.cfgText && <span className="console-cfg">{p.cfgText}</span>}
        <span className="spacer" />
        <button
          className={agentOn ? "primary" : ""}
          onClick={toggleAgent}
          title="Agent mode: the model can list/read files freely and write files or run commands with your approval, inside the workspace folder"
        >
          Agent {agentOn ? "On" : "Off"}
        </button>
        {agentOn && (
          <button onClick={p.onPickWorkspace} title={p.workspace ?? "pick a workspace folder"}>
            ⌂ {p.workspace ? baseName(p.workspace) : "pick workspace"}
          </button>
        )}
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
          <button onClick={clearSession} disabled={streaming}>
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
          <button onClick={clearSession} disabled={streaming}>
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
            <div className="transcript-empty">
              Reactor live. Message it below — or arm{" "}
              <span style={{ color: "var(--plasma)" }}>AGENT</span> mode to let it read and write
              code in a workspace folder, Claude Code style.
            </div>
          )}
          {turns.map(renderTurn)}
          {toolBusy && (
            <div className="tool-card running-tool">
              <span className="tool-tag">⚙ running tool…</span>
            </div>
          )}
          {pendingTool && (
            <div className="tool-card approve">
              <div className="tool-head">
                <span className="tool-tag danger">
                  ⚠ {pendingTool.tool === "run_command" ? "RUN COMMAND" : "WRITE FILE"}
                </span>
                <span className="tool-args">
                  {pendingTool.tool === "run_command"
                    ? pendingTool.args.command
                    : pendingTool.args.path}
                </span>
              </div>
              {pendingTool.tool === "write_file" && (
                <pre className="tool-preview">{(pendingTool.args.content ?? "").slice(0, 2000)}</pre>
              )}
              <div className="tool-actions">
                <button className="primary" onClick={() => execTool(pendingTool)}>
                  Approve
                </button>
                <button className="danger" onClick={() => denyTool(pendingTool)}>
                  Deny
                </button>
              </div>
            </div>
          )}
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
                  title={`${t.kind === "tool-result" ? "tool" : t.role} · ~${tok} tok`}
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
              placeholder={
                ready
                  ? agentOn
                    ? "give the agent a task in the workspace…"
                    : "message the reactor…"
                  : "ignite a model first"
              }
              value={input}
              disabled={!ready || busyLoop}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            {busyLoop ? (
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
