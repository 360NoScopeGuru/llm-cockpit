import { ReactNode, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Markdown } from "./Markdown";
import {
  InferenceMetrics,
  SamplerSnap,
  ServerStatus,
  SessionMeta,
  StoredSession,
  StoredTurn,
  baseName,
  ctxLabel,
} from "./types";

// The console: streaming chat wired straight to the running llama-server
// through the Rust backend. Two tabs share the pane:
//   CHAT — plain conversation, no tools.
//   CODE — the agent: workspace-sandboxed tools, one ```tool block per
//          reply, reads auto-run, writes/commands gated behind APPROVE.
// Each tab is its own session; every completed turn is persisted to
// <config>/tokamak/sessions/<id>.json with full detail (model, config,
// sampler settings, per-reply tokens + tok/s, thinking, tool calls,
// timestamps). The HISTORY panel lists and reopens them.

type TabKind = "chat" | "code";

interface Turn {
  role: "user" | "assistant";
  kind?: "chat" | "tool-result";
  toolName?: string;
  content: string;
  thinking?: string;
  meta?: string;
  tokens?: number;
  decodeTokS?: number;
  stopped?: boolean;
  error?: boolean;
  ts?: number;
  sampler?: SamplerSnap;
}

interface SessionInfo {
  model_name: string | null;
  model_path: string | null;
  binary_label: string | null;
  n_gpu_layers: number | null;
  ctx_size: number | null;
  workspace: string | null;
}

interface TabState {
  id: string | null;
  createdMs: number;
  turns: Turn[];
  input: string;
  loadedInfo: SessionInfo | null; // metadata from a reopened session
}

const emptyTab = (): TabState => ({
  id: null,
  createdMs: 0,
  turns: [],
  input: "",
  loadedInfo: null,
});

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
  liveCfg: { ngl: number; ctx: number } | null;
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

const newSessionId = () =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 7).replace(/[^a-z0-9]/g, "0")}`;

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

function metaLine(tokens?: number | null, rate?: number | null, stopped?: boolean | null) {
  if (tokens == null || rate == null) return undefined;
  return `${tokens} tok · ${rate.toFixed(1)} tok/s${stopped ? " · stopped" : ""}`;
}

function fmtDate(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

export function Console(p: ConsoleProps) {
  const [tab, setTab] = useState<TabKind>("chat");
  const [tabs, setTabs] = useState<Record<TabKind, TabState>>({
    chat: emptyTab(),
    code: emptyTab(),
  });
  const [streaming, setStreaming] = useState(false);
  const [system, setSystem] = useState("");
  const [temp, setTemp] = useState("0.7");
  const [topK, setTopK] = useState("40");
  const [topP, setTopP] = useState("0.95");
  const [minP, setMinP] = useState("0.05");
  const [maxTok, setMaxTok] = useState("2048");
  const [copied, setCopied] = useState(false);
  const [pendingTool, setPendingTool] = useState<ToolCall | null>(null);
  const [toolBusy, setToolBusy] = useState(false);
  const [histOpen, setHistOpen] = useState(false);
  const [histList, setHistList] = useState<SessionMeta[] | null>(null);

  const genId = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const tabsRef = useRef(tabs);
  const streamTab = useRef<TabKind>("chat");
  const wsRef = useRef<string | null>(null);
  const systemRef = useRef("");
  const samplerRef = useRef({ temp: "0.7", topK: "40", topP: "0.95", minP: "0.05", maxTok: "2048" });
  const serverRef = useRef<ServerStatus | null>(null);
  const cfgRef = useRef<{ ngl: number; ctx: number } | null>(null);
  const modelNameRef = useRef<string | null>(null);
  const roundsRef = useRef(0);

  tabsRef.current = tabs;
  wsRef.current = p.workspace;
  systemRef.current = system;
  samplerRef.current = { temp, topK, topP, minP, maxTok };
  serverRef.current = p.server;
  cfgRef.current = p.liveCfg;
  modelNameRef.current = p.modelName;

  const server = p.server;
  const health = server?.running ? server.health : "stopped";
  const ready = !!server?.running && health === "ok";
  const igniting = !!server?.running && (health === "starting" || health === "loading");
  const fault = !!server?.error && !ready && !igniting;
  const cur = tabs[tab];

  // ---- per-tab state helpers ----

  function patchTab(k: TabKind, fn: (t: TabState) => TabState) {
    setTabs((prev) => ({ ...prev, [k]: fn(prev[k]) }));
  }

  function patchTurns(k: TabKind, fn: (turns: Turn[]) => Turn[]) {
    patchTab(k, (t) => ({ ...t, turns: fn(t.turns) }));
  }

  // ---- persistence ----

  function toStored(t: Turn): StoredTurn {
    return {
      role: t.role,
      kind: t.kind === "tool-result" ? "tool-result" : null,
      tool_name: t.toolName ?? null,
      content: t.content,
      thinking: t.thinking ?? null,
      tokens: t.tokens ?? null,
      decode_tok_s: t.decodeTokS ?? null,
      stopped: t.stopped ?? null,
      error: t.error ?? null,
      timestamp_ms: t.ts ?? 0,
      sampler: t.sampler ?? null,
    };
  }

  function persist(k: TabKind) {
    const t = tabsRef.current[k];
    if (!t.id) return;
    const turns = t.turns.filter((x) => x.content || x.thinking);
    if (turns.length === 0) return;
    const srv = serverRef.current;
    const info: SessionInfo =
      srv?.running
        ? {
            model_name: modelNameRef.current,
            model_path: srv.model_path,
            binary_label: srv.binary_label,
            n_gpu_layers: cfgRef.current?.ngl ?? null,
            ctx_size: cfgRef.current?.ctx ?? null,
            workspace: k === "code" ? wsRef.current : null,
          }
        : t.loadedInfo ?? {
            model_name: modelNameRef.current,
            model_path: null,
            binary_label: null,
            n_gpu_layers: null,
            ctx_size: null,
            workspace: k === "code" ? wsRef.current : null,
          };
    const firstUser = turns.find((x) => x.role === "user" && x.kind !== "tool-result");
    const session: StoredSession = {
      id: t.id,
      kind: k,
      title: (firstUser?.content ?? "(untitled)").slice(0, 80),
      ...info,
      created_ms: t.createdMs || Date.now(),
      updated_ms: Date.now(),
      turns: turns.map(toStored),
    };
    invoke("history_save", { session }).catch(() => {});
  }

  // ---- message assembly + dispatch ----

  function buildMessages(k: TabKind, allTurns: Turn[]) {
    const sysParts: string[] = [];
    if (k === "code" && wsRef.current) sysParts.push(agentPrompt(wsRef.current));
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

  async function dispatch(k: TabKind, allTurns: Turn[]) {
    const id = ++genId.current;
    streamTab.current = k;
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
        messages: buildMessages(k, allTurns),
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
      patchTurns(k, (prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: `⚠ ${e}`, error: true, ts: Date.now() },
      ]);
    }
  }

  // ---- agent tool loop (CODE tab only) ----

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
      ts: Date.now(),
    });
  }

  function denyTool(call: ToolCall) {
    setPendingTool(null);
    continueWith({
      role: "user",
      kind: "tool-result",
      toolName: call.tool,
      content: "The user DENIED this tool call. Do not retry it; ask them or take another path.",
      ts: Date.now(),
    });
  }

  function continueWith(resultTurn: Turn) {
    const k: TabKind = "code";
    const next: Turn[] = [
      ...tabsRef.current[k].turns,
      resultTurn,
      { role: "assistant", content: "", ts: Date.now() },
    ];
    patchTab(k, (t) => ({ ...t, turns: next }));
    setTimeout(() => persist(k), 50);
    dispatch(k, next);
  }

  function maybeRunToolLoop() {
    if (streamTab.current !== "code" || !wsRef.current) return;
    const t = tabsRef.current.code.turns;
    const last = t[t.length - 1];
    if (!last || last.role !== "assistant" || last.error || !last.content) return;
    const call = parseToolCall(last.content);
    if (!call) {
      roundsRef.current = 0;
      return;
    }
    if (roundsRef.current >= MAX_TOOL_ROUNDS) {
      patchTurns("code", (prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠ agent stopped after ${MAX_TOOL_ROUNDS} tool rounds — send a message to continue`,
          error: true,
          ts: Date.now(),
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
        patchTurns(streamTab.current, (prev) => {
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
        const k = streamTab.current;
        patchTurns(k, (prev) => {
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
              decodeTokS: e.payload.decode_tok_s,
              stopped: e.payload.stopped,
              meta: e.payload.error
                ? undefined
                : metaLine(e.payload.tokens, e.payload.decode_tok_s, e.payload.stopped),
            };
          }
          return next;
        });
        // Deferred a tick so tabsRef reflects the update above, then save and
        // (CODE tab) look for a tool call to continue the loop.
        setTimeout(() => {
          persist(k);
          if (!e.payload.error && !e.payload.stopped) maybeRunToolLoop();
          else roundsRef.current = 0;
        }, 30);
      })
    );
    return () => {
      disposed = true;
      unlistens.forEach((u) => u());
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [tabs, pendingTool, tab]);

  // ---- actions ----

  async function send() {
    const k = tab;
    const t = tabs[k];
    const text = t.input.trim();
    if (!text || !ready || streaming || toolBusy || pendingTool) return;
    if (k === "code" && !p.workspace) {
      p.onPickWorkspace();
      return;
    }
    roundsRef.current = 0;
    const s = samplerRef.current;
    const snap: SamplerSnap = {
      temperature: parseFloat(s.temp) || null,
      top_k: parseInt(s.topK, 10) || null,
      top_p: parseFloat(s.topP) || null,
      min_p: parseFloat(s.minP) || null,
      max_tokens: parseInt(s.maxTok, 10) || null,
      system: systemRef.current.trim() || null,
    };
    const next: Turn[] = [
      ...t.turns,
      { role: "user", content: text, ts: Date.now(), sampler: snap },
      { role: "assistant", content: "", ts: Date.now() },
    ];
    patchTab(k, (prev) => ({
      ...prev,
      id: prev.id ?? newSessionId(),
      createdMs: prev.createdMs || Date.now(),
      turns: next,
      input: "",
    }));
    setTimeout(() => persist(k), 50);
    dispatch(k, next);
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

  function newSession() {
    persist(tab);
    patchTab(tab, () => emptyTab());
    if (tab === "code") {
      setPendingTool(null);
      roundsRef.current = 0;
    }
  }

  async function copyApi() {
    if (!server?.base_url) return;
    await navigator.clipboard.writeText(`${server.base_url}/v1`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  // ---- history panel ----

  async function openHistory() {
    try {
      setHistList(await invoke<SessionMeta[]>("history_list"));
    } catch {
      setHistList([]);
    }
    setHistOpen(true);
  }

  async function loadSession(id: string) {
    if (streaming || toolBusy) return;
    try {
      const s = await invoke<StoredSession>("history_get", { id });
      const kind: TabKind = s.kind === "code" ? "code" : "chat";
      const turns: Turn[] = s.turns.map((st) => ({
        role: st.role === "user" ? "user" : "assistant",
        kind: st.kind === "tool-result" ? "tool-result" : undefined,
        toolName: st.tool_name ?? undefined,
        content: st.content,
        thinking: st.thinking ?? undefined,
        tokens: st.tokens ?? undefined,
        decodeTokS: st.decode_tok_s ?? undefined,
        stopped: st.stopped ?? undefined,
        error: st.error ?? undefined,
        ts: st.timestamp_ms || undefined,
        sampler: st.sampler ?? undefined,
        meta:
          st.role === "assistant" && st.kind !== "tool-result"
            ? metaLine(st.tokens, st.decode_tok_s, st.stopped)
            : undefined,
      }));
      setTabs((prev) => ({
        ...prev,
        [kind]: {
          id: s.id,
          createdMs: s.created_ms,
          turns,
          input: "",
          loadedInfo: {
            model_name: s.model_name ?? null,
            model_path: s.model_path ?? null,
            binary_label: s.binary_label ?? null,
            n_gpu_layers: s.n_gpu_layers ?? null,
            ctx_size: s.ctx_size ?? null,
            workspace: s.workspace ?? null,
          },
        },
      }));
      setTab(kind);
      setHistOpen(false);
    } catch {
      /* row disappeared — refresh */
      openHistory();
    }
  }

  async function deleteSession(id: string) {
    try {
      await invoke("history_delete", { id });
    } catch {
      /* ignore */
    }
    // A deleted session that is currently open keeps its turns but must not
    // resurrect the file on the next save — detach the id.
    (["chat", "code"] as TabKind[]).forEach((k) => {
      if (tabsRef.current[k].id === id) {
        patchTab(k, (t) => ({ ...t, id: newSessionId() }));
      }
    });
    setHistList((prev) => (prev ? prev.filter((m) => m.id !== id) : prev));
  }

  const kvTokens = p.metrics?.kv_cache_tokens ?? 0;
  const kvPct = Math.round((p.metrics?.kv_cache_usage_ratio ?? 0) * 100);
  const busyLoop = streaming || toolBusy || !!pendingTool;
  const ctxSize = p.liveCfg?.ctx ?? null;

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

  function renderTurn(t: Turn, i: number, all: Turn[]) {
    const isLast = i === all.length - 1;
    const mine = streamTab.current === tab;
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
    const done = !(streaming && mine && isLast);
    const call = t.role === "assistant" && done ? parseToolCall(t.content) : null;
    const body = call ? stripToolBlock(t.content) : t.content;
    return (
      <div key={i} className={`turn ${t.role} ${t.error ? "error-turn" : ""}`}>
        <div className={`turn-label ${t.role}`}>
          {t.role === "user" ? "You" : p.modelName ?? "Model"}
          {t.ts ? <span className="turn-ts"> · {fmtDate(t.ts)}</span> : null}
        </div>
        {t.thinking && (
          <details className="thinking-box" open={isLast && streaming && mine && !t.content}>
            <summary>thinking · ~{estTokens(t.thinking)} tok</summary>
            <div className="thinking">{t.thinking}</div>
          </details>
        )}
        <div className="turn-body">
          {t.role === "assistant" ? <Markdown text={body} /> : body}
          {t.role === "assistant" && streaming && mine && isLast && (
            <span className="caret-blink" />
          )}
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
        {t.role === "assistant" && streaming && mine && isLast && p.metrics && (
          <div className="turn-meta">
            streaming · {p.metrics.predicted_tokens_per_sec.toFixed(1)} tok/s
          </div>
        )}
        {t.meta && <div className="turn-meta">{t.meta}</div>}
      </div>
    );
  }

  const historyPanel = (
    <div className="board history-panel">
      <div className="board-head">
        <span className="lbl">History</span>
        <span style={{ font: "10.5px var(--mono)", color: "var(--faint)" }}>
          {histList?.length ?? 0} saved sessions · every turn, config and measurement kept
        </span>
        <span className="spacer" />
        <button onClick={() => setHistOpen(false)}>✕</button>
      </div>
      {(histList ?? []).map((m) => (
        <div
          key={m.id}
          className="hist-row"
          onClick={() => loadSession(m.id)}
          title="open this session"
        >
          <span className={`hist-kind ${m.kind}`}>{m.kind === "code" ? "CODE" : "CHAT"}</span>
          <span className="hist-main">
            <span className="hist-title">{m.title}</span>
            <span className="hist-detail">
              {m.model_name ?? "unknown model"}
              {m.n_gpu_layers != null ? ` · ${m.n_gpu_layers}L` : ""}
              {m.ctx_size ? ` · ${ctxLabel(m.ctx_size)} ctx` : ""}
              {m.workspace ? ` · ⌂ ${baseName(m.workspace)}` : ""}
              {` · ${m.turn_count} turns`}
              {m.total_tokens > 0 ? ` · ${m.total_tokens.toLocaleString()} tok` : ""}
              {m.avg_decode_tok_s > 0 ? ` · ${m.avg_decode_tok_s.toFixed(1)} tok/s avg` : ""}
            </span>
          </span>
          <span className="hist-date">{fmtDate(m.updated_ms)}</span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              deleteSession(m.id);
            }}
            title="delete forever"
          >
            ✕
          </button>
        </div>
      ))}
      {histList && histList.length === 0 && (
        <div className="transcript-empty">No saved sessions yet — they appear here automatically as you chat.</div>
      )}
    </div>
  );

  return (
    <div className="console">
      <div className="console-head">
        <span className="tab-bar">
          <button className={`tab-btn ${tab === "chat" ? "active" : ""}`} onClick={() => setTab("chat")}>
            Chat
          </button>
          <button className={`tab-btn ${tab === "code" ? "active" : ""}`} onClick={() => setTab("code")}>
            Code
          </button>
        </span>
        {p.modelName && <span className="console-model">{p.modelName}</span>}
        {p.cfgText && <span className="console-cfg">{p.cfgText}</span>}
        <span className="spacer" />
        {tab === "code" && (
          <button onClick={p.onPickWorkspace} title={p.workspace ?? "pick a workspace folder"}>
            ⌂ {p.workspace ? baseName(p.workspace) : "pick workspace"}
          </button>
        )}
        <button onClick={openHistory} title="browse saved sessions">
          History
        </button>
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
        {cur.turns.length > 0 && (
          <button onClick={newSession} disabled={streaming} title="start a fresh session (this one stays in history)">
            New
          </button>
        )}
      </div>

      {p.kvAlert && ready && (
        <div className="alert-banner">
          <span className="alert-dot" />
          <span className="alert-title">CONTAINMENT NEAR CAPACITY — KV CACHE {kvPct}%</span>
          <span className="alert-sub">
            {kvTokens.toLocaleString()}
            {ctxSize ? ` / ${ctxSize.toLocaleString()}` : ""} tok · context is nearly full
          </span>
          <span className="spacer" />
          <button onClick={newSession} disabled={streaming}>
            New Session
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
      ) : histOpen ? (
        historyPanel
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
          {cur.turns.length === 0 && (
            <div className="transcript-empty">
              {tab === "chat" ? (
                <>Reactor live. Message it below — every session is saved to History automatically.</>
              ) : p.workspace ? (
                <>
                  Code tab: the agent can read anything in{" "}
                  <span style={{ color: "var(--plasma)" }}>{p.workspace}</span> and will ask before
                  writing files or running commands. Give it a task.
                </>
              ) : (
                <>
                  Code tab: pick a workspace folder (⌂ above) to give the agent somewhere to work,
                  then give it a task.
                </>
              )}
            </div>
          )}
          {cur.turns.map((t, i) => renderTurn(t, i, cur.turns))}
          {tab === "code" && toolBusy && (
            <div className="tool-card running-tool">
              <span className="tool-tag">⚙ running tool…</span>
            </div>
          )}
          {tab === "code" && pendingTool && (
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

      {!p.board && !histOpen && ready && ctxSize && (cur.turns.length > 0 || kvTokens > 0) && (
        <div className="timeline">
          <div className="timeline-track">
            {cur.turns.map((t, i) => {
              const tok = t.tokens ?? estTokens(t.content + (t.thinking ?? ""));
              const w = Math.max(0.6, (tok / ctxSize) * 100);
              const live =
                streaming && streamTab.current === tab && i === cur.turns.length - 1 &&
                t.role === "assistant";
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
              {kvTokens.toLocaleString()} / {ctxSize.toLocaleString()} tok
              {p.kvAlert && <span style={{ color: "var(--danger)" }}> · ceiling</span>}
            </span>
          </div>
        </div>
      )}

      {!p.board && !histOpen && (
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
                !ready
                  ? "ignite a model first"
                  : tab === "code"
                    ? p.workspace
                      ? "give the agent a task in the workspace…"
                      : "pick a workspace folder first (⌂ above)"
                    : "message the reactor…"
              }
              value={cur.input}
              disabled={!ready || busyLoop}
              onChange={(e) => {
                const v = e.target.value;
                patchTab(tab, (t) => ({ ...t, input: v }));
              }}
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
              <button className="send primary" onClick={send} disabled={!ready || !cur.input.trim()}>
                Send
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
