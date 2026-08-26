/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

import {
  Fragment,
  ReactNode,
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Markdown } from "./Markdown";
import {
  DraftCandidate,
  InferenceMetrics,
  SamplerSnap,
  ServerStatus,
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
// timestamps). The left rail lists and reopens them.

type TabKind = "chat" | "code";

interface Turn {
  /// Stable node id, assigned the moment the turn is created.
  id: string;
  /// The turn this one follows. `null` marks a root.
  parent: string | null;
  role: "user" | "assistant";
  kind?: "chat" | "tool-result" | "continue";
  toolName?: string;
  content: string;
  thinking?: string;
  meta?: string;
  tokens?: number;
  decodeTokS?: number;
  stopped?: boolean;
  /// `undefined` = unknown (e.g. an older saved turn); `null` = the stream
  /// ended without ever reporting why. These mean different things.
  finish?: string | null;
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
  /// Model-written session name. Null until the first exchange completes, at
  /// which point the raw first message is used as a placeholder.
  title: string | null;
}

const emptyTab = (): TabState => ({
  id: null,
  createdMs: 0,
  turns: [],
  input: "",
  loadedInfo: null,
  title: null,
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
  /// "stop" = the model finished. "length" = it was cut off. null = the stream
  /// died without saying why.
  finish: string | null;
  error: string | null;
}

export interface StagedIgnite {
  name: string;
  ngl: number;
  layers: number | null;
  ctx: number;
  busy: boolean;
  /// Models that could draft for this one. `null` while loading.
  drafts: DraftCandidate[] | null;
  draftPath: string | null;
  onPickDraft: (path: string | null) => void;
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
  /// Fired whenever the saved-session set changes, so the rail can refetch.
  onSessionsChanged: () => void;
  /// Pushes the state the rail needs to render: which sessions are open, and
  /// whether a generation is in flight (during which a session cannot be
  /// swapped in). Pushed rather than pulled through the ref, because a ref read
  /// during render would not re-render the rail when this changes.
  onStateChanged: (s: { openIds: string[]; busy: boolean }) => void;
}

/// Actions the left rail's session list drives on the console.
export interface ConsoleHandle {
  loadSession: (id: string) => void;
  deleteSession: (id: string) => void;
}

interface ToolCall {
  tool: "list_dir" | "read_file" | "write_file" | "run_command";
  args: Record<string, string>;
}

const TOOL_NAMES = new Set(["list_dir", "read_file", "write_file", "run_command"]);
const MAX_TOOL_ROUNDS = 24;
/// Consecutive "you stopped early, keep going" nudges before handing control
/// back. Capped separately from tool rounds and much lower: a tool round makes
/// visible progress, a nudge might not, and each one costs a full generation.
/// Reset as soon as the agent does something real.
const MAX_NUDGES = 3;

const estTokens = (s: string) => Math.max(1, Math.ceil(s.length / 4));

const newTurnId = () =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/// The turns on the branch ending at `head`, in conversation order.
///
/// Mirrors `history::path_indices` on the Rust side. `head` of `null` resolves
/// to the last turn, which is what every session written before Rev G means.
/// The visited set is there because a hand-edited file could contain a cycle,
/// and a console that hangs on a malformed session is worse than one that
/// shows a short transcript.
function pathTo(pool: StoredTurn[], head: string | null | undefined): StoredTurn[] {
  if (pool.length === 0) return [];
  // A pool with no ids is a pre-Rev-G transcript that the backend did not get
  // to migrate. It is already in order, and walking it as a tree would follow
  // no edges at all and render only its last turn.
  if (!pool.some((t) => t.id)) return [...pool];
  const byId = new Map(
    pool.flatMap((t) => (t.id ? [[t.id, t] as [string, StoredTurn]] : []))
  );
  let cur = (head && byId.get(head)) || pool[pool.length - 1];
  const out: StoredTurn[] = [];
  const seen = new Set<string>();
  while (cur && !seen.has(cur.id ?? "")) {
    seen.add(cur.id ?? "");
    out.push(cur);
    cur = (cur.parent && byId.get(cur.parent)) as StoredTurn;
  }
  return out.reverse();
}

/// Appends turns to a branch, linking each onto the one before it.
///
/// Every append in this file goes through here, so no call site can create a
/// turn without an id and a parent. Today that only ever builds a straight
/// line; Rev G's UI will pass a different tail to fork from, and this keeps
/// working unchanged because the edge always comes from whatever it is given.
function chain(prev: Turn[], ...added: Omit<Turn, "id" | "parent">[]): Turn[] {
  const out = [...prev];
  for (const t of added) {
    out.push({ ...t, id: newTurnId(), parent: out.length ? out[out.length - 1].id : null });
  }
  return out;
}

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

NEVER paste file contents into your reply. Writing code in chat produces
nothing on disk, burns the context window, and the work is lost. To create or
change a file you MUST use write_file. A reply that describes code without a
write_file block has accomplished nothing.

Keep going until the task is actually finished. Do not stop after planning,
after describing what you will do, or after one file when more are needed.
End EVERY reply one of exactly two ways:
  1. a \`\`\`tool block — you are continuing, or
  2. the single line TASK COMPLETE — everything is written and verified.
If you end any other way you will be asked to continue.`;

/// The agent's explicit end-of-work marker. Without one, a reply that merely
/// stops is indistinguishable from a reply that finished — which is how the
/// agent used to quit halfway through and look like it had crashed.
const DONE_MARKER = /^\s*(TASK COMPLETE|\*\*TASK COMPLETE\*\*)\s*\.?\s*$/im;

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

/// Blank, 0 or negative all mean "no cap": the field is omitted from the
/// request so the server falls back to its own n_predict = -1 (unlimited).
function capOf(v: string): number | null {
  const n = parseInt(v, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/// The sampler snapshot is written onto the *user* turn that opens an
/// exchange; the assistant reply it produced carries none. Walk back from `i`
/// to find the settings actually in force there. Without this, a reloaded
/// session reports every `finish: "length"` stop as a full context window even
/// when the user's own max-tokens cap was what ended it — the exact wrong-fix
/// advice `metaLine` goes out of its way to avoid on the live path.
function governingSampler(
  turns: readonly { sampler?: SamplerSnap | null }[],
  i: number
): SamplerSnap | null {
  for (let j = Math.min(i, turns.length - 1); j >= 0; j--) {
    const s = turns[j].sampler;
    if (s) return s;
  }
  return null;
}

/// `null` means the field was left blank, so it was omitted from the request
/// and the server applied its own default. That is not the same as zero, and
/// the row says so rather than inventing a number.
const SAMPLER_FIELDS: [keyof SamplerSnap, string, string][] = [
  ["temperature", "temp", "server default"],
  ["top_k", "top-k", "server default"],
  ["top_p", "top-p", "server default"],
  ["min_p", "min-p", "server default"],
  ["max_tokens", "max", "∞"],
];

/// One dim line under a user turn naming the settings that produced the reply
/// below it. Fields that moved since the previous exchange are marked, so
/// scanning a long session finds the moment a knob was turned without having
/// to diff five numbers by eye on every turn.
function SamplerRow({ snap, prev }: { snap: SamplerSnap; prev: SamplerSnap | null }) {
  const parts = SAMPLER_FIELDS.map(([key, label, dflt]) => {
    const v = snap[key] as number | null | undefined;
    return {
      key: key as string,
      text: `${label} ${v ?? dflt}`,
      changed: prev != null && (prev[key] ?? null) !== (v ?? null),
      title: undefined as string | undefined,
    };
  });
  const sys = snap.system?.trim() ?? "";
  const prevSys = prev?.system?.trim() ?? "";
  if (sys) {
    parts.push({ key: "system", text: "sys", changed: prev != null && prevSys !== sys, title: sys });
  } else if (prevSys) {
    // Silence here would read as "unchanged", which is the opposite of true.
    parts.push({ key: "system", text: "sys cleared", changed: true, title: undefined });
  }
  return (
    <div className="turn-meta sampler-prov">
      <span className="prov-tag">⚙</span>
      {parts.map((part, i) => (
        <Fragment key={part.key}>
          {i > 0 && " · "}
          <span className={part.changed ? "changed" : undefined} title={part.title}>
            {part.text}
          </span>
        </Fragment>
      ))}
    </div>
  );
}

function metaLine(
  tokens?: number | null,
  rate?: number | null,
  stopped?: boolean | null,
  finish?: string | null,
  /// The max-tokens cap in force for this turn, or null when it was unlimited.
  cap?: number | null
) {
  if (tokens == null || rate == null) return undefined;
  // `finish: "length"` covers two different causes and the fix differs, so
  // name the actual one. Telling someone to "raise max" when max is already ∞
  // sends them to a setting that cannot help.
  const cutOff =
    cap != null
      ? ` · ⚠ CUT OFF at the ${cap} token max — raise max`
      : " · ⚠ CUT OFF — context window full, not a token cap";
  const why = stopped
    ? " · stopped by you"
    : finish === "length"
      ? cutOff
      : finish === null
        ? " · ⚠ stream ended early"
        : ""; // undefined = unknown: say nothing rather than cry wolf
  return `${tokens} tok · ${rate.toFixed(1)} tok/s${why}`;
}

function fmtDate(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

export const Console = forwardRef<ConsoleHandle, ConsoleProps>(function Console(p, ref) {
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
  // Empty = unlimited: let the model run until it hits EOS or fills the
  // context, which is llama.cpp's own default (n_predict = -1). A finite
  // default here silently guillotines long answers — and on reasoning models
  // the thinking tokens eat the budget before the answer even starts.
  const [maxTok, setMaxTok] = useState("");
  const [copied, setCopied] = useState(false);
  const [pendingTool, setPendingTool] = useState<ToolCall | null>(null);
  const [toolBusy, setToolBusy] = useState(false);
  // The session list itself lives in the left rail now; the console only needs
  // to act on it and say when it changed.

  const genId = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const tabsRef = useRef(tabs);
  const streamTab = useRef<TabKind>("chat");
  const wsRef = useRef<string | null>(null);
  const systemRef = useRef("");
  const samplerRef = useRef({ temp: "0.7", topK: "40", topP: "0.95", minP: "0.05", maxTok: "" });
  const serverRef = useRef<ServerStatus | null>(null);
  const cfgRef = useRef<{ ngl: number; ctx: number } | null>(null);
  const modelNameRef = useRef<string | null>(null);
  const roundsRef = useRef(0);
  const nudgesRef = useRef(0);

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
  const lastTurn = cur.turns[cur.turns.length - 1];
  const canContinue =
    ready &&
    !streaming &&
    !toolBusy &&
    !pendingTool &&
    !!lastTurn &&
    lastTurn.role === "assistant" &&
    !lastTurn.error &&
    !!lastTurn.content;

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
      id: t.id,
      parent: t.parent,
      role: t.role,
      kind: t.kind === "tool-result" || t.kind === "continue" ? t.kind : null,
      tool_name: t.toolName ?? null,
      content: t.content,
      thinking: t.thinking ?? null,
      tokens: t.tokens ?? null,
      decode_tok_s: t.decodeTokS ?? null,
      stopped: t.stopped ?? null,
      finish: t.finish ?? null,
      error: t.error ?? null,
      timestamp_ms: t.ts ?? 0,
      sampler: t.sampler ?? null,
    };
  }

  function persist(k: TabKind) {
    const t = tabsRef.current[k];
    if (!t.id) return;
    // Empty turns are dropped, but only when nothing follows them: removing a
    // node mid-chain would orphan its children. Today the only empty turn is
    // the in-flight assistant reply at the tail, so this drops exactly what it
    // always did — it just stays correct once a pool can branch.
    const claimed = new Set(t.turns.map((x) => x.parent).filter(Boolean));
    const turns = t.turns.filter((x) => x.content || x.thinking || claimed.has(x.id));
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
      title: t.title ?? (firstUser?.content ?? "(untitled)").slice(0, 80),
      ...info,
      created_ms: t.createdMs || Date.now(),
      updated_ms: Date.now(),
      head: turns.length ? turns[turns.length - 1].id : null,
      turns: turns.map(toStored),
    };
    invoke("history_save", { session })
      .then(() => p.onSessionsChanged())
      .catch(() => {});
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
          max_tokens: capOf(s.maxTok) ?? undefined,
        },
      });
    } catch (e) {
      setStreaming(false);
      patchTurns(k, (prev) =>
        chain(prev.slice(0, -1), {
          role: "assistant",
          content: `⚠ ${e}`,
          error: true,
          ts: Date.now(),
        })
      );
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

  function continueWith(resultTurn: Omit<Turn, "id" | "parent">) {
    const k: TabKind = "code";
    const next = chain(tabsRef.current[k].turns, resultTurn, {
      role: "assistant",
      content: "",
      ts: Date.now(),
    });
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
      // No tool block. Either the agent is genuinely done, or it stopped
      // mid-task — which llama.cpp reports as an ordinary EOS (truncated = 0),
      // so nothing downstream can tell the difference. Treat only the explicit
      // marker as done and nudge otherwise, instead of silently giving up.
      if (DONE_MARKER.test(last.content) || last.stopped) {
        roundsRef.current = 0;
        nudgesRef.current = 0;
        return;
      }
      if (nudgesRef.current >= MAX_NUDGES || roundsRef.current >= MAX_TOOL_ROUNDS) {
        nudgesRef.current = 0;
        roundsRef.current = 0;
        patchTurns("code", (prev) =>
          chain(prev, {
            role: "assistant",
            content:
              `⚠ the model kept stopping early (${MAX_NUDGES} nudges). It may be ` +
              `out of its depth on this task — press ▸ Continue to push on, or ` +
              `give it a smaller step.`,
            error: true,
            ts: Date.now(),
          })
        );
        return;
      }
      nudgesRef.current += 1;
      continueWith({
        role: "user",
        kind: "continue",
        content:
          "You stopped without finishing. Continue from exactly where you left off. " +
          "Use write_file to put code on disk — do not paste it here. " +
          "End with a tool block, or with TASK COMPLETE if everything is written.",
        ts: Date.now(),
      });
      return;
    }
    if (roundsRef.current >= MAX_TOOL_ROUNDS) {
      patchTurns("code", (prev) =>
        chain(prev, {
          role: "assistant",
          content: `⚠ agent stopped after ${MAX_TOOL_ROUNDS} tool rounds — send a message to continue`,
          error: true,
          ts: Date.now(),
        })
      );
      roundsRef.current = 0;
      nudgesRef.current = 0;
      return;
    }
    // A real tool call is progress — forgive any earlier stalls.
    nudgesRef.current = 0;
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
              finish: e.payload.finish,
              meta: e.payload.error
                ? undefined
                : metaLine(
                    e.payload.tokens,
                    e.payload.decode_tok_s,
                    e.payload.stopped,
                    e.payload.finish,
                    governingSampler(next, next.length - 1)?.max_tokens ?? null
                  ),
            };
          }
          return next;
        });
        // Deferred a tick so tabsRef reflects the update above, then save and
        // (CODE tab) look for a tool call to continue the loop.
        setTimeout(() => {
          persist(k);
          nameSession(k);
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

  /// Resume a reply the model ended on its own. A model that emits EOS early
  /// is indistinguishable from one that finished, so this is a manual escape
  /// hatch for both tabs — the CODE tab also nudges itself automatically.
  function continueGen() {
    const k = tab;
    if (!ready || streaming || toolBusy || pendingTool) return;
    const next = chain(
      tabsRef.current[k].turns,
      {
        role: "user",
        kind: "continue",
        content:
          "Continue from exactly where you left off. Do not restart or repeat " +
          "what you already wrote.",
        ts: Date.now(),
      },
      { role: "assistant", content: "", ts: Date.now() }
    );
    patchTab(k, (t) => ({ ...t, turns: next }));
    dispatch(k, next);
  }

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
    nudgesRef.current = 0;
    const s = samplerRef.current;
    const snap: SamplerSnap = {
      temperature: parseFloat(s.temp) || null,
      top_k: parseInt(s.topK, 10) || null,
      top_p: parseFloat(s.topP) || null,
      min_p: parseFloat(s.minP) || null,
      max_tokens: parseInt(s.maxTok, 10) || null,
      system: systemRef.current.trim() || null,
    };
    const next = chain(
      t.turns,
      { role: "user", content: text, ts: Date.now(), sampler: snap },
      { role: "assistant", content: "", ts: Date.now() }
    );
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
    nudgesRef.current = 0;
    setPendingTool(null);
  }

  function newSession() {
    persist(tab);
    patchTab(tab, () => emptyTab());
    if (tab === "code") {
      setPendingTool(null);
      roundsRef.current = 0;
      nudgesRef.current = 0;
    }
  }

  async function copyApi() {
    if (!server?.base_url) return;
    await navigator.clipboard.writeText(`${server.base_url}/v1`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  /// Have the model name the session once the first exchange lands. Titles come
  /// from a separate non-streaming request so this never disturbs generation,
  /// and a failure is silent — the first-message fallback is still a title.
  async function nameSession(k: TabKind) {
    const t = tabsRef.current[k];
    if (t.title || !t.id || !serverRef.current?.running) return;
    const real = t.turns.filter((x) => x.kind !== "continue" && !x.error);
    const user = real.find((x) => x.role === "user");
    const reply = real.find((x) => x.role === "assistant" && x.content);
    if (!user || !reply) return;
    const transcript =
      `User: ${user.content.slice(0, 1200)}\n\n` +
      `Assistant: ${stripToolBlock(reply.content).slice(0, 1200)}`;
    try {
      const title = await invoke<string>("chat_title", { transcript });
      if (!title) return;
      // The tab may have been reset or reloaded while the title was in flight.
      if (tabsRef.current[k].id !== t.id) return;
      patchTab(k, (tab) => ({ ...tab, title }));
      setTimeout(() => {
        persist(k);
        p.onSessionsChanged();
      }, 0);
    } catch {
      /* keep the fallback title */
    }
  }

  // ---- sessions (list rendered by the left rail) ----

  async function loadSession(id: string) {
    if (streaming || toolBusy) return;
    try {
      const s = await invoke<StoredSession>("history_get", { id });
      const kind: TabKind = s.kind === "code" ? "code" : "chat";
      // The pool is stored unordered; the transcript is the branch `head`
      // names. For a session that never forked these are the same array.
      const path = pathTo(s.turns, s.head);
      const turns: Turn[] = path.map((st, i) => ({
        id: st.id ?? `t${i}`,
        parent: st.parent ?? null,
        role: st.role === "user" ? "user" : "assistant",
        kind:
          st.kind === "tool-result" || st.kind === "continue"
            ? (st.kind as "tool-result" | "continue")
            : undefined,
        toolName: st.tool_name ?? undefined,
        content: st.content,
        thinking: st.thinking ?? undefined,
        tokens: st.tokens ?? undefined,
        decodeTokS: st.decode_tok_s ?? undefined,
        stopped: st.stopped ?? undefined,
        finish: st.finish,
        error: st.error ?? undefined,
        ts: st.timestamp_ms || undefined,
        sampler: st.sampler ?? undefined,
        meta:
          st.role === "assistant" && st.kind !== "tool-result"
            ? metaLine(
                st.tokens,
                st.decode_tok_s,
                st.stopped,
                st.finish,
                governingSampler(path, i)?.max_tokens ?? null
              )
            : undefined,
      }));
      setTabs((prev) => ({
        ...prev,
        [kind]: {
          id: s.id,
          createdMs: s.created_ms,
          turns,
          input: "",
          title: s.title || null,
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
    } catch {
      /* row vanished under us — let the rail resync */
      p.onSessionsChanged();
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
    p.onSessionsChanged();
  }

  useImperativeHandle(ref, () => ({ loadSession, deleteSession }));

  useEffect(() => {
    p.onStateChanged({
      openIds: [tabs.chat.id, tabs.code.id].filter((x): x is string => !!x),
      busy: streaming || toolBusy,
    });
  }, [tabs.chat.id, tabs.code.id, streaming, toolBusy]);

  const kvTokens = p.metrics?.kv_cache_tokens ?? 0;
  const kvPct = Math.round((p.metrics?.kv_cache_usage_ratio ?? 0) * 100);
  const busyLoop = streaming || toolBusy || !!pendingTool;
  const ctxSize = p.liveCfg?.ctx ?? null;

  const sampler = (
    label: string,
    value: string,
    set: (v: string) => void,
    wide = false,
    placeholder?: string
  ) => (
    <span className="sampler">
      {label}
      <input
        className={wide ? "wide" : ""}
        value={value}
        placeholder={placeholder ?? (wide ? "(none)" : "")}
        onChange={(e) => set(e.target.value)}
      />
    </span>
  );

  // ---- turn rendering ----

  function renderTurn(t: Turn, i: number, all: Turn[]) {
    const isLast = i === all.length - 1;
    const mine = streamTab.current === tab;
    if (t.kind === "continue") {
      // Shown as a thin marker rather than a full user turn — it is bookkeeping,
      // not something the user said.
      return (
        <div key={i} className="continue-mark">
          ▸ continued — the model had stopped early
        </div>
      );
    }
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
        {t.role === "user" && t.sampler && (
          <SamplerRow snap={t.sampler} prev={governingSampler(all, i - 1)} />
        )}
      </div>
    );
  }

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
        {canContinue && (
          <button
            onClick={continueGen}
            title="the model stopped on its own — pick up where it left off"
          >
            ▸ Continue
          </button>
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
      ) : ready || cur.turns.length > 0 ? (
        // Turns render even with no model loaded: reopening a saved session
        // from the rail must be readable without igniting something first.
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
            <DraftPicker staged={p.staged} />
            <button className="ignite-cta" disabled={p.staged.busy} onClick={p.staged.onIgnite}>
              IGNITE · {p.staged.ngl}
              {p.staged.layers ? `/${p.staged.layers}` : ""} LAYERS · {ctxLabel(p.staged.ctx)} CTX
              {p.staged.draftPath ? " · +DRAFT" : ""}
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

      {!p.board && ready && ctxSize && (cur.turns.length > 0 || kvTokens > 0) && (
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

      {!p.board && (
        <>
          <div className="sampler-row">
            {sampler("temp", temp, setTemp)}
            {sampler("top-k", topK, setTopK)}
            {sampler("top-p", topP, setTopP)}
            {sampler("min-p", minP, setMinP)}
            {sampler("max", maxTok, setMaxTok, false, "∞")}
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
});

/// Draft-model picker for speculative decoding, shown on the staged model
/// just before ignition.
///
/// The verdicts here answer "does this fit, and what does it cost" — never
/// "will this be faster". That distinction is load-bearing: on the development
/// machine a pair this rates `recommended` (3.9% of the target's compute, zero
/// layers evicted) still measured **0.65x** — slower — because acceptance was
/// 44% and the target was already decoding at 63 tok/s. Whether speculation
/// pays is a question only the benchmark answers, so the copy points there
/// instead of implying a win.
function DraftPicker({ staged }: { staged: StagedIgnite }) {
  const [showRejected, setShowRejected] = useState(false);
  const list = staged.drafts;

  if (list === null) {
    return <div className="draft-pick muted">checking for draft models…</div>;
  }

  const usable = list.filter((d) => d.verdict.kind !== "incompatible");
  const rejected = list.filter((d) => d.verdict.kind === "incompatible");

  // Nothing offerable: say so only if something was rejected, otherwise stay
  // quiet rather than explaining an absence nobody asked about.
  if (usable.length === 0) {
    if (rejected.length === 0) return null;
    return (
      <div className="draft-pick muted">
        no model here shares this one's tokenizer — {rejected.length} checked
      </div>
    );
  }

  const pct = (d: DraftCandidate) =>
    d.cost_ratio == null ? "?" : `${(d.cost_ratio * 100).toFixed(1)}%`;

  return (
    <div className="draft-pick">
      <div className="draft-head">
        SPECULATIVE DRAFT
        <span className="draft-note">runs ahead of the model; costs VRAM either way</span>
      </div>
      <div className="draft-opts">
        <button
          className={staged.draftPath === null ? "on" : ""}
          onClick={() => staged.onPickDraft(null)}
        >
          NONE
        </button>
        {usable.slice(0, 4).map((d) => {
          const pair = d.pair;
          const evicted = pair?.target_layers_evicted ?? 0;
          // A pair that costs target layers is a worse trade than not
          // speculating at all: those layers run on the CPU for every token.
          const blocked = pair?.verdict === "too_big";
          const title = blocked
            ? "does not fit alongside this model"
            : evicted > 0
              ? `fits, but pushes ${evicted} target layer(s) onto the CPU`
              : `${pct(d)} of the target's compute per token, no layers displaced`;
          return (
            <button
              key={d.path}
              className={`${staged.draftPath === d.path ? "on" : ""} ${d.economics}`}
              disabled={blocked}
              title={title}
              onClick={() => staged.onPickDraft(d.path)}
            >
              {d.label.replace(/\.gguf$/i, "")}
              <small>
                {pct(d)}
                {evicted > 0 ? ` · −${evicted}L` : ""}
              </small>
            </button>
          );
        })}
      </div>
      {staged.draftPath && (
        <div className="draft-hint">
          benchmark it — a draft that fits can still be slower than none
        </div>
      )}
      {rejected.length > 0 && (
        <div className="draft-rejected">
          <button className="linkish" onClick={() => setShowRejected((v) => !v)}>
            {rejected.length} incompatible {showRejected ? "▾" : "▸"}
          </button>
          {showRejected && (
            <ul>
              {rejected.slice(0, 6).map((d) => (
                <li key={d.path}>
                  <span className="mono">{d.label.replace(/\.gguf$/i, "")}</span>
                  {d.verdict.kind === "incompatible" && d.verdict.reasons[0]
                    ? ` — ${d.verdict.reasons[0]}`
                    : ""}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
