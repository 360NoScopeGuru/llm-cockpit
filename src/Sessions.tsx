/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

import { SessionMeta, ctxLabel } from "./types";

// Saved sessions (lower half of the left rail). Chat and code transcripts are
// persisted continuously, so this is a permanent fixture rather than a panel
// you open — the model list above and the session list here are the two things
// you pick from, so they share the rail.

interface SessionsProps {
  sessions: SessionMeta[] | null;
  /// Ids currently open in the CHAT / CODE tabs, so they can be marked live.
  openIds: string[];
  busy: boolean;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}

function baseName(p: string): string {
  const parts = p.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? p;
}

function fmtDate(ms: number): string {
  const d = new Date(ms);
  return (
    d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
  );
}

export function Sessions(p: SessionsProps) {
  const list = p.sessions ?? [];
  const chats = list.filter((s) => s.kind !== "code").length;
  const codes = list.length - chats;
  return (
    <div className="sessions">
      <div className="sessions-head">
        <span className="lbl">Sessions</span>
        <span className="sessions-count">
          {p.sessions == null
            ? "loading…"
            : `${chats} chat · ${codes} code`}
        </span>
      </div>
      <div className="sessions-scroll">
        {list.map((m) => {
          const open = p.openIds.includes(m.id);
          return (
            <div
              key={m.id}
              className={`hist-row ${open ? "open" : ""}`}
              onClick={() => !p.busy && p.onOpen(m.id)}
              title={p.busy ? "finish the current generation first" : "open this session"}
            >
              <span className={`hist-kind ${m.kind}`}>
                {m.kind === "code" ? "CODE" : "CHAT"}
              </span>
              <span className="hist-main">
                <span className="hist-title">{m.title}</span>
                <span className="hist-detail">
                  {m.model_name ?? "unknown model"}
                  {m.ctx_size ? ` · ${ctxLabel(m.ctx_size)} ctx` : ""}
                  {m.workspace ? ` · ⌂ ${baseName(m.workspace)}` : ""}
                  {` · ${m.turn_count} turns`}
                  {/* Only worth saying when it is true. `turn_count` is the
                      branch you would reopen on, so a forked session needs
                      this to explain why the token total runs ahead of it. */}
                  {m.branch_count > 1 ? ` · ⑂ ${m.branch_count} branches` : ""}
                  {m.total_tokens > 0 ? ` · ${m.total_tokens.toLocaleString()} tok` : ""}
                  {m.avg_decode_tok_s > 0 ? ` · ${m.avg_decode_tok_s.toFixed(1)} tok/s` : ""}
                </span>
                <span className="hist-date">{fmtDate(m.updated_ms)}</span>
              </span>
              <button
                className="hist-del"
                onClick={(e) => {
                  e.stopPropagation();
                  p.onDelete(m.id);
                }}
                title="delete forever"
              >
                ✕
              </button>
            </div>
          );
        })}
        {p.sessions != null && list.length === 0 && (
          <div className="sessions-empty">
            No saved sessions yet — they appear here as you chat.
          </div>
        )}
      </div>
    </div>
  );
}
