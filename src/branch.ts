/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! The conversation tree.
//!
//! A session's turns are a pool, not a transcript: each one names the turn it
//! follows, so the pool describes a tree and the conversation you are reading
//! is one path through it. These are the operations on that shape, kept apart
//! from the console so the console can stay about talking to a model.
//!
//! Mirrors `history.rs` on the Rust side, which does the same walks over the
//! same fields for the session rail's counters.

import { SamplerSnap } from "./types";

export interface Turn {
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

export const newTurnId = () =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/// The turns on the branch ending at `head`, in conversation order.
///
/// Mirrors `history::path_indices` on the Rust side. `head` of `null` resolves
/// to the last turn, which is what every session written before Rev G means.
/// The visited set is there because a hand-edited file could contain a cycle,
/// and a console that hangs on a malformed session is worse than one that
/// shows a short transcript.
export function pathTo<T extends { id?: string | null; parent?: string | null }>(
  pool: T[],
  head: string | null | undefined
): T[] {
  if (pool.length === 0) return [];
  // A pool with no ids is a pre-Rev-G transcript that the backend did not get
  // to migrate. It is already in order, and walking it as a tree would follow
  // no edges at all and render only its last turn.
  if (!pool.some((t) => t.id)) return [...pool];
  const byId = new Map(pool.flatMap((t) => (t.id ? [[t.id, t] as [string, T]] : [])));
  let cur: T | undefined = (head && byId.get(head)) || pool[pool.length - 1];
  const out: T[] = [];
  const seen = new Set<string>();
  while (cur && !seen.has(cur.id ?? "")) {
    seen.add(cur.id ?? "");
    out.push(cur);
    cur = cur.parent ? byId.get(cur.parent) : undefined;
  }
  return out.reverse();
}

/// Turns that follow `id` directly, in the order they were created.
export function childrenOf(pool: Turn[], id: string | null): Turn[] {
  return pool.filter((t) => (t.parent ?? null) === id);
}

/// Walk down from `id` taking the newest child at each step, which is the
/// branch you were last on. Used when selecting a sibling: you pick a turn,
/// and the console shows you the whole branch it leads to.
export function leafUnder(pool: Turn[], id: string): string {
  let cur = id;
  const seen = new Set<string>();
  for (;;) {
    if (seen.has(cur)) return cur; // malformed file; do not spin
    seen.add(cur);
    const kids = childrenOf(pool, cur);
    if (kids.length === 0) return cur;
    cur = kids[kids.length - 1].id;
  }
}

/// Appends turns to a branch, linking each onto the one before it.
///
/// Every append in this file goes through here, so no call site can create a
/// turn without an id and a parent. Today that only ever builds a straight
/// line; Rev G's UI will pass a different tail to fork from, and this keeps
/// working unchanged because the edge always comes from whatever it is given.
export function chain(prev: Turn[], ...added: Omit<Turn, "id" | "parent">[]): Turn[] {
  const out = [...prev];
  for (const t of added) {
    out.push({ ...t, id: newTurnId(), parent: out.length ? out[out.length - 1].id : null });
  }
  return out;
}

