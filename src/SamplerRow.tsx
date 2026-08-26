/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! The sampler provenance line: which settings produced a given reply.
//!
//! Shared by the console, where it sits under a user turn and marks what moved
//! since the last exchange, and by the compare view, where it sits at the head
//! of each column and marks what differs from the branch beside it. Same
//! question either way — what were the inputs? — so it is the same row.

import { Fragment } from "react";
import { SamplerSnap } from "./types";
import { Turn } from "./branch";

/// The same search as [`governingSampler`], over a pool rather than a branch.
/// A branch array is already in ancestor order, so the index walk there is
/// enough; a pool is not ordered at all and has to follow `parent`.
export function ancestorSampler(byId: Map<string, Turn>, from: Turn | undefined): SamplerSnap | null {
  const seen = new Set<string>();
  let cur = from;
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id);
    if (cur.sampler) return cur.sampler;
    cur = cur.parent ? byId.get(cur.parent) : undefined;
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
export function SamplerRow({ snap, prev }: { snap: SamplerSnap; prev: SamplerSnap | null }) {
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
