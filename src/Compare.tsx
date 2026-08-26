/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Two branches of a conversation, side by side.
//!
//! Opened from a fork point, so the columns are already aligned: everything
//! before the fork is shared and shown once, above them. Each column carries
//! the settings that produced it and what it cost, because comparing two
//! answers without their inputs is comparing nothing — the same reason the
//! provenance row exists at all.

import { Turn, childrenOf, leafUnder, pathTo } from "./branch";
import { Markdown } from "./Markdown";
import { SamplerRow, ancestorSampler } from "./SamplerRow";
import { SamplerSnap } from "./types";

interface Column {
  /// The sibling this branch starts at.
  rootId: string;
  /// Where selecting this branch would put you.
  leafId: string;
  /// The turns after the fork. Never empty: a column exists because a turn does.
  turns: Turn[];
  tokens: number;
  /// Mean decode rate over the replies that reported one, or null if none did.
  rate: number | null;
  sampler: SamplerSnap | null;
}

/// Build one column per sibling at a fork.
///
/// The shared prefix is dropped from each column rather than repeated down
/// both: it is identical by construction, and repeating it would push the
/// only part that differs off the bottom of the screen.
function columnsAt(pool: Turn[], forkParent: string | null): Column[] {
  const shared = forkParent ? pathTo(pool, forkParent).length : 0;
  const byId = new Map(pool.map((t) => [t.id, t]));
  return childrenOf(pool, forkParent).map((sib) => {
    const leafId = leafUnder(pool, sib.id);
    const turns = pathTo(pool, leafId).slice(shared);
    const rates = turns.map((t) => t.decodeTokS ?? 0).filter((r) => r > 0);
    return {
      rootId: sib.id,
      leafId,
      turns,
      tokens: turns.reduce((n, t) => n + (t.tokens ?? 0), 0),
      rate: rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : null,
      // Walk from the tip, so a retry fork — which has no user turn of its
      // own — still reports the settings it inherited from before the fork.
      sampler: ancestorSampler(byId, turns[turns.length - 1]),
    };
  });
}

function CompareTurn({ t }: { t: Turn }) {
  if (t.kind === "tool-result") {
    return (
      <details className="tool-card result">
        <summary>
          <span className="tool-tag">⚙ {t.toolName}</span> result ·{" "}
          {t.content.split("\n").length} lines
        </summary>
        <pre>{t.content}</pre>
      </details>
    );
  }
  return (
    <div className={`turn ${t.role}`}>
      <div className={`turn-label ${t.role}`}>{t.role === "user" ? "You" : "Model"}</div>
      <div className="turn-body">
        {t.role === "assistant" ? <Markdown text={t.content} /> : t.content}
      </div>
      {t.meta && <div className="turn-meta">{t.meta}</div>}
    </div>
  );
}

export function Compare(p: {
  pool: Turn[];
  /// The turn the branches diverge after. `null` compares roots.
  forkParent: string | null;
  /// Which branch is on screen behind this view.
  currentLeaf: string | null;
  onUse: (leafId: string) => void;
  onClose: () => void;
}) {
  const cols = columnsAt(p.pool, p.forkParent);
  const shared = p.forkParent ? pathTo(p.pool, p.forkParent) : [];
  const divergesAfter = shared[shared.length - 1];
  // Every column is measured against the first, which is the only reading that
  // stays stable as you scan right. Column one marks nothing, by definition.
  const baseline = cols[0]?.sampler ?? null;
  const best = cols.reduce(
    (a, c) => (c.rate != null && (a == null || c.rate > a) ? c.rate : a),
    null as number | null
  );

  return (
    <div className="compare" role="dialog" aria-label="compare branches">
      <div className="compare-head">
        <span className="compare-title">⇄ COMPARE</span>
        <span className="compare-sub">
          {cols.length} versions · everything before this point is shared
        </span>
        <button className="compare-close" onClick={p.onClose} title="back to the conversation">
          ✕ close
        </button>
      </div>

      {divergesAfter && (
        <div className="compare-shared">
          <span className="compare-shared-tag">diverges after</span>
          <span className="compare-shared-body">
            {divergesAfter.content.replace(/\s+/g, " ").slice(0, 220) || "(empty turn)"}
            {divergesAfter.content.length > 220 ? "…" : ""}
          </span>
        </div>
      )}

      <div className="compare-cols">
        {cols.map((c, i) => (
          <section key={c.rootId} className={`compare-col ${c.leafId === p.currentLeaf ? "live" : ""}`}>
            <header>
              <span className="col-n">
                {i + 1} / {cols.length}
              </span>
              {c.leafId === p.currentLeaf && <span className="col-live">on screen</span>}
              <button onClick={() => p.onUse(c.leafId)} disabled={c.leafId === p.currentLeaf}>
                use this ▸
              </button>
            </header>
            {c.sampler && <SamplerRow snap={c.sampler} prev={i === 0 ? null : baseline} />}
            <div className="compare-body">
              {c.turns.map((t) => (
                <CompareTurn key={t.id} t={t} />
              ))}
            </div>
            <footer>
              <span>
                <b>{c.tokens.toLocaleString()}</b> tok
              </span>
              <span className={c.rate != null && c.rate === best && cols.length > 1 ? "fastest" : ""}>
                <b>{c.rate != null ? c.rate.toFixed(1) : "—"}</b> tok/s
              </span>
              <span>
                <b>{c.turns.length}</b> turns
              </span>
            </footer>
          </section>
        ))}
      </div>
    </div>
  );
}
