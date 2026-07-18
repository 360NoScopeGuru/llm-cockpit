import { Fragment, ReactNode } from "react";

// Minimal markdown → React renderer for model output. Deliberately builds
// React elements (never innerHTML) so a model can't inject markup into a
// webview that holds Tauri IPC. Covers what LLMs actually emit: headers,
// bold/italic/strike, inline + fenced code, lists (nested), blockquotes,
// tables, rules. Links render as styled text with the URL in the tooltip —
// navigation inside the app window would kill the session.

// ---- inline ----

const INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(__[^_\n]+__)|(\*[^*\n]+\*)|(_[^_\n]+_)|(~~[^~\n]+~~)|(\[[^\]\n]+\]\([^)\s]+\))/;

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let rest = text;
  let k = 0;
  while (rest.length > 0) {
    const m = INLINE.exec(rest);
    if (!m) {
      out.push(rest);
      break;
    }
    if (m.index > 0) out.push(rest.slice(0, m.index));
    const tok = m[0];
    if (m[1]) {
      out.push(<code key={k++}>{tok.slice(1, -1)}</code>);
    } else if (m[2] || m[3]) {
      out.push(<strong key={k++}>{inline(tok.slice(2, -2))}</strong>);
    } else if (m[4] || m[5]) {
      out.push(<em key={k++}>{inline(tok.slice(1, -1))}</em>);
    } else if (m[6]) {
      out.push(<s key={k++}>{inline(tok.slice(2, -2))}</s>);
    } else if (m[7]) {
      const t = /\[([^\]]+)\]\(([^)]+)\)/.exec(tok)!;
      out.push(
        <span key={k++} className="md-link" title={t[2]}>
          {inline(t[1])}
        </span>
      );
    }
    rest = rest.slice(m.index + tok.length);
  }
  return out;
}

// ---- blocks ----

interface ListItem {
  indent: number;
  ordered: boolean;
  text: string;
  children: ListItem[];
}

function parseListItems(lines: string[], start: number): { items: ListItem[]; next: number } {
  const flat: ListItem[] = [];
  let i = start;
  while (i < lines.length) {
    const m = /^(\s*)(?:([-*+])|(\d+)[.)])\s+(.*)$/.exec(lines[i]);
    if (!m) {
      // Indented continuation of the previous item.
      if (flat.length > 0 && /^\s{2,}\S/.test(lines[i])) {
        flat[flat.length - 1].text += "\n" + lines[i].trim();
        i++;
        continue;
      }
      break;
    }
    flat.push({
      indent: m[1].length,
      ordered: !!m[3],
      text: m[4],
      children: [],
    });
    i++;
  }
  // Nest by indent (2+ spaces deeper = child of the last shallower item).
  const roots: ListItem[] = [];
  const stack: ListItem[] = [];
  for (const item of flat) {
    while (stack.length > 0 && item.indent < stack[stack.length - 1].indent + 2) {
      stack.pop();
    }
    if (stack.length === 0) {
      roots.push(item);
    } else {
      stack[stack.length - 1].children.push(item);
    }
    stack.push(item);
  }
  return { items: roots, next: i };
}

function renderList(items: ListItem[], key: number): ReactNode {
  const ordered = items[0]?.ordered;
  const kids = items.map((it, j) => (
    <li key={j}>
      {it.text.split("\n").map((ln, x) => (
        <Fragment key={x}>
          {x > 0 && <br />}
          {inline(ln)}
        </Fragment>
      ))}
      {it.children.length > 0 && renderList(it.children, 0)}
    </li>
  ));
  return ordered ? <ol key={key}>{kids}</ol> : <ul key={key}>{kids}</ul>;
}

function isTableSep(line: string): boolean {
  return /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(line) && line.includes("-");
}

function splitRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((c) => c.trim());
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const out: ReactNode[] = [];
  let k = 0;
  let i = 0;
  let para: string[] = [];

  const flushPara = () => {
    if (para.length === 0) return;
    out.push(
      <p key={k++}>
        {para.map((ln, x) => (
          <Fragment key={x}>
            {x > 0 && <br />}
            {inline(ln)}
          </Fragment>
        ))}
      </p>
    );
    para = [];
  };

  while (i < lines.length) {
    const line = lines[i];

    // fenced code
    const fence = /^\s*```(\w*)\s*$/.exec(line);
    if (fence) {
      flushPara();
      const lang = fence[1];
      const buf: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
        buf.push(lines[i]);
        i++;
      }
      i++; // closing fence (or EOF)
      out.push(
        <pre key={k++} data-lang={lang || undefined}>
          <code>{buf.join("\n")}</code>
        </pre>
      );
      continue;
    }

    // blank
    if (/^\s*$/.test(line)) {
      flushPara();
      i++;
      continue;
    }

    // header
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      flushPara();
      const lvl = Math.min(4, h[1].length); // h5/h6 add nothing at this size
      const Tag = `h${lvl}` as "h1";
      out.push(<Tag key={k++}>{inline(h[2].replace(/\s#+\s*$/, ""))}</Tag>);
      i++;
      continue;
    }

    // horizontal rule
    if (/^\s*([-*_])\s*(\1\s*){2,}$/.test(line)) {
      flushPara();
      out.push(<hr key={k++} />);
      i++;
      continue;
    }

    // blockquote
    if (/^\s*>\s?/.test(line)) {
      flushPara();
      const buf: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      out.push(
        <blockquote key={k++}>
          <Markdown text={buf.join("\n")} />
        </blockquote>
      );
      continue;
    }

    // table
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      flushPara();
      const header = splitRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].includes("|") && !/^\s*$/.test(lines[i])) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      out.push(
        <div key={k++} className="md-table-wrap">
          <table>
            <thead>
              <tr>
                {header.map((c, x) => (
                  <th key={x}>{inline(c)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, y) => (
                <tr key={y}>
                  {r.map((c, x) => (
                    <td key={x}>{inline(c)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    // list
    if (/^(\s*)(?:[-*+]|\d+[.)])\s+/.test(line)) {
      flushPara();
      const { items, next } = parseListItems(lines, i);
      if (items.length > 0) {
        out.push(renderList(items, k++));
        i = next;
        continue;
      }
    }

    para.push(line);
    i++;
  }
  flushPara();

  return <div className="md">{out}</div>;
}
