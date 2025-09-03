#!/usr/bin/env python3
from pathlib import Path
import re, difflib, argparse, html
from typing import List, Tuple, Dict, Any

# ----------------------------
# Parsing LaTeX structure
# ----------------------------
HEADER_RE = re.compile(r"^\s*\\(chapter|section|subsection|subsubsection)\*?\s*(?:\[[^\]]*\])?\s*\{(.*?)}\s*$")
LEVEL_ORDER = {"chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3}

def strip_latex_commands(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", "", s)
    s = re.sub(r"~", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_units(tex: str, level: str) -> List[Dict[str, Any]]:
    """
    Split LaTeX into units at or above the requested level.
    Returns dicts with: level, title, title_clean, start_line, end_line, content
    """
    lines = tex.splitlines()
    headers = []
    for i, line in enumerate(lines):
        m = HEADER_RE.match(line)
        if m:
            hlevel, title = m.group(1), m.group(2)
            if LEVEL_ORDER[hlevel] <= LEVEL_ORDER[level]:
                headers.append({"idx": i, "level": hlevel, "title": title})

    # Fallback: if nothing at requested level, accept any headers
    if not headers:
        for i, line in enumerate(lines):
            m = HEADER_RE.match(line)
            if m:
                hlevel, title = m.group(1), m.group(2)
                headers.append({"idx": i, "level": hlevel, "title": title})

    units = []
    for j, h in enumerate(headers):
        start = h["idx"]
        end = headers[j + 1]["idx"] if j + 1 < len(headers) else len(lines)
        block_lines = lines[start:end]
        m = HEADER_RE.match(block_lines[0])
        if not m:
            continue
        hlevel, title = m.group(1), m.group(2)
        if LEVEL_ORDER[hlevel] > LEVEL_ORDER[level]:
            continue
        content = "\n".join(block_lines[1:])  # exclude header line itself
        title_clean = strip_latex_commands(title)
        units.append(
            {
                "level": hlevel,
                "title": title.strip(),
                "title_clean": title_clean,
                "start_line": start + 1,
                "end_line": end,
                "content": content,
            }
        )
    return units

# ----------------------------
# Similarity (content-first)
# ----------------------------
TOKEN_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
TOKEN_NONALNUM = re.compile(r"[^A-Za-z0-9]+")

def _tokenize(text: str):
    # Strip common LaTeX noise, comments, and split
    t = TOKEN_CMD.sub(" ", text)
    t = re.sub(r"%.*", " ", t)  # comments
    t = TOKEN_NONALNUM.sub(" ", t)
    toks = [w.lower() for w in t.split() if len(w) > 2]
    return toks

def _shingles(tokens, n=3):
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def content_similarity(a_text: str, b_text: str) -> float:
    # Jaccard on 3-gram token shingles — robust to small edits and reformatting
    A = _shingles(_tokenize(a_text), 3)
    B = _shingles(_tokenize(b_text), 3)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def title_similarity(a_title: str, b_title: str) -> float:
    return difflib.SequenceMatcher(None, a_title.lower(), b_title.lower()).ratio()

def greedy_match(A: List[Dict[str, Any]],
                 B: List[Dict[str, Any]],
                 threshold: float = 0.50,
                 mode: str = "hybrid",
                 w_title: float = 0.4,
                 w_content: float = 0.6):
    """
    Reciprocal best match with content-based detection of renamed/moved units.

    mode:
      - 'content' : use only content_similarity
      - 'title'   : use only title_similarity
      - 'hybrid'  : combine (default), leaning on content to catch renames

    threshold: minimum combined score to accept a match (0..1).
    """
    assert mode in {"content", "title", "hybrid"}

    # Cache content sims (can be O(n*m), so cache helps)
    cont_cache: Dict[Tuple[int,int], float] = {}

    def combined_score(i, j):
        if mode == "content":
            cs = cont_cache.get((i, j))
            if cs is None:
                cs = content_similarity(A[i]["content"], B[j]["content"])
                cont_cache[(i, j)] = cs
            return cs
        elif mode == "title":
            return title_similarity(A[i]["title_clean"], B[j]["title_clean"])
        else:
            ts = title_similarity(A[i]["title_clean"], B[j]["title_clean"])
            cs = cont_cache.get((i, j))
            if cs is None:
                cs = content_similarity(A[i]["content"], B[j]["content"])
                cont_cache[(i, j)] = cs
            # If titles are very different, lean more on content
            if ts < 0.2:
                return 0.15 * ts + 0.85 * cs
            return w_title * ts + w_content * cs

    best_for_a: Dict[int, Tuple[int, float]] = {}
    for i in range(len(A)):
        if not B: break
        j, s = max(((j, combined_score(i, j)) for j in range(len(B))), key=lambda t: t[1])
        if s >= threshold:
            best_for_a[i] = (j, s)

    best_for_b: Dict[int, Tuple[int, float]] = {}
    for j in range(len(B)):
        if not A: break
        i, s = max(((i, combined_score(i, j)) for i in range(len(A))), key=lambda t: t[1])
        if s >= threshold:
            best_for_b[j] = (i, s)

    matches: List[Tuple[int, int, float]] = []
    used_i, used_j = set(), set()
    for i, (j, s) in best_for_a.items():
        if j in best_for_b and best_for_b[j][0] == i and i not in used_i and j not in used_j:
            matches.append((i, j, s))
            used_i.add(i); used_j.add(j)
    return matches, used_i, used_j

# ----------------------------
# Diff + HTML Rendering (dark)
# ----------------------------
def diff_block(a: str, b: str, fromfile: str, tofile: str) -> str:
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    ud = difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile, lineterm="")
    return "\n".join(ud)

def render_html_report(level: str,
                       A_units: List[Dict[str, Any]],
                       B_units: List[Dict[str, Any]],
                       matches: List[Tuple[int, int, float]],
                       matched_i: set,
                       matched_j: set,
                       a_name: str,
                       b_name: str,
                       out_path: Path):
    head = '''<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="color-scheme" content="light dark">
<title>LaTeX Structural Diff</title>
<style>
/* -----------------------
   THEME VARIABLES
   ----------------------- */
:root {
  --bg: #ffffff;
  --fg: #1f2937;
  --muted: #6b7280;
  --panel: #f8fafc;
  --border: #e5e7eb;

  --code-bg: #0b1021;
  --code-fg: #d6deeb;

  --badge-bg: #e5e7eb;
  --badge-fg: #111827;

  --band-moved-bg: #ffcc00; --band-moved-fg: #000;
  --band-added-bg: #22c55e; --band-added-fg: #000;
  --band-removed-bg: #dc2626; --band-removed-fg: #fff;

  --diff-add: #16a34a;
  --diff-del: #ef4444;
  --diff-hdr: #0891b2;

  --btn-bg: #f3f4f6; --btn-fg: #111827; --btn-border: #d1d5db; --btn-hover: #e5e7eb;
  --menu-bg: #ffffff; --menu-fg: #111827; --menu-border: #e5e7eb; --menu-shadow: rgba(0,0,0,.1);
}

/* Dark palette */
body[data-theme="dark"] {
  --bg: #1e1e1e;
  --fg: #dddddd;
  --muted: #b3b3b3;
  --panel: #252525;
  --border: #333333;

  --code-bg: #0b1021;
  --code-fg: #d6deeb;

  --badge-bg: #444444;
  --badge-fg: #eeeeee;

  --band-moved-bg: #ffcc00; --band-moved-fg: #000;
  --band-added-bg: #4ade80; --band-added-fg: #000;
  --band-removed-bg: #dc2626; --band-removed-fg: #fff;

  --diff-add: #a6e22e;
  --diff-del: #f92672;
  --diff-hdr: #66d9ef;

  --btn-bg: #2a2a2a; --btn-fg: #f3f4f6; --btn-border: #3a3a3a; --btn-hover: #333333;
  --menu-bg: #252525; --menu-fg: #f3f4f6; --menu-border: #333333; --menu-shadow: rgba(0,0,0,.35);
}

/* High contrast (approx) */
body[data-theme="hc"] {
  --bg: #000000;
  --fg: #ffffff;
  --muted: #d1d5db;
  --panel: #0a0a0a;
  --border: #666666;

  --code-bg: #000000;
  --code-fg: #ffffff;

  --badge-bg: #222222;
  --badge-fg: #ffffff;

  --band-moved-bg: #ffff00; --band-moved-fg: #000000;
  --band-added-bg: #00ff00; --band-added-fg: #000000;
  --band-removed-bg: #ff0000; --band-removed-fg: #ffffff;

  --diff-add: #00ff00;
  --diff-del: #ff5555;
  --diff-hdr: #00ccff;

  --btn-bg: #111; --btn-fg: #fff; --btn-border: #777; --btn-hover: #222;
  --menu-bg: #000; --menu-fg: #fff; --menu-border: #777; --menu-shadow: rgba(255,255,255,.15);
}

/* -----------------------
   BASE + LAYOUT
   ----------------------- */
html, body { height: 100%; }
body {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  margin: 1.5rem;
  background: var(--bg);
  color: var(--fg);
}
h1 { margin-top: 0; color: var(--fg); }

/* Theme switcher */
.theme-switcher {
  position: fixed;
  top: 12px;
  right: 12px;
  z-index: 9999;
}
.theme-button {
  background: var(--btn-bg);
  color: var(--btn-fg);
  border: 1px solid var(--btn-border);
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 12px;
  cursor: pointer;
}
.theme-button:hover { background: var(--btn-hover); }
.theme-menu {
  display: none;
  position: absolute;
  right: 0;
  margin-top: 6px;
  background: var(--menu-bg);
  color: var(--menu-fg);
  border: 1px solid var(--menu-border);
  border-radius: 8px;
  min-width: 170px;
  box-shadow: 0 10px 24px var(--menu-shadow);
  overflow: hidden;
}
.theme-menu.open { display: block; }
.theme-menu button {
  width: 100%;
  display: block;
  text-align: left;
  background: transparent;
  border: 0;
  padding: 8px 12px;
  color: inherit;
  cursor: pointer;
  font-size: 13px;
}
.theme-menu button:hover { background: var(--btn-hover); }

/* Summary + badges */
.summary {
  background: var(--panel);
  padding: 1rem;
  border-radius: 10px;
  border: 1px solid var(--border);
  margin-top: 44px; /* leave space for switcher */
}
.badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  margin-right: 0.5rem;
  font-size: 0.9em;
  font-weight: 600;
  background: var(--badge-bg);
  color: var(--badge-fg);
}
.badge.matched { background: var(--band-moved-bg); color: var(--band-moved-fg); }
.badge.added   { background: var(--band-added-bg); color: var(--band-added-fg); }
.badge.removed { background: var(--band-removed-bg); color: var(--band-removed-fg); }
small { color: var(--muted); }

/* Cards */
.section {
  border: 1px solid var(--border);
  border-radius: 10px;
  margin: 1rem 0;
  background: var(--panel);
  overflow: hidden;
}

/* Colored headers per status */
.section > .header {
  padding: 0.65rem 0.85rem;
  font-weight: 700;
  border-bottom: 1px solid var(--border);
}
.section.moved   > .header { background: var(--band-moved-bg); color: var(--band-moved-fg); }
.section.added   > .header { background: var(--band-added-bg); color: var(--band-added-fg); }
.section.removed > .header { background: var(--band-removed-bg); color: var(--band-removed-fg); }

/* Code / diff area */
.section pre {
  padding: 0.9rem;
  overflow: auto;
  background: var(--code-bg);
  color: var(--code-fg);
  white-space: pre-wrap;
  margin: 0;
}
.code { background: var(--code-bg); color: var(--code-fg); }

/* Tables (if any) */
table { border-collapse: collapse; width: 100%; }
td, th { border-bottom: 1px solid var(--border); padding: 0.4rem 0.2rem; text-align: left; }

/* Diff colors */
pre .diff-added   { color: var(--diff-add); }
pre .diff-removed { color: var(--diff-del); }
pre .diff-header  { color: var(--diff-hdr); }

/* Respect OS color scheme if opened in a normal browser (fallback) */
@media (prefers-color-scheme: dark) {
  body:not([data-theme]) { /* only if we haven't set a theme explicitly */
    color-scheme: dark;
    background: #1e1e1e; color: #ddd;
  }
}
</style>
<script>
(function() {
  function getExplicitThemeFromURL() {
    try {
      var url = new URL(window.location.href);
      var q = (url.searchParams.get('theme') || '').toLowerCase();
      if (q === 'dark' || q === 'light' || q === 'hc') return q;
      var h = (url.hash || '').replace('#','').toLowerCase();
      if (h === 'dark' || h === 'light' || h === 'hc') return h;
    } catch(e) {}
    return null;
  }

  function getVSCodeThemeClass() {
    var cls = (document.body.className + ' ' + document.documentElement.className).toLowerCase();
    if (cls.includes('vscode-high-contrast')) return 'hc';
    if (cls.includes('vscode-dark')) return 'dark';
    if (cls.includes('vscode-light')) return 'light';
    return null;
  }

  function pickTheme() {
    // 0) localStorage
    var saved = localStorage.getItem('latex-structural-diff-theme');
    if (saved === 'dark' || saved === 'light' || saved === 'hc') return saved;

    // 1) URL override
    var urlT = getExplicitThemeFromURL();
    if (urlT) return urlT;

    // 2) VS Code classes
    var vsT = getVSCodeThemeClass();
    if (vsT) return vsT;

    // 3) OS fallback
    var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return prefersDark ? 'dark' : 'light';
  }

  function applyTheme(t) {
    document.body.setAttribute('data-theme', t);
  }

  function saveTheme(t) {
    try { localStorage.setItem('latex-structural-diff-theme', t); } catch (e) {}
  }

  function initSwitcher() {
    var btn = document.querySelector('.theme-button');
    var menu = document.querySelector('.theme-menu');
    if (!btn || !menu) return;

    btn.addEventListener('click', function() {
      menu.classList.toggle('open');
    });

    // Close menu on outside click / escape
    document.addEventListener('click', function(e) {
      if (!menu.contains(e.target) && e.target !== btn) menu.classList.remove('open');
    });
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') menu.classList.remove('open');
    });

    ['light','dark','hc'].forEach(function(mode) {
      var el = document.getElementById('choose-' + mode);
      if (el) {
        el.addEventListener('click', function() {
          applyTheme(mode);
          saveTheme(mode);
          menu.classList.remove('open');
        });
      }
    });
  }

  function watchOSChanges() {
    if (!window.matchMedia) return;
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) {
      mq.addEventListener('change', function() {
        // Only update if theme wasn't explicitly saved or forced
        var saved = localStorage.getItem('latex-structural-diff-theme');
        var urlT = getExplicitThemeFromURL();
        var vsT = getVSCodeThemeClass();
        if (!saved && !urlT && !vsT) applyTheme(mq.matches ? 'dark' : 'light');
      });
    }
  }

  document.addEventListener('DOMContentLoaded', function() {
    applyTheme(pickTheme());
    initSwitcher();
    watchOSChanges();
  });
})();
</script>
</head><body>
<div class="theme-switcher">
  <button class="theme-button" aria-haspopup="true" aria-expanded="false" title="Theme">Theme ▾</button>
  <div class="theme-menu" role="menu" aria-label="Choose theme">
    <button id="choose-light" role="menuitem">Light</button>
    <button id="choose-dark" role="menuitem">Dark</button>
    <button id="choose-hc" role="menuitem">High contrast</button>
  </div>
</div>
'''
    summary = f'''
<h1>LaTeX structural diff</h1>
<div class="summary">
  <div><strong>Granularity:</strong> {html.escape(level.capitalize())}</div>
  <div><span class="badge">Old: {html.escape(a_name)}</span> <span class="badge">New: {html.escape(b_name)}</span></div>
  <div>
    <span class="badge matched">Matched: {len(matches)}</span>
    <span class="badge added">Added: {len([j for j in range(len(B_units)) if j not in matched_j])}</span>
    <span class="badge removed">Removed: {len([i for i in range(len(A_units)) if i not in matched_i])}</span>
  </div>
</div>
'''
    def highlight_diff(text: str) -> str:
        out = []
        for line in text.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                out.append(f'<span class="diff-added">{html.escape(line)}</span>')
            elif line.startswith('-') and not line.startswith('---'):
                out.append(f'<span class="diff-removed">{html.escape(line)}</span>')
            elif line.startswith('@@') or line.startswith('---') or line.startswith('+++'):
                out.append(f'<span class="diff-header">{html.escape(line)}</span>')
            else:
                out.append(html.escape(line))
        return "\n".join(out)

    sections_html = []

    # Matched/moved
    for i, j, s in matches:
        A = A_units[i]; B = B_units[j]
        header = (
            f'<div class="header"><strong>Matched {html.escape(level)}:</strong> '
            f'"{html.escape(A["title"])}" → "{html.escape(B["title"])}" '
            f'(similarity {s:.2f}) <small>[{a_name} #{i+1} → {b_name} #{j+1}]</small></div>'
        )
        difftext = diff_block(
            A["content"], B["content"],
            f"{a_name}:{A['title_clean'][:30]}",
            f"{b_name}:{B['title_clean'][:30]}"
        )
        pre = f'<pre class="code">{highlight_diff(difftext) or "(no intra-section changes detected)"}</pre>'
        sections_html.append(f'<div class="section moved">{header}{pre}</div>')

    # Added
    for j in range(len(B_units)):
        if j not in matched_j:
            B = B_units[j]
            header = (
                f'<div class="header"><strong>Added {html.escape(level)}:</strong> '
                f'"{html.escape(B["title"])}" <small>[{b_name} #{j+1}]</small></div>'
            )
            pre = f'<pre class="code">{html.escape(B["content"])}</pre>'
            sections_html.append(f'<div class="section added">{header}{pre}</div>')

    # Removed
    for i in range(len(A_units)):
        if i not in matched_i:
            A = A_units[i]
            header = (
                f'<div class="header"><strong>Removed {html.escape(level)}:</strong> '
                f'"{html.escape(A["title"])}" <small>[{a_name} #{i+1}]</small></div>'
            )
            pre = f'<pre class="code">{html.escape(A["content"])}</pre>'
            sections_html.append(f'<div class="section removed">{header}{pre}</div>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(head + summary + "\n".join(sections_html) + "\n</body></html>")


# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Structural diff for LaTeX that matches moved/renamed chapters/sections by CONTENT.")
    p.add_argument("old", help="Old .tex file")
    p.add_argument("new", help="New .tex file")
    p.add_argument("--level", choices=["chapter","section","subsection","subsubsection"], default="chapter",
                   help="Granularity to compare (default: chapter)")
    p.add_argument("--out", default="latex_structural_diff.html", help="HTML report path")
    p.add_argument("--mode", choices=["content","title","hybrid"], default="hybrid",
                   help="Matching mode: content, title, or hybrid (default: hybrid)")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Min match score (0..1). Lower if many renames. Default: 0.50")
    p.add_argument("--wtitle", type=float, default=0.4, help="Weight for title similarity in hybrid mode")
    p.add_argument("--wcontent", type=float, default=0.6, help="Weight for content similarity in hybrid mode")
    args = p.parse_args()

    a_name = Path(args.old).name; b_name = Path(args.new).name
    A_txt = Path(args.old).read_text(encoding="utf-8", errors="ignore")
    B_txt = Path(args.new).read_text(encoding="utf-8", errors="ignore")

    A_units = parse_units(A_txt, args.level)
    B_units = parse_units(B_txt, args.level)

    matches, matched_i, matched_j = greedy_match(
        A_units, B_units,
        threshold=args.threshold,
        mode=args.mode,
        w_title=args.wtitle,
        w_content=args.wcontent,
    )

    render_html_report(args.level, A_units, B_units, matches, matched_i, matched_j,
                       a_name, b_name, Path(args.out))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
