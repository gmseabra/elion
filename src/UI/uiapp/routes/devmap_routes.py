# =============================================================================
# routes/devmap_routes.py
# -----------------------------------------------------------------------------
# Dev source map — "which file, which line is this button actually running?"
#
# The hub is ~2 600 lines of template, ~12 000 lines of JavaScript across 20
# files, and 90-odd Flask routes. When a button misbehaves, the slow part is not
# fixing it: it is finding it. `onclick="_vinaDock()"` tells you the function
# name and nothing else, and the name is not greppable to one place — it may be
# defined in a file, re-wrapped in another, and reached through a delegated
# `data-nav` dispatcher in a third.
#
# This module builds an index of the repository and serves it to the browser,
# where `web/static/js/devmap.js` turns it into a hover tooltip on every button:
#
#     Handler   _vinaDock()            web/static/js/vina.js:826
#     Markup    #vinaDockBtn           web/templates/hub.html:1102
#     Calls     POST /vina_visualization/vina_dock
#                                      uiapp/routes/vina_dock_routes.py:499
#
# Design notes
# ------------
# * ONE request, not one per hover. The whole index ships as a single JSON
#   document that the client caches; a per-hover round trip would make the
#   tooltip lag behind the cursor and put a request on the server for every
#   mouse movement across a toolbar.
# * Regex, not a parser. A JS parser would be more precise, but it would also be
#   a dependency and would fail closed on the template's inline `<script>`
#   blocks. Every pattern here is anchored and conservative: a missed definition
#   costs one blank tooltip row, whereas a wrong one sends someone to the wrong
#   file, so ambiguity is REPORTED (all matches are returned) rather than
#   resolved by guessing.
# * Rebuilt when the tree changes. The index carries the max mtime of every file
#   it read; a request re-scans if anything is newer. Editing vina.js and
#   reloading the page shows the new line numbers, which is the whole point of a
#   development aid.
# * Read-only and repo-scoped. Only files under the repository root are scanned,
#   and only their line numbers and a one-line snippet are exposed — never file
#   contents wholesale.
# =============================================================================

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

from flask import jsonify, request

from uiapp import app
from uiapp import config as _cfg
from uiapp.routes.shared import logger

REPO_ROOT = Path(_cfg.REPO_ROOT)

# Directories worth indexing. Everything else in the tree is data, engine
# binaries, or third-party code that no button dispatches into.
_SCAN_DIRS = (
    ("web/static/js", (".js",)),
    ("web/templates", (".html",)),
    ("uiapp", (".py",)),
)

_SKIP_DIR_NAMES = {"__pycache__", "node_modules", ".git", "data", "runs", ".raytmp"}

# Cap so a stray large generated file cannot blow up the index. Anything past
# this is skipped and reported in `skipped`, never silently truncated — a
# half-indexed file would produce line numbers that are subtly wrong.
_MAX_FILE_BYTES = 3_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────
# JS definitions. Each yields (name) in group 1 and is matched per line, so the
# line number is exact. Ordered most-specific first; a line can register under
# more than one pattern and that is fine — duplicates are merged by (file,line).
_JS_DEF_PATTERNS = (
    ("function",  re.compile(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")),
    ("window",    re.compile(r"^\s*window\.([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function|\()")),
    ("const-fn",  re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)")),
    ("assign-fn", re.compile(r"^\s*([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function\b")),
    ("const",     re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*[\{\[]")),
    ("method",    re.compile(r"^\s*([A-Za-z_$][\w$]*)\s*[:=]\s*(?:async\s+)?function\s*\(")),
)

# Python: route decorators and plain defs.
_PY_ROUTE_RE = re.compile(r"""^\s*@app\.route\(\s*['"]([^'"]+)['"]\s*(?:,\s*methods\s*=\s*(\[[^\]]*\]))?""")
_PY_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][\w]*)\s*\(")

# HTML: element ids and inline handlers.
_HTML_ID_RE = re.compile(r"""\bid\s*=\s*["']([^"']+)["']""")
_HTML_NAV_RE = re.compile(r"""\bdata-nav\s*=\s*["']([^"']+)["']""")
_HTML_ON_RE = re.compile(r"""\bon(click|change|input|submit)\s*=\s*["']([^"']*)["']""")

# Endpoint references inside a function body.
_URL_RE = re.compile(r"""(?:fetch|EventSource|open)\s*\(\s*[`'"]([^`'"]+)[`'"]""")
_URL_METHOD_RE = re.compile(r"""method\s*:\s*['"](\w+)['"]""")

# `case 'vina':` inside hub.js's delegated dispatcher — this is what turns a
# `data-nav` attribute into a real handler, and without it every sidebar card
# would report "no handler found".
_JS_CASE_RE = re.compile(r"""^\s*case\s+['"]([\w-]+)['"]\s*:(.*)$""")
_JS_CALL_RE = re.compile(r"""\bcall\(\s*['"]([A-Za-z_$][\w$]*)['"]""")
_JS_IDENT_CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")


# ─────────────────────────────────────────────────────────────────────────────
# Index build
# ─────────────────────────────────────────────────────────────────────────────
_INDEX: dict = {}
_INDEX_LOCK = threading.Lock()


def _iter_files():
    for rel_dir, suffixes in _SCAN_DIRS:
        base = REPO_ROOT / rel_dir
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
            for fn in filenames:
                if fn.endswith(suffixes):
                    yield Path(dirpath) / fn


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _body_urls(lines: list[str], start: int, max_lines: int = 400) -> list[dict]:
    """Endpoint URLs referenced between `start` and the end of that function.

    The end is found by brace balance from the definition line. Brace counting
    is not a JS parser — a brace inside a string or regex literal miscounts —
    so it is bounded by `max_lines` and treated as a heuristic: the worst case
    is that a tooltip lists one endpoint too many, which is still a lead.
    """
    depth = 0
    seen_open = False
    urls: list[dict] = []
    end = min(len(lines), start + max_lines)

    for i in range(start, end):
        line = lines[i]

        for m in _URL_RE.finditer(line):
            raw = m.group(1)
            if not raw.startswith("/"):
                continue
            # `/vina_visualization/ts_mol_svg/${key}/${id}` →
            # `/vina_visualization/ts_mol_svg/<var>/<var>` so it matches the
            # Flask rule `/vina_visualization/ts_mol_svg/<key>/<mol_id>`.
            norm = re.sub(r"\$\{[^}]*\}", "<var>", raw).split("?")[0]
            method = "GET"
            window = " ".join(lines[i:min(len(lines), i + 6)])
            mm = _URL_METHOD_RE.search(window)
            if mm:
                method = mm.group(1).upper()
            elif "EventSource" in line:
                method = "SSE"
            urls.append({"url": norm, "method": method, "line": i + 1})

        depth += line.count("{") - line.count("}")
        if "{" in line:
            seen_open = True
        if seen_open and depth <= 0 and i > start:
            break

    # De-duplicate, keeping first occurrence order.
    out, seen = [], set()
    for u in urls:
        key = (u["method"], u["url"])
        if key not in seen:
            seen.add(key)
            out.append(u)
    return out


def _scan_js(path: Path, text: str, idx: dict) -> None:
    rel = _rel(path)
    lines = text.splitlines()

    for i, line in enumerate(lines):
        for kind, pattern in _JS_DEF_PATTERNS:
            m = pattern.match(line)
            if not m:
                continue
            name = m.group(1)
            entry = {
                "file": rel, "line": i + 1, "kind": kind, "lang": "js",
                "snippet": line.strip()[:160],
            }
            if kind in ("function", "window", "const-fn", "assign-fn", "method"):
                calls = _body_urls(lines, i)
                if calls:
                    entry["calls"] = calls
            idx["symbols"].setdefault(name, []).append(entry)
            break                                    # first matching pattern wins

        # data-nav dispatch: `case 'vina': ... call('_vinaShow');`
        m = _JS_CASE_RE.match(line)
        if m:
            key, rest = m.group(1), m.group(2)
            # Prefer the explicit `call('_x')` indirection hub.js uses, then any
            # plain identifier call on the same line.
            targets = _JS_CALL_RE.findall(rest)
            if not targets:
                targets = [t for t in _JS_IDENT_CALL_RE.findall(rest)
                           if t not in ("call", "if", "for", "while", "switch", "return", "break")]
            idx["nav"].setdefault(key, []).append({
                "file": rel, "line": i + 1, "targets": targets,
                "snippet": line.strip()[:200],
            })


def _scan_html(path: Path, text: str, idx: dict) -> None:
    rel = _rel(path)
    lines = text.splitlines()
    in_script = False

    for i, line in enumerate(lines):
        low = line.lower()
        if "<script" in low:
            in_script = True
        if "</script" in low:
            in_script = False

        # Inline <script> blocks define real functions — index them like JS.
        if in_script:
            for kind, pattern in _JS_DEF_PATTERNS:
                m = pattern.match(line)
                if m:
                    idx["symbols"].setdefault(m.group(1), []).append({
                        "file": rel, "line": i + 1, "kind": kind, "lang": "js-inline",
                        "snippet": line.strip()[:160],
                    })
                    break

        for m in _HTML_ID_RE.finditer(line):
            idx["elements"].setdefault(m.group(1), []).append({
                "file": rel, "line": i + 1, "snippet": line.strip()[:160],
            })

        for m in _HTML_NAV_RE.finditer(line):
            idx["nav_markup"].setdefault(m.group(1), []).append({
                "file": rel, "line": i + 1, "snippet": line.strip()[:160],
            })

        for m in _HTML_ON_RE.finditer(line):
            handler = m.group(2).strip()
            call = _JS_IDENT_CALL_RE.search(handler)
            if call:
                idx["handlers"].setdefault(call.group(1), []).append({
                    "file": rel, "line": i + 1, "event": m.group(1),
                    "snippet": handler[:160],
                })


def _scan_py(path: Path, text: str, idx: dict) -> None:
    rel = _rel(path)
    lines = text.splitlines()
    pending: list[tuple[str, list[str], int]] = []

    for i, line in enumerate(lines):
        m = _PY_ROUTE_RE.match(line)
        if m:
            rule = m.group(1)
            methods_raw = m.group(2) or "['GET']"
            methods = re.findall(r"""['"](\w+)['"]""", methods_raw) or ["GET"]
            pending.append((rule, methods, i + 1))
            continue

        d = _PY_DEF_RE.match(line)
        if d:
            name = d.group(1)
            idx["symbols"].setdefault(name, []).append({
                "file": rel, "line": i + 1, "kind": "def", "lang": "py",
                "snippet": line.strip()[:160],
            })
            # Any @app.route decorators immediately above bind to this def.
            for rule, methods, dec_line in pending:
                # Flask converters (`<job_id>`, `<int:n>`) normalise to `<var>`
                # so a client URL built from a template literal matches.
                norm = re.sub(r"<[^>]+>", "<var>", rule)
                idx["routes"].setdefault(norm, []).append({
                    "file": rel, "line": dec_line, "view": name,
                    "methods": methods, "rule": rule, "def_line": i + 1,
                })
            pending = []
            continue

        if line.strip() and not line.lstrip().startswith(("@", "#")):
            pending = []                             # decorator run interrupted


def _build_index() -> dict:
    t0 = time.time()
    idx: dict = {
        "symbols": {},        # JS/Py name  → [definition sites]
        "elements": {},       # DOM id      → [markup sites]
        "handlers": {},       # fn name     → [inline onclick sites]
        "nav": {},            # data-nav    → [dispatcher sites]
        "nav_markup": {},     # data-nav    → [markup sites]
        "routes": {},         # URL rule    → [route sites]
        "skipped": [],
    }

    newest = 0.0
    n_files = 0
    for path in _iter_files():
        try:
            st = path.stat()
            if st.st_size > _MAX_FILE_BYTES:
                idx["skipped"].append({"file": _rel(path), "reason": "too large",
                                       "bytes": st.st_size})
                continue
            newest = max(newest, st.st_mtime)
            text = path.read_text(errors="replace")
        except OSError as exc:
            idx["skipped"].append({"file": _rel(path), "reason": str(exc)})
            continue

        n_files += 1
        if path.suffix == ".js":
            _scan_js(path, text, idx)
        elif path.suffix == ".html":
            _scan_html(path, text, idx)
        elif path.suffix == ".py":
            _scan_py(path, text, idx)

    idx["meta"] = {
        "repo_root": str(REPO_ROOT),
        "files_indexed": n_files,
        "newest_mtime": newest,
        "built_at": time.time(),
        "build_ms": round((time.time() - t0) * 1000, 1),
        "n_symbols": len(idx["symbols"]),
        "n_routes": len(idx["routes"]),
        "n_elements": len(idx["elements"]),
    }
    logger.info(f"[devmap] indexed {n_files} files, {len(idx['symbols'])} symbols, "
                f"{len(idx['routes'])} routes in {idx['meta']['build_ms']} ms")
    return idx


def _newest_mtime() -> float:
    newest = 0.0
    for path in _iter_files():
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def _get_index(force: bool = False) -> dict:
    """Return the index, rebuilding when the tree has changed.

    The staleness check walks the tree for mtimes (a few hundred stats, ~2 ms)
    rather than trusting a TTL. A time-based cache would either serve stale line
    numbers right after an edit — the exact moment the tool matters — or rebuild
    constantly.
    """
    global _INDEX
    with _INDEX_LOCK:
        if force or not _INDEX:
            _INDEX = _build_index()
            return _INDEX
        try:
            if _newest_mtime() > (_INDEX.get("meta", {}).get("newest_mtime") or 0):
                _INDEX = _build_index()
        except Exception as exc:                     # noqa: BLE001
            logger.debug(f"[devmap] staleness check failed, serving cached index: {exc}")
        return _INDEX


# ─────────────────────────────────────────────────────────────────────────────
# GET /devmap/index
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/devmap/index", methods=["GET"])
def devmap_index():
    """The whole index, for the client to cache.

    Roughly 200–400 KB uncompressed on this tree and gzipped by any real
    deployment. One fetch on first hover beats a request per mouseover by a
    wide enough margin that the size is not worth optimising.
    """
    idx = _get_index(force=request.args.get("refresh") in ("1", "true", "yes"))
    return jsonify({
        "ok": True,
        "meta": idx["meta"],
        "symbols": idx["symbols"],
        "elements": idx["elements"],
        "handlers": idx["handlers"],
        "nav": idx["nav"],
        "nav_markup": idx["nav_markup"],
        "routes": idx["routes"],
        "skipped": idx["skipped"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /devmap/config
# -----------------------------------------------------------------------------
# The overlay's own settings, read from `visualizer.devmap` in the engine's
# input_TS.yml (see uiapp/config.devmap_settings for the precedence rules).
#
# Deliberately a separate route from /devmap/index rather than a field on it.
# The index is 200–400 KB and is fetched lazily on first hover; the config is
# ~200 bytes and is needed at page load, because it carries `enabled` and the
# overlay must not flash on before it arrives. Folding the two together would
# mean paying for the index on every page load just to learn the hotkey.
#
# Re-read on every request, not cached at import: editing input_TS.yml and
# reloading the page should take effect, the same rule the index follows for
# source files. It is one small YAML parse per page load.
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/devmap/config", methods=["GET"])
def devmap_config():
    cfg = _cfg.devmap_settings()
    if cfg.get("hotkey_error"):
        logger.warning("[devmap] %s", cfg["hotkey_error"])
    return jsonify({
        "ok": True,
        "devmap": cfg,
        "yml": _cfg.ELION_YML_PATH,
        "yml_found": _cfg.ELION_FOUND,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /devmap/lookup
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/devmap/lookup", methods=["GET"])
def devmap_lookup():
    """Resolve one symbol / element id / data-nav key / URL.

    The client resolves from its cached index and does not normally call this.
    It exists for the console and for `curl` — being able to ask
    `/devmap/lookup?symbol=_vinaDock` from a terminal while reading a bug report
    is worth the twenty lines.
    """
    idx = _get_index()
    symbol = (request.args.get("symbol") or "").strip()
    element = (request.args.get("element") or "").strip()
    nav = (request.args.get("nav") or "").strip()
    url = (request.args.get("url") or "").strip()

    if not any((symbol, element, nav, url)):
        return jsonify({"ok": False, "err": "pass one of symbol, element, nav, url"}), 400

    out = {"ok": True}
    if symbol:
        out["symbol"] = symbol
        out["definitions"] = idx["symbols"].get(symbol, [])
        out["used_by"] = idx["handlers"].get(symbol, [])
    if element:
        out["element"] = element
        out["markup"] = idx["elements"].get(element, [])
    if nav:
        out["nav"] = nav
        out["dispatch"] = idx["nav"].get(nav, [])
        out["markup"] = idx["nav_markup"].get(nav, [])
    if url:
        norm = re.sub(r"<[^>]+>", "<var>", url.split("?")[0])
        out["url"] = norm
        out["routes"] = idx["routes"].get(norm, [])
    return jsonify(out)


# ─────────────────────────────────────────────────────────────────────────────
# GET /devmap/source
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/devmap/source", methods=["GET"])
def devmap_source():
    """A few lines of context around file:line, for the tooltip's peek panel.

    Confined to the repository root and to the extensions the indexer scans.
    `Path.resolve()` before the containment check is what makes `../../etc/passwd`
    fail — a prefix test on the raw string would not.
    """
    rel = (request.args.get("file") or "").strip()
    try:
        line = int(request.args.get("line") or 1)
        context = max(0, min(40, int(request.args.get("context") or 6)))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "err": "line and context must be integers"}), 400

    if not rel:
        return jsonify({"ok": False, "err": "file is required"}), 400

    try:
        target = (REPO_ROOT / rel).resolve()
        target.relative_to(REPO_ROOT.resolve())
    except (ValueError, OSError):
        return jsonify({"ok": False, "err": "path outside the repository"}), 403

    if target.suffix not in (".js", ".html", ".py", ".yml", ".yaml", ".md"):
        return jsonify({"ok": False, "err": f"unsupported file type: {target.suffix}"}), 403
    if not target.is_file():
        return jsonify({"ok": False, "err": "not found"}), 404

    try:
        lines = target.read_text(errors="replace").splitlines()
    except OSError as exc:
        return jsonify({"ok": False, "err": str(exc)}), 500

    start = max(0, line - 1 - context)
    end = min(len(lines), line + context)
    return jsonify({
        "ok": True, "file": rel, "line": line,
        "start_line": start + 1,
        "lines": [{"n": start + k + 1, "text": lines[start + k][:400]} for k in range(end - start)],
        "total_lines": len(lines),
    })
