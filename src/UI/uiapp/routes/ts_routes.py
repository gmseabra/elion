# =============================================================================
# routes/ts_routes.py
# Thompson Sampling generator endpoints:
#   ts_run    — POST: launch one elion.py per reaction in parallel (CPU-pinned)
#   ts_status — GET SSE: stream stdout line-by-line until __DONE__
#   ts_kill   — POST: SIGTERM the elion.py subprocess(es)
#   ts_cpu    — GET SSE: stream per-core CPU% every second via psutil
# =============================================================================

import os, re, subprocess, threading, uuid, queue, signal, shutil, tempfile, json
import sys as _sys
from flask import jsonify, request, Response, stream_with_context
from uiapp import app
from uiapp import config as _tscfg

try:
    from uiapp.routes import ts_mol_render as _mol
except Exception:
    try:
        from uiapp.routes import ts_mol_render as _mol   # fallback if placed alongside
    except Exception as _mol_err:
        _mol = None
        import logging as _logging_tmp
        _logging_tmp.getLogger(__name__).warning(
            "[TS:mol] could not import ts_mol_render: %s — molecule rendering disabled",
            _mol_err)

from uiapp.routes.shared import (
    logger, _ts_jobs, _ts_lock, _ELION_CWD, _ELION_YML, _ELION_VENV,
)

# ══ Thompson Sampling Generator  /vina_visualization/ts_* ══════════════════

# Module-level job store — same pattern as _finetune_jobs
_ts_jobs: dict[str, dict] = {}
_ts_lock = threading.Lock()

# Elion project root — elion.py lives here
_ELION_CWD  = _tscfg.ELION_CWD
_ELION_YML  = _tscfg.ELION_YML
_ELION_VENV = _tscfg.ELION_VENV

# ── Reaction catalogue ─────────────────────────────────────────────────────
# Add new reactions here — key → smarts + reagent csv filenames.
# ts_run reads this to spawn parallel elion.py jobs, one per reaction.
_REACTION_CATALOGUE = {
    "rxn101_amide": {
        "short_name":    "amide",
        "smarts": "[N;D1$(N-[#6]),D2$(N(-[#6])-[#6]);$(N-[#6])!$(N-C=[O,N,S]):1].[C;D1,$(C[#6]):2](=[OD1:3])[OD1,Cl]>>[N:1][C:2](=[O:3])",
        "reagent_files": ["rxn101_1.csv", "rxn101_2.csv"],
    },
    "rxn102_buchwald": {
        "short_name":    "buchwald",
        "smarts": "[#6;a;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1][#17,#35,#53;A;D1].[#7;A;$(N[#6])!$(N=*)!$([N-])!$(N#*)!$([ND3])!$([ND4])!$(N[O,N])!$(N[C,S]=[S,O,N]):2]>>[#6:1]-[#7:2]",
        "reagent_files": ["rxn102_1.csv", "rxn102_2.csv"],
    },
    "rxn108_sonogashira": {
        "short_name":    "sonogashira",
        "smarts": "[#6;$(C=C-[#6]),$(c:c):1][#35,#53;A;D1].[#6;A;D1;$(C#C[#6,#14]):2]>>[#6:1]-[#6:2]",
        "reagent_files": ["rxn108_1.csv", "rxn108_2.csv"],
    },
    "rxn110_suzuki": {
        "short_name":    "suzuki",
        "smarts": "[#6;a;D3;$([#6](~[#6])~[#6]):2][#35;A;D1].[#6;a;D3;$([#6]([#6])[#6]):1]-[#5]([#8])[#8]>>[#6:2]-[#6:1]",
        "reagent_files": ["rxn110_1.csv", "rxn110_2.csv"],
    },
    "rxn113_sulfonamide": {
        "short_name":    "sulfonamide",
        "smarts": "[N;D1$(N-[#6]),D2$(N(-[#6])-[#6]);!$(N-C=[O,N,S]):1].[#6:2][S:3](=[O:4])(=[O:5])Cl>>[N:1][S:3](=[O:4])(=[O:5])[#6:2]",
        "reagent_files": ["rxn113_1.csv", "rxn113_2.csv"],
    },
    "rxn208_snar": {
        "short_name":    "snar",
        "smarts": "[#9,#17;A;D1][c:1]1[n,c;D2:6][n,c:5][n,c:4][n,c:3][c:2]1-[N$(N(=O)O),S$(S(=O)O),C$(C(=O)O),C$(C#N):7].[#7;A;$(N[#6])!$(N=*)!$([N-])!$(N#*)!$([ND3])!$([ND4])!$(N[O,N])!$(N[C,S]=[S,O,N]):8]>>[#7:8]-[c:1]1[n,c:6][n,c:5][n,c:4][n,c:3][c:2]1-[*:7]",
        "reagent_files": ["rxn208_1.csv", "rxn208_2.csv"],
    },
}

# ── Building-block library root ──────────────────────────────────────────────
# Where a RUN draws its reagent CSVs from. Relative entries in
# generator.TS.reagent_file_list are joined onto this; absolute entries ignore
# it. NOT the same as visualizer.bb_scan_dir, which only pre-fills the Import
# box and is usually a superset of this.
#
# Precedence:  $UI_TS_BB_BASE  >  generator.TS.bb_base  >  ELION_CWD-derived
#
# The middle level is the important one: generators/TS.py reads that SAME yml
# key, so the engine and this dashboard resolve reagents to the same directory
# by construction. Before it existed, the UI derived its base from ELION_CWD
# while the yml handed elion.py absolute paths into a different tree, and the
# two silently disagreed.
#
# Read once at import (restart Flask after editing the yml). Every failure —
# missing file, missing key, malformed YAML, non-string value — is swallowed
# and the ELION_CWD-derived default is used, so a config mistake can never
# break module import. Same contract as _resolve_output_dir/_resolve_debug_dir.
def _resolve_bb_base() -> str:
    env = os.environ.get("UI_TS_BB_BASE", "").strip()
    if env:
        return os.path.expanduser(env)
    try:
        import yaml as _yaml
        with open(os.path.join(_ELION_CWD, _ELION_YML)) as _f:
            _gen = ((_yaml.safe_load(_f) or {}).get("generator") or {})
        _ts_sec = _gen.get("TS") or {}
        _b = _ts_sec.get("bb_base")
        if isinstance(_b, str) and _b.strip():
            return os.path.expanduser(_b.strip())
    except Exception as _e:
        logger.warning("[ts_routes] generator.TS.bb_base unreadable (%s) — "
                       "falling back to %s", _e, _tscfg.TS_BB_BASE)
    return _tscfg.TS_BB_BASE

_BB_BASE    = _resolve_bb_base()


# ── Results root ─────────────────────────────────────────────────────────────
# Where a RUN writes its result CSVs. A relative generator.TS.results_filename
# is joined onto this; an absolute one ignores it. Read by generators/TS.py from
# the SAME yml key, so engine and dashboard agree by construction.
#
# Precedence:  $UI_TS_RESULTS_BASE  >  generator.TS.results_base  >  ""
#
# Empty is a legitimate answer here (unlike _BB_BASE, which has an
# ELION_CWD-derived fallback): with no base, an absolute results_filename still
# works exactly as it always did, and a relative one resolves against the elion
# process CWD. Failures are swallowed, same contract as the resolvers above.
def _resolve_results_base() -> str:
    env = os.environ.get("UI_TS_RESULTS_BASE", "").strip()
    if env:
        return os.path.expanduser(env)
    try:
        import yaml as _yaml
        with open(os.path.join(_ELION_CWD, _ELION_YML)) as _f:
            _gen = ((_yaml.safe_load(_f) or {}).get("generator") or {})
        _b = (_gen.get("TS") or {}).get("results_base")
        if isinstance(_b, str) and _b.strip():
            return os.path.expanduser(_b.strip())
    except Exception as _e:
        logger.warning("[ts_routes] generator.TS.results_base unreadable (%s) — "
                       "results_filename will be used as-is", _e)
    return ""

_RESULTS_BASE = _resolve_results_base()


def _abs_results_path(results_filename) -> str:
    """Absolutise a yml results_filename against _RESULTS_BASE.

    Needed because results_filename may now be relative ("results_TS/x.csv").
    Taking os.path.dirname() of that directly yields the bare fragment
    "results_TS", which is what would land in the UI's OUTPUT DIR box.
    """
    if not isinstance(results_filename, str) or not results_filename.strip():
        return ""
    p = os.path.expanduser(results_filename.strip())
    if not os.path.isabs(p) and _RESULTS_BASE:
        p = os.path.join(_RESULTS_BASE, p)
    return os.path.abspath(p)

# ── Output directory for dashboard JSON artifacts ──────────────────────────
# The TS backend writes two kinds of JSON under a single base "output" folder:
#     <output_dir>/TS_Session   — live-run chart session files (_write_session)
#     <output_dir>/Warmup_TS    — warmup belief-state checkpoints (_save_warmup_from_log)
# The base folder is configurable via the  visualizer: output_dir:  key in
# input_TS.yml (read once here, at import). If the key/section/file is missing
# or unreadable, it falls back to _DEFAULT_OUTPUT_DIR so a config problem can
# never break module import. The TS_Session / Warmup_TS subfolder names are
# kept stable so everything that reads them back (this module + the elion
# subprocess via TS_WARMUP_CHECKPOINT) stays consistent.
_DEFAULT_OUTPUT_DIR = _tscfg.TS_OUTPUT_DIR

def _resolve_output_dir() -> str:
    """Return the base output folder from input_TS.yml, or the default.

    Reads visualizer.output_dir from the same yml elion runs on
    (os.path.join(_ELION_CWD, _ELION_YML)). Any failure — missing file,
    missing section/key, malformed YAML, non-string value — is swallowed and
    the default is used, so a config mistake can never break module import.
    """
    try:
        import yaml as _yaml
        _yml_path = os.path.join(_ELION_CWD, _ELION_YML)
        with open(_yml_path) as _f:
            _cfg = _yaml.safe_load(_f) or {}
        _vis = _cfg.get("visualizer") or {}
        _out = _vis.get("output_dir")
        if isinstance(_out, str) and _out.strip():
            return os.path.expanduser(_out.strip())
    except Exception as _e:
        logger.warning("[ts_routes] visualizer.output_dir unreadable (%s) — "
                       "using default %s", _e, _DEFAULT_OUTPUT_DIR)
    return _DEFAULT_OUTPUT_DIR

_OUTPUT_DIR = _resolve_output_dir()
_WARMUP_DIR = os.path.join(_OUTPUT_DIR, "Warmup_TS")
_STATE_DIR  = os.path.join(_OUTPUT_DIR, "TS_Session")
os.makedirs(_STATE_DIR, exist_ok=True)

# ── Debug trace ──────────────────────────────────────────────────────────────
# <output_dir>/debug — one file per job plus a client-side log, written so the
# reagent-panel chain can be diagnosed end to end without a debugger:
#
#   elion stdout → regex match → _update_reagent → _recompute_top5
#     → history["top5"] → /ts_top5 → _ts._activeBars → #tsTsBars
#
# Every symptom of "the panel is empty" looks the same in the browser, and the
# links are split across a subprocess, a Flask route and a browser. Each link
# logs its own counts, so one file says which one is at fault. Override the
# location with visualizer.debug_dir or $UI_TS_DEBUG_DIR; set UI_TS_DEBUG=0 to
# turn it off entirely.
def _resolve_debug_dir() -> str:
    env = os.environ.get("UI_TS_DEBUG_DIR", "").strip()
    if env:
        return os.path.expanduser(env)
    try:
        import yaml as _yaml
        with open(os.path.join(_ELION_CWD, _ELION_YML)) as _f:
            _vis = ((_yaml.safe_load(_f) or {}).get("visualizer") or {})
        _d = _vis.get("debug_dir")
        if isinstance(_d, str) and _d.strip():
            return os.path.expanduser(_d.strip())
    except Exception:
        pass
    return os.path.join(_OUTPUT_DIR, "debug")


_DEBUG_DIR     = _resolve_debug_dir()
_DEBUG_ENABLED = os.environ.get("UI_TS_DEBUG", "1").strip() not in ("0", "false", "False")
_DEBUG_MAX_LINES = 20000          # per file, so a marathon run cannot fill the disk
_debug_counts: dict = {}
_debug_lock = threading.Lock()

try:
    if _DEBUG_ENABLED:
        os.makedirs(_DEBUG_DIR, exist_ok=True)
except Exception as _dbg_e:                                   # pragma: no cover
    logger.warning("[TS:debug] cannot create %s (%s) — tracing disabled", _DEBUG_DIR, _dbg_e)
    _DEBUG_ENABLED = False


# ── elion stdout patterns ────────────────────────────────────────────────────
# The reagent table that feeds `history["top5"]` — and therefore `#tsTsBars` —
# is built EXCLUSIVELY from RE_POST and RE_WINNER. The chart, the iteration
# counter and the score readout come from `[TS:stats]`, which this module emits
# itself, so they keep working perfectly if these drift. That asymmetry is why a
# format change shows up as "the reagent panel is empty" and nothing else.
# tests/test_ts_parse_patterns.py pins them against real captured lines.
#
#   [evaluate] score=8.8478
#   post-update 17089792 μ=6.6926 σ=0.7219 n=3
#   winner | cycle_id=0 | reagent=49830974 | sampled=9.50 | mu=5.88 | std=0.88 | num_scores=2
#
# μ/σ are matched as a character class ([μuµ], [σs]) because elion emits the
# Unicode glyphs and U+00B5 MICRO SIGN and U+03BC GREEK SMALL LETTER MU are
# different codepoints that render identically.
RE_SCORE_LINE = re.compile(r'\[evaluate\]\s+score=([\d.eE+\-]+)')     # PRIMARY (matches ts_worker.js)
RE_SCORE_ALT  = re.compile(r'score:\s*([\d.eE+\-]+)\s*\|\s*smiles:')  # fallback if no [evaluate]
RE_POST = re.compile(
    r'post-update\s+(\S+)\s+[μuµ]=([\d.eE+\-]+)\s+[σs]=([\d.eE+\-]+)\s+n=(\d+)')
RE_WINNER = re.compile(
    r'winner\s*\|\s*cycle_id=(\d+)\s*\|\s*reagent=(\S+)\s*\|\s*'
    r'sampled=([\d.eE+\-]+)\s*\|\s*mu=([\d.eE+\-]+)\s*\|\s*'
    r'std=([\d.eE+\-]+)\s*\|\s*num_scores=(\d+)')


def _dbg(name: str, msg: str) -> None:
    """Append one line to <debug>/<name>.log. Never raises, never blocks a run."""
    if not _DEBUG_ENABLED:
        return
    try:
        with _debug_lock:
            n = _debug_counts.get(name, 0)
            if n >= _DEBUG_MAX_LINES:
                if n == _DEBUG_MAX_LINES:
                    _debug_counts[name] = n + 1
                    with open(os.path.join(_DEBUG_DIR, f"{name}.log"), "a") as fh:
                        fh.write(f"--- capped at {_DEBUG_MAX_LINES} lines ---\n")
                return
            _debug_counts[name] = n + 1
        from datetime import datetime as _dt
        stamp = _dt.now().strftime("%H:%M:%S.%f")[:-3]
        with open(os.path.join(_DEBUG_DIR, f"{name}.log"), "a") as fh:
            fh.write(f"[{stamp}] {msg}\n")
    except Exception:
        pass


_dbg("session", "=" * 70)
_dbg("session", f"ts_routes imported | ELION_CWD={_ELION_CWD} | yml={_ELION_YML}")
_dbg("session", f"OUTPUT_DIR={_OUTPUT_DIR} | STATE_DIR={_STATE_DIR} | DEBUG_DIR={_DEBUG_DIR}")

# Version banner — visible in Flask logs immediately on import
logger.info("=" * 60)
logger.info("[ts_routes] MODULE LOADED — version 2026-06-13-v1")
logger.info("[ts_routes] molecule rendering: %s",
            "ENABLED" if _mol is not None else "DISABLED (ts_mol_render import failed)")
logger.info("[ts_routes] OUTPUT_DIR = %s", _OUTPUT_DIR)
logger.info("[ts_routes] STATE_DIR  = %s", _STATE_DIR)
logger.info("[ts_routes] WARMUP_DIR = %s", _WARMUP_DIR)
logger.info("=" * 60)

def _purge_corrupt_sessions() -> None:
    """Remove session files where points are doubled (written before the duplicate-point fix).
    Corrupt files have pts.length ≈ 2 × scores.length because both stdout and stderr score
    lines used to each append a point. Heuristic: if pts > 2 and scores > 0 and pts > scores * 1.4,
    the file is corrupt.
    Called once at module import so stale files never reach the frontend.
    """
    import glob as _glob, json as _json
    removed = 0
    for path in _glob.glob(os.path.join(_STATE_DIR, "*.json")):
        try:
            with open(path) as f:
                s = _json.load(f)
            hist   = s.get("history", {})
            pts    = hist.get("points", [])
            scores = hist.get("scores",  [])
            if len(pts) > 2 and len(scores) > 0 and len(pts) > len(scores) * 1.4:
                os.remove(path)
                removed += 1
                logger.info("[TS] purged corrupt session %s (pts=%d scores=%d)",
                            s.get("job_id","?"), len(pts), len(scores))
        except Exception:
            pass
    if removed:
        logger.info("[TS] purged %d corrupt session file(s)", removed)

def _cleanup_tmp_files() -> None:
    """Remove orphaned .json.tmp files left by interrupted writes (e.g. after kill or crash)."""
    import glob as _glob
    for path in _glob.glob(os.path.join(_STATE_DIR, '*.json.tmp')):
        try:
            os.remove(path)
            logger.info('[TS] removed orphaned tmp file: %s', path)
        except Exception:
            pass

_purge_corrupt_sessions()
_cleanup_tmp_files()

def _session_path(job_id: str) -> str:
    return os.path.join(_STATE_DIR, f"{job_id}.json")

def _write_session(job_id: str, job: dict) -> None:
    """Persist job session to disk atomically."""
    path = _session_path(job_id)
    tmp  = path + ".tmp"
    try:
        payload = {
            "job_id":     job_id,
            "rxn_key":    job.get("rxn_key",""),
            "short_name": job.get("short_name",""),
            "bb_names":   job.get("bb_names",[]),
            "cpu_cores":  job.get("cpu_cores",[]),
            "pid":        job.get("pid"),
            "status":     job.get("status","running"),
            "launch_idx": job.get("launch_idx", 0),
            "history":    job.get("history", {"points": [], "scores": [], "reagents": {}}),
        }
        with open(tmp, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("[TS] session write failed for %s: %s", job_id, e)

def _delete_session(job_id: str) -> None:
    try:
        os.remove(_session_path(job_id))
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("[TS] session delete failed for %s: %s", job_id, e)

def _load_sessions() -> list:
    """Load all running session files; skip orphaned ones (process dead).
    Returns (sessions, debug_log) where debug_log is a list of strings
    surfaced to the browser via /ts_active for console visibility.
    """
    import glob as _glob
    sessions  = []
    debug_log = []

    import os as _os
    import datetime as _dt
    _now = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    all_files  = _glob.glob(_os.path.join(_STATE_DIR, "*.json"))
    tmp_files  = _glob.glob(_os.path.join(_STATE_DIR, "*.json.tmp"))
    # Also do a direct os.listdir to cross-check glob
    try:
        _listdir = [f for f in _os.listdir(_STATE_DIR) if f.endswith('.json')]
    except Exception as _le:
        _listdir = [f"listdir error: {_le}"]
    debug_log.append(f"[{_now}] _load_sessions: STATE_DIR={_STATE_DIR}")
    debug_log.append(f"[{_now}] glob found {len(all_files)} .json: {[_os.path.basename(p) for p in all_files]}")
    debug_log.append(f"[{_now}] listdir found {len(_listdir)} .json: {_listdir}")
    debug_log.append(f"[{_now}] glob found {len(tmp_files)} .tmp: {[_os.path.basename(p) for p in tmp_files]}")
    debug_log.append(f"[{_now}] STATE_DIR exists={_os.path.isdir(_STATE_DIR)} readable={_os.access(_STATE_DIR, _os.R_OK)}")
    logger.info("[TS:_load_sessions] found %d json, %d tmp", len(all_files), len(tmp_files))

    for path in all_files:
        try:
            with open(path) as f:
                s = json.load(f)
            pid      = s.get("pid")
            alive    = False
            kill_err = None
            if pid is None:
                # pid=None means session was written before Popen completed.
                # File is fresh — treat as alive, pid will be updated momentarily.
                alive    = True
                kill_err = "pid=None (pre-Popen write, assume starting)"
            elif pid:
                try:
                    _os.kill(pid, 0)
                    alive = True
                    kill_err = "ok"
                except PermissionError as _ke:
                    proc_exists = _os.path.exists(f"/proc/{pid}")
                    alive    = proc_exists
                    kill_err = f"PermissionError({_ke}) → /proc/{pid} exists={proc_exists}"
                except ProcessLookupError as _ke:
                    alive    = False
                    kill_err = f"ProcessLookupError (pid not found)"
            else:
                kill_err = "pid=0 (invalid)"

            msg = (f"session {s.get('job_id','?')[:8]} "
                   f"pid={pid} alive={alive} reason={kill_err}")
            debug_log.append(msg)
            logger.info("[TS:_load_sessions] %s", msg)

            if not alive:
                debug_log.append(f"  → REMOVING {_os.path.basename(path)}")
                _os.remove(path)
                continue
            sessions.append(s)
        except Exception as e:
            debug_log.append(f"  → EXCEPTION loading {_os.path.basename(path)}: {e}")
            logger.warning("[TS:_load_sessions] failed to load %s: %s", path, e)

    # Sort by launch_idx so reconnect rebuilds tabs in the SAME order as launch.
    # Without this, glob() returns files in arbitrary filesystem order, so with
    # multiple jobs (e.g. Amide, Suzuki, Amide, Suzuki) the tabs map to the wrong
    # jobs after reload and the chart/scores appear swapped.
    sessions.sort(key=lambda s: s.get("launch_idx", 0))
    debug_log.append(f"_load_sessions: returning {len(sessions)} live sessions "
                     f"(order: {[s.get('launch_idx') for s in sessions]})")
    logger.info("[TS:_load_sessions] returning %d live sessions", len(sessions))
    _load_sessions._last_debug = debug_log
    return sessions




def _warmup_cache_path(rxn_key: str, timestamp: str = "") -> str:
    """Path to the warmup checkpoint file. Uses short_name + timestamp."""
    short_name = _REACTION_CATALOGUE.get(rxn_key, {}).get("short_name", rxn_key)
    if timestamp:
        return os.path.join(_WARMUP_DIR, f"{short_name}_{timestamp}_warmup.json")
    # No timestamp — find the latest existing checkpoint for this reaction
    import glob as _glob
    pattern = os.path.join(_WARMUP_DIR, f"{short_name}_*_warmup.json")
    matches = sorted(_glob.glob(pattern), reverse=True)  # newest first by name
    return matches[0] if matches else os.path.join(_WARMUP_DIR, f"{short_name}_warmup.json")


def _warmup_exists(rxn_key: str) -> bool:
    """True if at least one completed warmup checkpoint exists for this reaction."""
    short_name = _REACTION_CATALOGUE.get(rxn_key, {}).get("short_name", rxn_key)
    import glob as _glob
    pattern = os.path.join(_WARMUP_DIR, f"{short_name}_*_warmup.json")
    matches = [p for p in _glob.glob(pattern) if os.path.getsize(p) > 0]
    return len(matches) > 0


def _save_warmup_from_log(rxn_key: str, log_lines: list, timestamp: str = "") -> None:
    """
    Parse elion's stdout lines and save warmup checkpoint immediately.
    Called as soon as the warmup-summary log lines appear in the stream.

    Parses (INFO level, always present):
      '[warm_up] prior_mean=X | prior_std=X | num_warmup_scores=N'
    And (DEBUG level, only with log_level=DEBUG):
      '[warm_up] component I, reagent J (NAME): init mu=X, std=X, num_scores=N'
    """
    import json as _json, re as _re2

    prior_mean = prior_std = None
    components: dict[int, list[dict]] = {}

    pat_prior   = _re2.compile(
        r'\[warm_up\]\s+prior_mean=([0-9.eE+\-]+)\s*\|\s*prior_std=([0-9.eE+\-]+)')
    pat_reagent = _re2.compile(
        r'\[warm_up\]\s+component\s+(\d+),\s+reagent\s+\d+\s+\(([^)]+)\):\s+'
        r'init\s+mu=([0-9.eE+\-]+),\s+std=([0-9.eE+\-]+),\s+num_scores=(\d+)')

    for line in log_lines:
        m = pat_prior.search(line)
        if m:
            prior_mean, prior_std = float(m.group(1)), float(m.group(2))
            continue
        m = pat_reagent.search(line)
        if m:
            comp_idx  = int(m.group(1))
            name      = m.group(2).strip()
            mu        = float(m.group(3))
            std       = float(m.group(4))
            n         = int(m.group(5))
            known_var = (prior_std ** 2) if prior_std is not None else None
            components.setdefault(comp_idx, []).append({
                'reagent_name': name,
                'current_mean': mu,
                'current_std':  std,
                'known_var':    known_var,
                'num_scores':   n,
            })

    if prior_mean is None:
        logger.warning("[TS] warmup checkpoint: no [warm_up] prior line found for %s", rxn_key)
        return None

    checkpoint = {
        'rxn_key':      rxn_key,
        'timestamp':    timestamp,
        'prior_mean':   prior_mean,
        'prior_std':    prior_std,
        'known_var':    prior_std ** 2,
        'components':   components,
        'n_components': len(components),
        'n_reagents':   sum(len(v) for v in components.values()),
    }

    os.makedirs(_WARMUP_DIR, exist_ok=True)
    path = _warmup_cache_path(rxn_key, timestamp)
    try:
        with open(path, 'w') as f:
            _json.dump(checkpoint, f, indent=2)
        logger.info("[TS] warmup checkpoint saved: %s (%d reagents, %d components)",
                    path, checkpoint['n_reagents'], checkpoint['n_components'])
        return path
    except Exception as e:
        logger.warning("[TS] could not save warmup checkpoint: %s", e)
        return None


def _write_warmup_loader(checkpoint_path: str) -> None:
    """Write warmup_checkpoint_loader.py into _ELION_CWD.
    Patches ThompsonSampler.__init__ to install a per-instance warm_up override
    immediately when the sampler is created — before TS.py ever calls warm_up().
    """
    loader_path = os.path.join(_ELION_CWD, "warmup_checkpoint_loader.py")
    loader_code = r"""
import os, json, sys

print('[LOADER] warmup_checkpoint_loader.py executing', flush=True)
_ckpt_path = os.environ.get("TS_WARMUP_CHECKPOINT", "")
print(f'[LOADER] checkpoint path: {_ckpt_path!r}', flush=True)

if not _ckpt_path or not os.path.isfile(_ckpt_path):
    print('[LOADER] no checkpoint — warmup runs normally', flush=True)
else:
    try:
        with open(_ckpt_path) as _f:
            _ckpt = json.load(_f)

        _prior_mean = _ckpt["prior_mean"]
        _prior_std  = _ckpt["prior_std"]
        _known_var  = _prior_std ** 2
        _belief_by_name = {}
        for _comp_list in _ckpt.get("components", {}).values():
            for _r in _comp_list:
                _belief_by_name[_r["reagent_name"]] = _r

        print(f'[LOADER] loaded: prior_mean={_prior_mean:.4f} prior_std={_prior_std:.4f} '
              f'n_reagents={len(_belief_by_name)}', flush=True)

        # thompson_sampling.py lives in generators/TS/, which isn't on sys.path
        # yet (the loader runs before elion.py sets up its path). Add it.
        if 'thompson_sampling' not in sys.modules:
            _ts_dir_cands = [c for c in (
                __TS_ENGINE_DIR__,
                os.environ.get('TS_ENGINE_DIR', ''),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generators', 'TS'),
            ) if c]
            for _d in _ts_dir_cands:
                if os.path.isfile(os.path.join(_d, 'thompson_sampling.py')):
                    if _d not in sys.path:
                        sys.path.insert(0, _d)
                        print(f'[LOADER] added to sys.path: {_d}', flush=True)
                    break

        import thompson_sampling as _ts_mod

        _orig_init = _ts_mod.ThompsonSampler.__init__

        def _patched_init(self, *args, **kwargs):
            # Call original __init__ first
            _orig_init(self, *args, **kwargs)
            # Then install warm_up override directly on this instance
            def _instance_warm_up(num_warmup_trials, *a, **kw):
                print('[LOADER] instance warm_up called — injecting checkpoint beliefs', flush=True)
                import traceback as _tb
                try:
                    restored = skipped = 0
                    for reagent_list in self.reagent_lists:
                        for reagent in reagent_list:
                            belief = _belief_by_name.get(reagent.reagent_name)
                            if belief:
                                reagent.current_phase  = "search"
                                reagent.current_mean   = belief["current_mean"]
                                reagent.current_std    = belief["current_std"]
                                reagent.known_var      = belief.get("known_var") or _known_var
                                reagent.num_scores     = belief["num_scores"]
                                reagent.initial_scores = []
                                restored += 1
                            else:
                                reagent.current_phase  = "search"
                                reagent.current_mean   = _prior_mean
                                reagent.current_std    = _prior_std
                                reagent.known_var      = _known_var
                                reagent.num_scores     = 0
                                reagent.initial_scores = []
                                skipped += 1
                    self._warmup_std = _prior_std
                    print(f'[LOADER] restored={restored} skipped={skipped}', flush=True)
                    # Return synthetic result — callers expect list of [score, smiles, name]
                    return [[_prior_mean, "checkpoint", "checkpoint"]]
                except Exception as _e:
                    print(f'[LOADER] ERROR: {_e}', flush=True)
                    _tb.print_exc()
                    # Fall back to real warmup
                    return _ts_mod.ThompsonSampler.warm_up(self, num_warmup_trials, *a, **kw)
            # Assign directly — instance attributes bypass the descriptor protocol
            # so types.MethodType is not needed (and was causing a double-self bug:
            # TS.py calls ts.warm_up(n) → bound method passes self_ but n was left
            # unbound → TypeError: missing 1 required positional argument 'n').
            self.warm_up = _instance_warm_up
            print('[LOADER] instance.warm_up override installed', flush=True)

        _ts_mod.ThompsonSampler.__init__ = _patched_init
        print('[LOADER] ThompsonSampler.__init__ patched — override will install on every new instance', flush=True)

    except Exception as _e:
        import traceback
        print(f'[LOADER] FATAL ERROR loading checkpoint: {_e}', flush=True)
        traceback.print_exc()
"""
    # `loader_code` is a RAW, NON-f string — deliberately, because the loader
    # body is full of its own f-strings ('{_ckpt_path!r}', '{len(...)}') that an
    # f-template would try to evaluate here. The consequence is that anything
    # meant to come from THIS module has to be substituted explicitly.
    #
    # This bit me: a previous edit replaced a hardcoded engine path with a bare
    # `_tscfg.TS_ENGINE_DIR` inside the literal. `_tscfg` is defined in
    # ts_routes.py, not in the generated file, so the token was written out
    # verbatim and every run died with
    #     warmup_checkpoint_loader.py line 29: NameError: name '_tscfg' is not defined
    # which killed checkpoint loading — so warmup re-ran from scratch every time.
    #
    # repr() rather than a bare substitution: it quotes and escapes the path, so
    # a directory containing a quote or backslash cannot produce broken source.
    loader_code = loader_code.replace("__TS_ENGINE_DIR__", repr(_tscfg.TS_ENGINE_DIR))
    if "__TS_ENGINE_DIR__" in loader_code or "_tscfg" in loader_code:
        # Fail loudly here rather than writing a file that NameErrors at runtime.
        logger.error("[TS] loader template still references this module's names "
                     "after substitution — refusing to write %s", loader_path)
        return

    try:
        with open(loader_path, 'w') as f:
            f.write(loader_code.strip() + '\n')
        logger.info("[TS] wrote warmup checkpoint loader: %s", loader_path)
    except Exception as e:
        logger.warning("[TS] could not write warmup loader: %s", e)



def _patch_yml_for_reaction(yml_src: str, tmp_dir: str, rxn_key: str,
                             smarts: str, reagent_files: list,
                             iters, rxn_index: int,
                             output_dir: str = None) -> str:
    """Copy yml to a temp file and patch reaction_smarts, reagent_file_list,
    results_filename, num_ts_iterations, log_level=DEBUG.
    If output_dir is given, results_filename = <output_dir>/<rxn_key>.csv
    """
    yml_to_run = os.path.join(tmp_dir, f"input_TS_{rxn_key}.yml")
    shutil.copy2(yml_src, yml_to_run)
    content = open(yml_to_run).read()

    # Patch reaction_smarts
    content = re.sub(
        r'(reaction_smarts\s*:\s*).*',
        lambda m: m.group(1) + f'"{smarts}"',
        content,
    )
    # Patch reagent_file_list — replace existing list entries
    indent = "      "
    new_list = f"\n{indent}- " + f"\n{indent}- ".join(
        os.path.join(_BB_BASE, f) for f in reagent_files
    )
    content = re.sub(
        r'(reagent_file_list\s*:)[^\n]*(\n(?:[ \t]*-[^\n]*\n)*)',
        lambda m: m.group(1) + new_list + "\n",
        content,
    )
    # Patch results_filename
    if output_dir:
        from datetime import datetime as _dt
        ts_str     = _dt.now().strftime("%Y%m%d_%H%M%S")
        short_name = _REACTION_CATALOGUE.get(rxn_key, {}).get("short_name", rxn_key)
        # Placeholder name at launch — will be renamed after run with mean_std
        filename   = f"{short_name}_{ts_str}_tmp.csv"
        new_results = os.path.join(output_dir.rstrip('/'), filename)
        content = re.sub(
            r'(results_filename\s*:\s*"?)([^"\n]+)',
            lambda m: m.group(1) + new_results,
            content,
        )
    else:
        # Default: insert rxn_key suffix before .csv
        new_results = None
        content = re.sub(
            r'(results_filename\s*:\s*"?)([^"\n]+)',
            lambda m: m.group(1) + re.sub(r'(\.csv)$', f'_{rxn_key}\\1', m.group(2)),
            content,
        )
    # Patch num_ts_iterations if supplied
    if iters:
        content = re.sub(
            r'(num_ts_iterations\s*:\s*)\d+',
            lambda m: m.group(1) + str(int(iters)),
            content,
        )
    # When checkpoint exists, warmup_checkpoint_loader patches warm_up() directly
    # so we do NOT set num_warmup_trials:0 (that causes empty warmup_scores crash)

    # Force DEBUG log level
    if re.search(r'log_level\s*:', content):
        content = re.sub(r'(log_level\s*:\s*)\S+', r'\g<1>DEBUG', content)
    else:
        content = content.rstrip() + '\n    log_level: DEBUG\n'
    open(yml_to_run, 'w').write(content)
    return yml_to_run, new_results if output_dir else None


def _run_ts_job(job_id: str, yml_path: str, extra_env: dict,
                cpu_cores: list = None,
                results_path: str = None) -> None:
    """
    Worker thread: runs  python elion.py -i <yml>  in _ELION_CWD.
    If results_path is given, renames the output CSV after successful
    completion to include mean and std of the score column:
      <short_name>_<timestamp>_mean<X.XX>_std<X.XX>.csv
    """
    import subprocess, re as _re_ts
    q = _ts_jobs[job_id]["queue"]

    def push(line: str):
        q.put(line)

    import sys as _sys
    python = _ELION_VENV if os.path.isfile(_ELION_VENV) else _sys.executable

    cmd = [python, "elion.py", "-i", yml_path]
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONPATH"] = _ELION_CWD + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    # ── WHICH interpreter is about to run the engine, and can it see the GPU ──
    # This single line would have saved several rounds of guessing.
    #
    # ELION_VENV defaults to `sys.executable` — the interpreter running FLASK.
    # That is the UI's environment, not necessarily the compute environment. On
    # this box they differ: Flask runs in `elion-ui` (py3.10, torch built for
    # CUDA 13, which the 525 driver cannot initialise -> CPU), while the engine
    # is meant to run in `elion_backend` (py3.11, working CUDA). Same code, same
    # config, ~90x apart — and NOTHING in any log said which one was used. The
    # only trace was a filesystem path buried inside an unrelated UserWarning.
    #
    # `input_TS.yml` already points the GIGN scorer at `elion_backend`
    # (pose.gign_conda_env); TS had no equivalent, so it silently inherited
    # Flask's.
    #
    # Probe runs with a hard timeout and never raises: a diagnostic must not be
    # able to stop a run.
    try:
        _probe = subprocess.run(
            [python, "-c",
             "import sys,warnings;warnings.filterwarnings('ignore');"
             "v='%d.%d'%sys.version_info[:2]\n"
             "try:\n"
             " import torch;t=torch.__version__;c=torch.cuda.is_available();"
             "d=torch.cuda.get_device_name(0) if c else 'CPU'\n"
             "except Exception as e:\n t,c,d='(no torch)',False,str(e)[:40]\n"
             "print(f'{sys.executable}|{v}|{t}|{c}|{d}')"],
            capture_output=True, text=True, timeout=60, env=env)
        _info = (_probe.stdout or "").strip().split("|")
        if len(_info) == 5:
            _exe, _pyv, _tv, _cuda, _dev = _info
            _ok = (_cuda == "True")
            push(f"[DEBUG:env] interpreter={_exe}")
            push(f"[DEBUG:env] python={_pyv} torch={_tv} cuda_available={_cuda} device={_dev}")
            if not _ok:
                push("⚠ [DEBUG:env] engine will score on CPU. If another env on this "
                     "box has working CUDA, point the engine at it:")
                push("⚠ [DEBUG:env]     export ELION_VENV=/path/to/that/env/bin/python")
                push(f"⚠ [DEBUG:env] (ELION_VENV is currently "
                     f"{'set' if os.environ.get('ELION_VENV') else 'UNSET, so it defaulted to Flask''s own interpreter'})")
        else:
            push(f"[DEBUG:env] interpreter={python} (probe returned nothing usable)")
    except Exception as _e:
        push(f"[DEBUG:env] interpreter={python} (probe failed: {type(_e).__name__})")

    rxn_key_val = extra_env.get("TS_RXN_KEY", "")
    ckpt_path   = _warmup_cache_path(rxn_key_val) if rxn_key_val else None

    push(f"[DEBUG] python={python}")
    push(f"[DEBUG] cwd={_ELION_CWD}")
    push(f"[DEBUG] yml={yml_path} exists={os.path.isfile(yml_path)}")
    push(f"[DEBUG] rxn_key={rxn_key_val}")
    push(f"[DEBUG] ckpt_path={ckpt_path} exists={ckpt_path and os.path.isfile(ckpt_path)}")

    # ── DEBUG: verify reagent building-block CSVs exist & are non-empty ──────
    # "Run complete — 0 molecules" almost always means the reagent files never
    # loaded, so warmup scores nothing and no checkpoint can be written. Surface
    # each file's path / existence / size / line-count straight to the UI log
    # (these [DEBUG:reagents] lines are mirrored into the 💬 mini-log panel).
    _cat_entry     = _REACTION_CATALOGUE.get(rxn_key_val, {})
    _reagent_files = _cat_entry.get("reagent_files", [])
    push(f"[DEBUG:reagents] rxn_key={rxn_key_val!r} "
         f"short_name={_cat_entry.get('short_name','?')!r} "
         f"in_catalogue={rxn_key_val in _REACTION_CATALOGUE} "
         f"reagent_files={_reagent_files}")
    push(f"[DEBUG:reagents] _BB_BASE={_BB_BASE}")
    if not _reagent_files:
        push(f"[DEBUG:reagents] ⚠ NO reagent_files for {rxn_key_val!r} — reaction missing "
             f"from _REACTION_CATALOGUE? known keys: {list(_REACTION_CATALOGUE.keys())}")
    for _rf in _reagent_files:
        _rp = _rf if os.path.isabs(_rf) else os.path.join(_BB_BASE, _rf)
        try:
            if os.path.isfile(_rp):
                _sz = os.path.getsize(_rp)
                try:
                    with open(_rp, 'r', errors='replace') as _fh:
                        _nlines = sum(1 for _ in _fh)
                except Exception:
                    _nlines = -1
                push(f"[DEBUG:reagents] ✓ {_rp} — {_sz} bytes, {_nlines} lines")
                if _sz == 0 or 0 <= _nlines <= 1:
                    push(f"[DEBUG:reagents] ⚠ {_rp} is EMPTY or header-only — "
                         f"warmup will score 0 molecules")
            else:
                push(f"[DEBUG:reagents] ✗ MISSING: {_rp}")
        except Exception as _e:
            push(f"[DEBUG:reagents] ✗ error checking {_rp}: {_e}")
    # Echo the reagent_file_list + reaction_smarts actually written into the yml
    try:
        _yml_txt = open(yml_path).read()
        _m_smarts = re.search(r'reaction_smarts\s*:\s*(.+)', _yml_txt)
        push(f"[DEBUG:reagents] yml reaction_smarts = "
             f"{(_m_smarts.group(1).strip()[:140] if _m_smarts else '?? not found')}")
        _yml_reagents = re.findall(r'^\s*-\s*(\S*\.csv)\s*$', _yml_txt, re.M)
        push(f"[DEBUG:reagents] yml reagent_file_list = {_yml_reagents}")
    except Exception as _e:
        push(f"[DEBUG:reagents] could not read patched yml {yml_path}: {_e}")

    if ckpt_path and os.path.isfile(ckpt_path):
        env["TS_WARMUP_CHECKPOINT"] = ckpt_path
        _write_warmup_loader(ckpt_path)
        wrapper_py = os.path.join(_ELION_CWD, f"_warmup_wrapper_{rxn_key_val}.py")
        loader_py  = os.path.join(_ELION_CWD, "warmup_checkpoint_loader.py")
        wrapper_content = (
            f"import sys, os, logging\n"
            f"print('[WRAPPER] starting', flush=True)\n"
            f"# Redirect ALL logging levels (including DEBUG) to stdout\n"
            f"# so elion's [evaluate] score lines appear in stdout not stderr.\n"
            f"logging.basicConfig(\n"
            f"    level=logging.DEBUG,\n"
            f"    stream=sys.stdout,\n"
            f"    force=True,\n"
            f"    format='%(levelname)s %(name)s: %(message)s'\n"
            f")\n"
            f"sys.argv = ['elion.py', '-i', {repr(yml_path)}]\n"
            f"os.environ['TS_WARMUP_CHECKPOINT'] = {repr(ckpt_path)}\n"
            f"\n"
            f"# Import and patch BEFORE any elion modules load\n"
            f"print('[WRAPPER] sys.path before loader: ' + str(sys.path[:6]), flush=True)\n"
            f"import warmup_checkpoint_loader\n"
            f"print('[WRAPPER] loader done, running elion.py', flush=True)\n"
            f"\n"
            f"# Use exec instead of runpy to avoid module caching issues\n"
            f"with open('elion.py') as _f:\n"
            f"    _code = _f.read()\n"
            f"exec(compile(_code, 'elion.py', 'exec'), {{'__name__': '__main__', '__file__': 'elion.py'}})\n"
        )
        try:
            with open(wrapper_py, 'w') as _wf:
                _wf.write(wrapper_content)
            cmd = [python, wrapper_py]
            push(f"[DEBUG] warmup checkpoint active — wrapper: {wrapper_py}")
            push(f"[DEBUG] loader exists={os.path.isfile(loader_py)}")
            push(f"[DEBUG] wrapper content:\n{wrapper_content}")

            # Quick syntax-check: try importing the loader before launching
            import subprocess as _sp
            chk = _sp.run(
                [python, "-c",
                 f"import sys; sys.path.insert(0,{repr(_ELION_CWD)}); "
                 f"import warmup_checkpoint_loader; print('loader OK')"],
                capture_output=True, text=True, cwd=_ELION_CWD, env=env, timeout=10
            )
            push(f"[DEBUG] loader check stdout: {chk.stdout.strip()}")
            if chk.stderr.strip():
                push(f"[DEBUG] loader check stderr: {chk.stderr.strip()}")
            if chk.returncode != 0:
                push(f"[DEBUG] loader check FAILED (rc={chk.returncode}) — falling back to plain elion")
                cmd = [python, "elion.py", "-i", yml_path]
        except Exception as _we:
            push(f"[DEBUG] wrapper write/check failed: {_we} — running without checkpoint")
            cmd = [python, "elion.py", "-i", yml_path]

    # Force Python logging to stdout so all levels (DEBUG/INFO) go to proc.stdout
    # Without this, DEBUG messages like [evaluate] go to stderr, not stdout
    env["PYTHONUNBUFFERED"] = "1"

    logger.info("[TS] job %s starting: %s (cwd=%s)", job_id, " ".join(cmd), _ELION_CWD)
    push(f"$ cd {_ELION_CWD} && {' '.join(cmd)}")

    # Timestamp for this job — used to name the warmup checkpoint file
    from datetime import datetime as _dt
    _job_timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")

    # Write session file BEFORE Popen — ensures reconnect finds it immediately
    # even before the process starts. pid=None here, updated after Popen below.
    with _ts_lock:
        _ts_jobs[job_id].setdefault("history", {"points": [], "scores": [], "reagents": {}})
    _write_session(job_id, _ts_jobs.get(job_id, {}))
    push(f"[DEBUG:pre-popen] session file written before Popen")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # separate stderr so tracebacks don't get swallowed
            bufsize=0,
            cwd=_ELION_CWD,
            env=env,
        )
        with _ts_lock:
            _ts_jobs[job_id]["pid"] = proc.pid
        # Update session with real pid now that process is running
        _write_session(job_id, _ts_jobs.get(job_id, {}))
        push(f"[DEBUG:L524] job_id={job_id} proc.pid={proc.pid}")
        push(f"[DEBUG:L524] cmd={cmd}")
        import time as _t2; _t2.sleep(0.1)
        push(f"[DEBUG:L524] proc.poll()={proc.poll()} (None=running, int=already exited!)")
        push(f"[DEBUG:L524] pid stored in job={_ts_jobs.get(job_id,{}).get('pid')}")
        push(f"[DEBUG] pid={proc.pid} cmd={' '.join(str(c) for c in cmd)}")

        # Stream stderr in a background thread so we don't deadlock
        import threading as _thr, queue as _errq
        _stderr_lines = []
        _stderr_queue = _errq.Queue()

        def _drain_stderr():
            _stderr_count = [0]
            try:
                for raw in proc.stderr:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        _stderr_count[0] += 1
                        _stderr_lines.append(line)
                        _stderr_queue.put(line)
                        # Store AND push first 5 stderr lines
                        if "_dbg_first_stderr" not in _history:
                            _history["_dbg_first_stderr"] = []
                        if len(_history["_dbg_first_stderr"]) < 5:
                            _history["_dbg_first_stderr"].append(line[:120])
                            push(f"[STDERR:#{len(_history['_dbg_first_stderr'])}] {line!r}")
                        # Flag if [evaluate] appears in stderr
                        if "[evaluate]" in line:
                            _history["_dbg_evaluate_seen"] += 1
                            push(f"[STDERR:evaluate!] {line!r}")
                            _history["_dbg_first_stderr"].append(f"[evaluate]{line[:120]}")
                        push(f"[STDERR] {line}")
            except Exception:
                pass
        _thr.Thread(target=_drain_stderr, daemon=True).start()

        # CRITICAL: drain stdout in a DEDICATED thread, decoupled from parsing.
        # The OS pipe buffer is ~64KB; if elion writes faster than we read, the
        # buffer fills and elion BLOCKS on write() — the process sleeps at 0% CPU
        # until we drain. Previously the same thread both read the pipe AND ran
        # the (increasingly expensive) per-line parsing/top5-sort, so once
        # parsing slowed down the pipe backed up and the elion subprocess froze.
        # Now this thread does nothing but split the pipe into lines and enqueue
        # them, so the pipe is always drained no matter how slow parsing is.
        # UNBOUNDED so the drain's put() can NEVER block. The drain exists to
        # keep the OS pipe empty so elion never blocks on write(). A bounded queue
        # defeats that: when the SSE consumer lags on a long run, the queue fills,
        # put() blocks, the drain stops reading the pipe, the 64KB pipe fills, and
        # elion blocks anyway. Unbounded means the drain always keeps draining;
        # the queue absorbs transient consumer lag. Memory stays bounded in
        # practice because the consumer keeps up on average (session writes are
        # time-throttled), so this only buffers short bursts, not the whole run.
        _stdout_queue: "queue.Queue[str|None]" = queue.Queue()

        def _drain_stdout():
            _dbuf = b""
            try:
                while True:
                    chunk = proc.stdout.read(65536)
                    if not chunk:
                        break
                    _dbuf += chunk
                    _dparts = re.split(rb'[\r\n]+', _dbuf)
                    _dbuf = _dparts[-1]
                    for part in _dparts[:-1]:
                        line = part.decode("utf-8", errors="replace").strip()
                        if line:
                            _stdout_queue.put(line)
            except Exception:
                pass
            finally:
                if _dbuf:
                    line = _dbuf.decode("utf-8", errors="replace").strip()
                    if line:
                        _stdout_queue.put(line)
                _stdout_queue.put(None)   # sentinel: stdout closed
        _thr.Thread(target=_drain_stdout, daemon=True).start()

        # Pin CPU affinity after spawn (best-effort)
        if cpu_cores:
            try:
                import psutil
                psutil.Process(proc.pid).cpu_affinity(cpu_cores)
                push(f"[TS] pinned pid {proc.pid} to cores {cpu_cores}")
            except Exception as e:
                push(f"[TS] cpu_affinity skipped: {e}")
        buf = b""
        _all_log_lines:     list[str] = []
        _warmup_saved                 = False
        _warmup_warned                = False  # True once a "no scores" warning has been pushed
        _warmup_window:     list[str] = []   # lines buffered for checkpoint extraction
        _prior_seen                   = False  # once True, stop discarding window lines

        # Compact history for page-reload restoration.
        # Point _history at the same dict already set on the job so every
        # append is immediately visible — no explicit re-assignment needed.
        with _ts_lock:
            _history = _ts_jobs[job_id].setdefault(
                "history", {"points": [], "scores": [], "reagents": {}})
            _dbg("session",
                 f"stream open job={job_id} rxn={_ts_jobs.get(job_id, {}).get('rxn_key','?')} "
                 f"launch_idx={_ts_jobs.get(job_id, {}).get('launch_idx')} "
                 f"pid={_ts_jobs.get(job_id, {}).get('pid')}")

        # Build reagent-id → SMILES index from this reaction's building-block CSVs.
        # Used to attach a SMILES to each reagent entry (for 2D rendering in the UI).
        _smiles_index: dict = {}
        try:
            _job_meta   = _ts_jobs.get(job_id, {})
            _bb_files   = [f"{bb}.csv" for bb in _job_meta.get("bb_names", [])]
            if _mol is not None and _bb_files:
                _smiles_index = _mol.build_smiles_index(_BB_BASE, _bb_files)
                push(f"[TS] loaded {len(_smiles_index)} reagent SMILES from "
                     f"{len(_bb_files)} building-block file(s)")
        except Exception as _se:
            push(f"[TS] SMILES index build failed: {_se}")

        # Debug counters — visible in session file at any time
        _history["_dbg_lines_seen"]    = 0
        _history["_dbg_gate_opened"]   = False
        _history["_dbg_gate_trigger"]  = None
        _history["_dbg_evaluate_seen"] = 0
        _history["_dbg_score_added"]   = 0
        _history["_dbg_first_stdout"]  = []   # first 10 stdout lines
        _history["_dbg_first_stderr"]  = []   # first 5 stderr lines
        _write_session(job_id, _ts_jobs.get(job_id, {}))
        # Score source — MUST match the worker's priority to avoid live/reload mismatch.
        # Worker (ts_worker.js) priority: 1. [evaluate] score=X  2. score: X | smiles:
        # After the logging-to-stdout fix, [evaluate] now appears in stdout, so we use it
        # as primary here too. The point is created on the [evaluate] line.
        # Bound to the module-level patterns (see RE_* near the top of this file).
        # They live at module scope so tests can exercise them against known
        # elion output without spawning a subprocess: these four regexes are the
        # ONLY thing that turns elion stdout into the reagent table behind
        # #tsTsBars, so a silent format drift empties that panel and nothing else.
        _re_score_line = RE_SCORE_LINE
        _re_score_alt  = RE_SCORE_ALT
        _re_post       = RE_POST
        _re_winner     = RE_WINNER
        # Buffer of the current iteration's winners (cleared when score is assigned)
        _winner_buf: list = []
        # Map reagent_id -> partner reagent_id (the two cycle winners react together)
        _partner_map: dict = {}
        _pending_hist_score = [None]
        _pending_eval_score = [None]  # stderr [evaluate] score for current iter
        _score_count    = [0]
        _rolling_buf:   list = []
        _rolling_sum    = [0.0]
        _rolling_sum_sq = [0.0]
        _BATCH_MAX      = 256  # matches worker default

        _history_in_ts = [False]  # True once TS inference phase starts

        _top5_dirty   = [0]    # count of updates since last top5 recompute
        _TOP5_EVERY    = 32     # recompute the ranking at most every N updates
        # …but not before the panel has anything in it. See _update_reagent.
        _TOP5_WARM_N   = 64     # while the reagent dict is this small, every update
        # How many reagents the ranking carries. The KEY stays "top5" and
        # #tsTsBars still shows 5 (the frontend slices) — this is headroom for
        # the RL tab's user-settable "top N" picker, which would otherwise be
        # capped at whatever this list happens to be. nlargest(20) costs the
        # same single O(n) pass as nlargest(5); only the payload grows, by ~15
        # small dicts per poll.
        _TOP_N         = 20

        # ── Parse-match trace (see _dbg) ──────────────────────────────────
        # `#tsTsBars` is fed from _history["top5"], which is fed from these
        # regexes. If elion's stdout format drifts, every counter below stays 0
        # while the chart and iteration counter keep working — because those come
        # from [TS:stats], which is emitted by THIS module, not by elion. That
        # asymmetry is exactly why an empty panel is hard to read from the UI.
        _dbg_name  = f"job_{job_id[:8]}"
        _m_counts  = {"post": 0, "winner": 0, "score": 0, "lines": 0,
                      "update_reagent": 0, "recompute": 0}
        _unmatched_samples: list = []

        def _update_reagent(name, mu, std, sc, best=None, partner=None):
            """Upsert a reagent into _history['reagents'] and keep the JSON's
            'top5' list (highest 'best' molecule score) maintained.
            best is monotonic — only raised, never lowered."""
            r = _history["reagents"].get(name, {})
            prev_best = r.get("best")
            if best is not None:
                new_best = best if prev_best is None else max(prev_best, best)
            else:
                new_best = prev_best
            r.update({
                "mu": round(mu, 6), "std": round(std, 6), "sc": sc,
                "best": (round(new_best, 6) if new_best is not None else None),
                "smiles": r.get("smiles") or _smiles_index.get(name, ""),
            })
            if partner:
                r["partner"] = partner
            _history["reagents"][name] = r
            # Throttle the ranking recompute. Sorting the full (unbounded) reagent
            # dict on EVERY line was O(n log n) per line with n growing to tens of
            # thousands — by ~1000 iters this dominated the parse loop and starved
            # the stdout pipe reader, freezing the elion subprocess at 0% CPU.
            # The top5 only needs to be eventually-consistent (it's polled every
            # few seconds by the UI), so recompute periodically instead.
            #
            # WARM START. A pure every-32nd throttle has a bad opening: until the
            # 32nd update `_history["top5"]` is [], `/ts_top5` returns nothing,
            # `_tsRenderTsBars` finds no bars and #tsTsBars stays empty. On a short
            # run (num_ts_iterations in the single digits) the whole job can finish
            # in fewer than 32 updates, so the ONLY recompute that ever lands is the
            # final one after the stream closes — which is why the reagent panel
            # looked like it "only appears once Run TS is done".
            #
            # The cost the throttle exists to avoid is a function of dict SIZE, not
            # of update count: the comment above is about n in the tens of thousands.
            # While n is small, nlargest(5, n) is free. So recompute on every update
            # until the dict outgrows _TOP5_WARM_N, then fall back to the throttle.
            # Panel populates on the first poll; the marathon-run protection is
            # untouched because it only ever mattered once n was large.
            _m_counts["update_reagent"] += 1
            _top5_dirty[0] += 1
            if len(_history["reagents"]) <= _TOP5_WARM_N or _top5_dirty[0] >= _TOP5_EVERY:
                _top5_dirty[0] = 0
                _recompute_top5()
                _m_counts["recompute"] += 1
                if _m_counts["recompute"] in (1, 2, 3):
                    _dbg(_dbg_name,
                         f"recompute #{_m_counts['recompute']}: "
                         f"reagents={len(_history['reagents'])} "
                         f"top5={len(_history.get('top5') or [])} "
                         f"first={(_history.get('top5') or [{}])[0].get('name')}")

        def _recompute_top5():
            """Timed wrapper — see the [DEBUG:loop] top5_ms/call counter.

            This walks _history['reagents'] in FULL on every call, and a warmup
            checkpoint restore seeds that dict with tens of thousands of
            entries. Whether that matters is a measurement, not an assumption:
            top5_ms/call x top5_calls per iteration is the answer.
            """
            import time as _t_mod          # local: _time_ws is imported further down
            _t0 = _t_mod.perf_counter()
            try:
                return _recompute_top5_inner()
            finally:
                _history["_dbg_top5_s"] = (_history.get("_dbg_top5_s", 0.0)
                                           + _t_mod.perf_counter() - _t0)
                _history["_dbg_top5_calls"] = _history.get("_dbg_top5_calls", 0) + 1

        def _recompute_top5_inner():
            """Maintain _history['top5']: the _TOP_N reagents with the highest
            'best' molecule score (monotonic ranking, stable across restarts).
            Reagents with no best score yet are ranked below scored ones by μ.
            Uses heapq.nlargest (O(n) single pass) rather than a full sort.

            The key is still called "top5" for wire compatibility — every reader
            (/ts_top5, _tsTop5ToBars, the session files on disk) keys off that
            name, and the list has always been "the ranking", not "exactly five".
            #tsTsBars slices to 5 client-side so its appearance is unchanged."""
            import heapq
            items = _history["reagents"].items()
            # Partition without building two full intermediate lists where avoidable.
            scored   = [(r["best"], n, r) for n, r in items if r.get("best") is not None]
            top_scored = heapq.nlargest(_TOP_N, scored, key=lambda t: t[0])
            top = [(n, r) for _b, n, r in top_scored]
            if len(top) < _TOP_N:
                # Backfill with the highest-μ unscored reagents.
                need = _TOP_N - len(top)
                unscored = ((n, r) for n, r in items if r.get("best") is None)
                top_uns = heapq.nlargest(need, unscored, key=lambda kv: kv[1].get("mu", 0))
                top.extend(top_uns)
            _history["top5"] = [
                {"id": n, "name": n, "mu": r.get("mu"), "std": r.get("std"),
                 "sc": r.get("sc"), "best": r.get("best"),
                 "smiles": r.get("smiles", ""), "partner": r.get("partner", "")}
                for n, r in top
            ]

        def _history_parse(ln):
            # Count what each line matched BEFORE the parse logic runs, so the
            # trace reflects the regexes rather than the branches they feed.
            _m_counts["lines"] += 1
            _hit = False
            if _re_post.search(ln):
                _m_counts["post"] += 1
                _hit = True
            if _re_winner.search(ln):
                _m_counts["winner"] += 1
                _hit = True
            if _re_score_line.search(ln):
                _m_counts["score"] += 1
                _hit = True
            # Keep a few genuinely unmatched lines that *look* like reagent
            # output — a regex drift shows up here as recognisable text.
            if (not _hit and len(_unmatched_samples) < 25
                    and any(k in ln for k in ("post-update", "winner", "reagent",
                                              "μ=", "mu=", "score"))):
                _unmatched_samples.append(ln[:200])
                _dbg(_dbg_name, f"UNMATCHED[{len(_unmatched_samples)}]: {ln[:200]!r}")
            if _m_counts["lines"] % 200 == 0:
                _dbg(_dbg_name,
                     f"lines={_m_counts['lines']} post={_m_counts['post']} "
                     f"winner={_m_counts['winner']} score={_m_counts['score']} "
                     f"update_reagent={_m_counts['update_reagent']} "
                     f"recompute={_m_counts['recompute']} "
                     f"reagents={len(_history.get('reagents') or {})} "
                     f"top5={len(_history.get('top5') or [])}")
            return _history_parse_impl(ln)

        def _history_parse_impl(ln):
            # Detect TS inference start.
            # Three signals that TS inference has begun:
            #   1. 'cycle_id' in line — explicit TS cycle marker
            #   2. _warmup_saved — warmup just completed (real warmup path)
            #   3. '[evaluate]' in line — score line appeared; warmup must be done
            #      (handles checkpoint-loaded runs where warmup is skipped silently
            #       and neither 'Top score found during warmup' nor cycle_id may appear
            #       before the first [evaluate] line)
            if not _history_in_ts[0]:
                # Gate opens on any signal that TS inference has started:
                # 1. cycle_id line (TS cycle marker)
                # 2. _warmup_saved (real warmup just completed)
                # 3. score: X | smiles: (first TS score line in stdout)
                # 4. winner | cycle_id (TS winner line)
                sm_check = _re_score_line.search(ln) or _re_score_alt.search(ln)
                is_ts_signal = ('cycle_id' in ln or _warmup_saved or
                                bool(sm_check) or 'winner | cycle_id' in ln)
                if is_ts_signal:
                    _history_in_ts[0] = True
                    _pending_eval_score[0] = None
                    _pending_hist_score[0] = None
                    push(f"[TS:history] gate opened by: {ln[:80]!r}")
                    logger.info("[TS:history] gate opened by: %r", ln[:80])
                    _history["_dbg_gate_opened"]  = True
                    _history["_dbg_gate_trigger"] = ln[:120]
                    # If triggering line is a score line, fall through to process it now
                    if not sm_check:
                        return  # non-score gate line; score on NEXT line
                else:
                    return

            # Parse winner lines — track reagent μ/σ and buffer this iteration's
            # winners so the molecule score (arriving on the next [evaluate]) can
            # be assigned to them as their 'best'.
            wm = _re_winner.search(ln)
            if wm:
                cycle = int(wm.group(1)); rname = wm.group(2)
                rmu = float(wm.group(4)); rstd = float(wm.group(5)); rsc = int(wm.group(6))
                # Update reagent belief from the winner line
                _update_reagent(rname, rmu, rstd, rsc)
                _winner_buf.append({"cycle": cycle, "name": rname})
                # When both cycle winners are present, link them as partners
                cycles = {w["cycle"] for w in _winner_buf}
                if 0 in cycles and 1 in cycles:
                    w0 = next(w for w in _winner_buf if w["cycle"] == 0)
                    w1 = next(w for w in _winner_buf if w["cycle"] == 1)
                    _partner_map[w0["name"]] = w1["name"]
                    _partner_map[w1["name"]] = w0["name"]
                    if w0["name"] in _history["reagents"]:
                        _history["reagents"][w0["name"]]["partner"] = w1["name"]
                    if w1["name"] in _history["reagents"]:
                        _history["reagents"][w1["name"]]["partner"] = w0["name"]
                    _recompute_top5()
                return

            # Use [evaluate] score=X exclusively to create history points.
            # - Appears in stdout at INFO level (always present with log_level=DEBUG forced by ts_routes)
            # - Matches worker's priority 1 score (pendingScore) — same value worker displays
            # - Exactly one per iteration, strictly ordered in stdout
            # - "score: X | smiles:" (DEBUG) is ignored — it's the same value but secondary
            sm = _re_score_line.search(ln)
            if sm:
                v = float(sm.group(1))
                _pending_hist_score[0] = v
                _rolling_buf.append(v)
                logger.debug("[TS:history] score point #%d v=%.4f from: %r",
                             _score_count[0]+1, v, ln[:60])
                _history["_dbg_score_added"] = _score_count[0] + 1
                _rolling_sum[0]    += v
                _rolling_sum_sq[0] += v * v
                while len(_rolling_buf) > _BATCH_MAX:
                    old = _rolling_buf.pop(0)
                    _rolling_sum[0]    -= old
                    _rolling_sum_sq[0] -= old * old
                n    = len(_rolling_buf)
                mean = _rolling_sum[0] / n
                vari = ((_rolling_sum_sq[0] - _rolling_sum[0]**2 / n) / (n - 1)) if n > 1 else 0.0
                std  = max(0.0, vari) ** 0.5
                _history["points"].append({
                    "mean":  round(mean, 6),
                    "std":   round(std, 6),
                    # Store raw score for phase 1 (iter < BATCH_MAX) for hover tooltip
                    "score": round(v, 6) if len(_rolling_buf) <= _BATCH_MAX else None,
                })
                # Bound the points array on marathon runs so _write_session's
                # json.dump stays cheap regardless of run length. Mirrors the
                # frontend's _TS_SPARK_FULL_MAX rule (ts_chart.js): once the buffer
                # exceeds 60000, keep every other point. The frontend renumbers
                # points by index and stride-decimates on replay, so this keeps the
                # chart visually identical before/after refresh while capping the
                # serialized size (and thus the write cost) at ~60k points.
                if len(_history["points"]) > 60000:
                    _history["points"] = _history["points"][::2]
                if "scores" not in _history:
                    _history["scores"] = []
                _history["scores"].append(round(v, 6))
                if len(_history["scores"]) > 256:
                    _history["scores"].pop(0)
                _score_count[0] += 1
                # Assign this molecule's score to the iteration's winning reagents
                # as their 'best' (monotonic). This is what makes the JSON top5
                # match the live UI, which ranks reagents by best molecule score.
                for w in _winner_buf:
                    rr = _history["reagents"].get(w["name"])
                    if rr is not None:
                        _update_reagent(w["name"], rr.get("mu", 0.0),
                                        rr.get("std", 0.0), rr.get("sc", 0),
                                        best=v, partner=_partner_map.get(w["name"]))
                _winner_buf.clear()
                # Emit the authoritative per-iteration stats over SSE. The worker
                # uses THIS instead of re-parsing scores itself — guarantees the
                # live chart matches what gets persisted and restored on reload.
                # Format is machine-parsed by ts_worker.js (_re_ts_stats).
                push(f"[TS:stats] iter={_score_count[0]} "
                     f"mean={round(mean,6)} std={round(std,6)} score={round(v,6)}")
                # Write every iteration during debug (change % 10 after debugging)
                _write_session(job_id, _ts_jobs.get(job_id, {}))
                return
            pm = _re_post.search(ln)
            if pm:
                name, mu, std, sc = pm.group(1), float(pm.group(2)), float(pm.group(3)), int(pm.group(4))
                # post-update carries belief (μ/σ/n); the iteration score (best)
                # is associated separately via the winner buffer above.
                _update_reagent(name, mu, std, sc,
                                best=_pending_hist_score[0],
                                partner=_partner_map.get(name))

        # Session-write throttle: bound the cost of persisting the growing
        # history so the stdout consumer never falls behind elion (see the
        # _write_session call inside the loop for the full rationale).
        import time as _time_ws
        _SESSION_WRITE_INTERVAL_S = 3.0
        _last_session_write = [0.0]   # list = mutable closure cell

        while True:
            line = _stdout_queue.get()
            if line is None:        # sentinel: stdout closed, no more lines
                break
            if True:
                if True:
                    push(line)
                    # Only the first lines (warmup phase) are ever used
                    # (_save_warmup_from_log parses [warm_up] lines emitted once at
                    # startup). Cap this list so it doesn't grow unboundedly for the
                    # whole run — important now that _stdout_queue is unbounded.
                    if len(_all_log_lines) < 5000:
                        _all_log_lines.append(line)
                    _history["_dbg_lines_seen"] += 1
                    if "[evaluate]" in line:
                        _history["_dbg_evaluate_seen"] += 1
                        if _history["_dbg_evaluate_seen"] == 1:
                            push(f"[DEBUG:evaluate] FIRST [evaluate] line seen: {line!r}")
                    if "cycle_id" in line and _history["_dbg_lines_seen"] <= 5:
                        push(f"[DEBUG:cycle_id] seen: {line!r}")
                    # Store AND push first 10 stdout lines
                    if len(_history.get("_dbg_first_stdout", [])) < 10:
                        if "_dbg_first_stdout" not in _history:
                            _history["_dbg_first_stdout"] = []
                        _history["_dbg_first_stdout"].append(line[:120])
                        push(f"[DEBUG:stdout#{len(_history['_dbg_first_stdout'])}] {line!r}")
                    if _history["_dbg_lines_seen"] % 50 == 0:
                        # ── Reader-side cost accounting ──────────────────────
                        # If the engine reports a large write_ms, it is blocked
                        # on the pipe, which means THIS loop is the bottleneck.
                        # These numbers say which part of it:
                        #   parse_ms  — total time in _history_parse per line
                        #   top5_ms   — time inside _recompute_top5 alone
                        #   nreag     — size of _history['reagents'], which
                        #               _recompute_top5 walks in FULL on every
                        #               call. A checkpoint restore seeds this
                        #               with tens of thousands of entries, so
                        #               this is O(nreag) per parsed line and
                        #               grows as more reagents get scored —
                        #               which would show up as an iteration
                        #               time that climbs, exactly as tqdm
                        #               reported (3.16 -> 4.74 s/it).
                        _pt = _history.get("_dbg_parse_s", 0.0)
                        _tt = _history.get("_dbg_top5_s", 0.0)
                        _tc = _history.get("_dbg_top5_calls", 0) or 1
                        _ln = _history["_dbg_lines_seen"] or 1
                        push(f"[DEBUG:loop] {_history['_dbg_lines_seen']} stdout lines | "
                             f"evaluate_seen={_history['_dbg_evaluate_seen']} | "
                             f"gate={_history['_dbg_gate_opened']} | "
                             f"scores={_history['_dbg_score_added']} | "
                             f"qlen={_stdout_queue.qsize()} | "
                             f"parse_ms/line={_pt / _ln * 1000:.3f} | "
                             f"top5_ms/call={_tt / _tc * 1000:.3f} | "
                             f"top5_calls={_history.get('_dbg_top5_calls', 0)} | "
                             f"nreag={len(_history.get('reagents', {}))} | "
                             f"last_line={line[:60]!r}")
                    _p_t0 = _time_ws.perf_counter()
                    _history_parse(line)
                    _history["_dbg_parse_s"] = (_history.get("_dbg_parse_s", 0.0)
                                                + _time_ws.perf_counter() - _p_t0)
                    # Persist the session on a TIME throttle, not per-N-lines.
                    # _write_session json-dumps the full history (points+scores,
                    # which grow one-per-molecule). Doing that every 10 lines makes
                    # the per-line cost grow O(n) with the run, so eventually this
                    # consumer falls behind elion's stdout -> the stdout queue fills
                    # -> the OS pipe backs up -> elion BLOCKS on write() (seen as
                    # erratic per-phase timing spikes ~iter 4000-6000). Writing at
                    # most once every few seconds bounds the cost regardless of
                    # how long the run gets.
                    _now_ws = _time_ws.monotonic()
                    if _now_ws - _last_session_write[0] >= _SESSION_WRITE_INTERVAL_S:
                        _write_session(job_id, _ts_jobs.get(job_id, {}))
                        _last_session_write[0] = _now_ws

                    # Also drain any pending stderr lines into the warmup window
                    # (per-reagent DEBUG lines go to stderr via Python logging)
                    while not _warmup_saved:
                        try:
                            err_line = _stderr_queue.get_nowait()
                            if not _prior_seen and '[warm_up]' in err_line and 'prior_mean=' in err_line:
                                _prior_seen = True
                            _warmup_window.append(err_line)
                            if not _prior_seen and len(_warmup_window) > 500:
                                _warmup_window.pop(0)
                        except _errq.Empty:
                            break

                    # Rolling window: once we've seen prior_mean, keep everything
                    # (need all the per-reagent component lines that follow).
                    # Before that, limit to last 500 lines to cap memory.
                    if not _prior_seen and '[warm_up]' in line and 'prior_mean=' in line:
                        _prior_seen = True
                    _warmup_window.append(line)
                    if not _prior_seen and len(_warmup_window) > 500:
                        _warmup_window.pop(0)

                    # Save AFTER all per-reagent lines are written.
                    # Trigger: "Top score found during warmup" — appears after the
                    # per-reagent DEBUG lines in thompson_sampling.py warm_up().
                    if not _warmup_saved and not _warmup_warned and 'Top score found during warmup' in line:
                        rxn_key_for_job = rxn_key_val   # spawn-captured; reaper-immune
                        _n_wu    = sum(1 for _l in _warmup_window if '[warm_up]' in _l)
                        _n_prior = sum(1 for _l in _warmup_window if '[warm_up]' in _l and 'prior_mean=' in _l)
                        push(f"[DEBUG:warmup] trigger seen (stdout) rxn_key={rxn_key_for_job!r} "
                             f"window_lines={len(_warmup_window)} warm_up_lines={_n_wu} "
                             f"prior_mean_lines={_n_prior} → attempting checkpoint save")
                        if rxn_key_for_job:
                            _saved_path = _save_warmup_from_log(rxn_key_for_job, _warmup_window, _job_timestamp)
                            if _saved_path:
                                _warmup_saved = True
                                push(f"[TS] warmup checkpoint written → {_saved_path}")
                                _warmup_window = []  # free memory; no longer needed
                            else:
                                push(f"[DEBUG:warmup] mid-stream save found no prior in window yet "
                                     f"(window_lines={len(_warmup_window)}) — deferring to end-of-run fallback")

        if False:  # buf no longer used — stdout is line-split in the drain thread
            pass

        # Final ranking recompute to capture the last <N throttled updates.
        _recompute_top5()
        _dbg(_dbg_name,
             f"STREAM CLOSED | lines={_m_counts['lines']} post={_m_counts['post']} "
             f"winner={_m_counts['winner']} score={_m_counts['score']} "
             f"update_reagent={_m_counts['update_reagent']} "
             f"recompute={_m_counts['recompute']} "
             f"reagents={len(_history.get('reagents') or {})} "
             f"top5={len(_history.get('top5') or [])}")
        if _m_counts["post"] == 0 and _m_counts["winner"] == 0:
            _dbg(_dbg_name,
                 "DIAGNOSIS: no post-update and no winner line ever matched. The "
                 "reagent table is built ONLY from those two regexes, so top5 stays "
                 "empty and #tsTsBars renders nothing — while the chart and the "
                 "iteration counter keep working, because they come from [TS:stats] "
                 "which this module emits itself. Compare the UNMATCHED samples above "
                 "against _re_post / _re_winner in ts_routes.py.")

        # Drain any remaining stderr into warmup window
        import time as _time
        _time.sleep(0.2)  # let stderr thread finish
        while True:
            try:
                err_line = _stderr_queue.get_nowait()
                if not _prior_seen and '[warm_up]' in err_line and 'prior_mean=' in err_line:
                    _prior_seen = True
                _warmup_window.append(err_line)
                # Check trigger in stderr too
                if not _warmup_saved and not _warmup_warned and 'Top score found during warmup' in err_line:
                    rxn_key_for_job = rxn_key_val   # spawn-captured; reaper-immune
                    if rxn_key_for_job:
                        _saved_path = _save_warmup_from_log(rxn_key_for_job, _warmup_window, _job_timestamp)
                        if _saved_path:
                            _warmup_saved = True
                            push(f"[TS] warmup checkpoint written → {_saved_path}")
                        else:
                            push(f"[DEBUG:warmup] mid-stream save (stderr) found no prior in window yet "
                                 f"(window_lines={len(_warmup_window)}) — deferring to end-of-run fallback")
            except _errq.Empty:
                break

        proc.wait()
        rc = proc.returncode
        with _ts_lock:
            _ts_jobs[job_id]["returncode"] = rc
            _ts_jobs[job_id]["status"] = "done" if rc == 0 else "error"

        # If the process died abnormally, surface WHY (exit code + stderr tail)
        # so it appears in the UI error panel instead of silently vanishing.
        # rc < 0 means killed by signal -rc (e.g. -9 OOM-kill, -15 SIGTERM).
        if rc not in (0, None):
            _sig = f" (killed by signal {-rc})" if rc < 0 else ""
            _tail = _stderr_lines[-30:] if _stderr_lines else []
            push(f"[STDERR] ERROR: elion.py exited with code {rc}{_sig}")
            if _tail:
                push(f"[STDERR] ── full stderr tail ({len(_tail)} lines) ──")
                for _tl in _tail:
                    # Prefix every line with [STDERR] so the bridge forwards it
                    push(f"[STDERR] {_tl}")
            else:
                push(f"[STDERR] (no stderr captured — likely OOM-kill or hard crash)")
            logger.error("[TS] job %s (%s) exited code %s%s; stderr tail: %s",
                         job_id, rxn_key_val or "?",
                         rc, _sig, _tail)

        # Final history flush to disk, then remove session (job complete)
        _write_session(job_id, _ts_jobs.get(job_id, {}))
        _delete_session(job_id)

        # ── DEBUG: end-of-run warmup summary ────────────────────────────────
        # Whether this run LOADED an existing checkpoint (warmup skipped) or ran a
        # FRESH warmup decides what to expect. When a checkpoint is loaded there is
        # NO new warmup to parse or save, so the "no prior line / no checkpoint"
        # warnings do NOT apply — reporting them was a false alarm.
        _ckpt_loaded = bool(ckpt_path and os.path.isfile(ckpt_path))
        _combined    = _all_log_lines + _stderr_lines
        _n_warm_out  = sum(1 for _l in _all_log_lines if '[warm_up]' in _l)
        _n_warm_err  = sum(1 for _l in _stderr_lines  if '[warm_up]' in _l)
        _n_prior     = sum(1 for _l in _combined if '[warm_up]' in _l and 'prior_mean=' in _l)
        _n_eval      = sum(1 for _l in _combined if '[evaluate]' in _l)
        push(f"[DEBUG:warmup] end-of-run summary: rxn_key={rxn_key_val!r} rc={rc} "
             f"ckpt_loaded={_ckpt_loaded} "
             f"stdout_lines={len(_all_log_lines)} stderr_lines={len(_stderr_lines)} "
             f"warm_up_lines(stdout={_n_warm_out},stderr={_n_warm_err}) "
             f"prior_mean_lines={_n_prior} evaluate_lines={_n_eval} warmup_saved={_warmup_saved}")

        if _ckpt_loaded:
            # Warmup was SKIPPED — an existing checkpoint was injected. Report it and
            # stop; there is nothing new to parse or save (no warning, no dump).
            try:
                import json as _json_dbg
                with open(ckpt_path) as _cf:
                    _nreag = _json_dbg.load(_cf).get('n_reagents', '?')
            except Exception:
                _nreag = '?'
            push(f"[DEBUG:warmup] ✓ warmup SKIPPED — loaded existing checkpoint "
                 f"{ckpt_path} ({_nreag} per-reagent beliefs). No new checkpoint written this run.")
        else:
            # FRESH run — dump the warmup/score region (only in 'debug' verbosity,
            # via the [DEBUG:wdump] prefix) then try to save a new checkpoint.
            _tagged = ([('o', _l) for _l in _all_log_lines] +
                       [('e', _l) for _l in _stderr_lines])
            _probe  = [(s, _l) for (s, _l) in _tagged
                       if re.search(r'warm|prior|mu=|std=|TS-draw|eval score|cycle=|'
                                    r'no product|no reaction|invalid', _l)]
            if _probe:
                push(f"[DEBUG:wdump] ── {len(_probe)} warmup/score lines (o=stdout e=stderr, ≤40) ──")
                for _s, _pl in _probe[:40]:
                    push(f"[DEBUG:wdump]   {_s}| {_pl[:200]}")

            # Save a new checkpoint from the FULL combined stream (uncapped: the
            # '[warm_up] prior_mean=…' line can land tens of thousands of lines deep).
            # Use rxn_key_val (spawn-captured) — NOT a late _ts_jobs read, which the
            # background reaper can delete once the process exits.
            rxn_key_for_job = rxn_key_val
            if not _warmup_saved and rxn_key_for_job and _combined:
                _saved_path = _save_warmup_from_log(rxn_key_for_job, _combined, _job_timestamp)
                if _saved_path:
                    _warmup_saved = True
                    push(f"[TS] warmup checkpoint written (end-of-run) → {_saved_path}")
                elif not _warmup_warned:
                    _warmup_warned = True
                    push(f"[TS] ⚠ warmup produced NO parseable prior — no checkpoint saved for "
                         f"'{rxn_key_for_job}'. No '[warm_up] prior_mean' line in stdout or stderr; "
                         f"see the [DEBUG:wdump] lines (set verbosity to 'debug') for the log format.")

            # Report where the new checkpoint landed.
            _ckpt_final = _warmup_cache_path(rxn_key_for_job, _job_timestamp) if rxn_key_for_job else None
            if _ckpt_final and os.path.isfile(_ckpt_final):
                try:
                    _csz = os.path.getsize(_ckpt_final)
                    import json as _json_dbg2
                    with open(_ckpt_final) as _cf2:
                        _ncomp = _json_dbg2.load(_cf2).get('n_reagents', 0)
                except Exception:
                    _csz, _ncomp = -1, '?'
                push(f"[DEBUG:warmup] ✓ checkpoint JSON saved → {_ckpt_final} "
                     f"({_csz} bytes, {_ncomp} per-reagent beliefs)")
                if _ncomp == 0:
                    push(f"[DEBUG:warmup] ⚠ saved with GLOBAL PRIOR ONLY (0 per-reagent beliefs) — "
                         f"per-reagent warmup lines use a format the component parser doesn't read.")
            elif rxn_key_for_job:
                push(f"[DEBUG:warmup] ✗ NO checkpoint JSON on disk. Expected → {_ckpt_final} "
                     f"(Warmup_TS dir: {_WARMUP_DIR})")

        _results_saved_path = results_path   # tracked locally (reaper-safe) for [DEBUG:tsproc]
        if rc == 0 and results_path and os.path.isfile(results_path):
            # Compute mean and std of the score column, then rename
            try:
                import csv as _csv, statistics as _stats
                scores = []
                with open(results_path, newline='') as f:
                    reader = _csv.DictReader(f)
                    # Try common score column names
                    score_col = None
                    for candidate in ('score', 'total_score', 'reward', 'Score', 'Total_Score'):
                        if candidate in (reader.fieldnames or []):
                            score_col = candidate
                            break
                    if score_col:
                        for row in reader:
                            try:
                                scores.append(float(row[score_col]))
                            except (ValueError, KeyError):
                                pass

                if scores:
                    mean_s = _stats.mean(scores)
                    std_s  = _stats.pstdev(scores)  # population std
                    # Build new name: strip _tmp, insert mean/std
                    base = os.path.basename(results_path)            # e.g. amide_20250611_123456_tmp.csv
                    stem = base.replace('_tmp.csv', '')              # amide_20250611_123456
                    new_name = f"{stem}_mean{mean_s:.3f}_std{std_s:.3f}.csv"
                    new_path = os.path.join(os.path.dirname(results_path), new_name)
                    os.rename(results_path, new_path)
                    _results_saved_path = new_path
                    with _ts_lock:
                        _ts_jobs[job_id]["results_path"] = new_path
                    push(f"[TS] results saved → {new_name}")
                    logger.info("[TS] job %s results renamed to %s", job_id, new_name)
                else:
                    logger.warning("[TS] job %s: could not find score column in %s", job_id, results_path)
            except Exception as rename_exc:
                logger.warning("[TS] job %s: rename failed: %s", job_id, rename_exc)

        # ── DEBUG: end-of-TS-process summary (always mirrored to the 💬 panel;
        # in 'debug' verbosity these + the [DEBUG:tail] block are the ONLY lines
        # shown). Answers "where is the results CSV?" and pinpoints where a run
        # stopped when it ends mid-way (e.g. the iter-1668 error).
        push(f"[DEBUG:tsproc] TS process ended: rc={rc} rxn_key={rxn_key_val!r}")
        if _results_saved_path and os.path.isfile(_results_saved_path):
            try:
                _rsz = os.path.getsize(_results_saved_path)
                with open(_results_saved_path, newline='') as _rf:
                    _rrows = sum(1 for _ in _rf)
            except Exception:
                _rsz, _rrows = -1, -1
            push(f"[DEBUG:tsproc] ✓ results CSV saved → {_results_saved_path} "
                 f"({_rsz} bytes, {_rrows} rows incl. header)")
        else:
            push(f"[DEBUG:tsproc] ✗ results CSV NOT on disk — expected {_results_saved_path} "
                 f"(tmp path was {results_path})")
        # Progress: number of scored molecules ([evaluate] lines) + elion's outer
        # tqdm "Cycle: N/M". (This build doesn't emit the old 'iter N' format.)
        _n_eval_all = (sum(1 for _l in _all_log_lines if '[evaluate]' in _l) +
                       sum(1 for _l in _stderr_lines  if '[evaluate]' in _l))
        _iter_nums  = [int(_m.group(1)) for _l in _all_log_lines
                       for _m in [re.search(r'\biter\s+(\d+)\b', _l)] if _m]
        _cycle_str  = 'n/a'
        for _l in reversed(_stderr_lines):
            _cm = re.search(r'Cycle:\s*\d+%[^\d]*(\d+)/(\d+)', _l)
            if _cm:
                _cycle_str = f"{_cm.group(1)}/{_cm.group(2)}"
                break
        push(f"[DEBUG:tsproc] progress: evaluations={_n_eval_all} "
             f"last_iter={max(_iter_nums) if _iter_nums else 'n/a'} outer_cycle={_cycle_str}")

        # ── DEBUG (verbose — shown only in 'debug' verbosity): why a mid-run
        # crash ended the run. Surfaces stderr tracebacks/errors + the final tail.
        _errs = [l for l in _stderr_lines
                 if ('Traceback' in l or 'Error' in l or 'ERROR' in l or 'Exception' in l)]
        if _errs:
            push(f"[DEBUG:tail] ── {len(_errs)} crash/exception line(s) in stderr (showing ≤25) ──")
            for _el in _errs[:25]:
                push(f"[DEBUG:tail]   ! {_el[:220]}")
        else:
            push(f"[DEBUG:tail] stderr clean — no crash/exception lines detected")
        if _stderr_lines:
            _pt = _stderr_lines[-30:]
            push(f"[DEBUG:tail] ── last {len(_pt)} stderr lines (tail) ──")
            for _tl in _pt:
                push(f"[DEBUG:tail]   e| {_tl[:220]}")

        if rc != 0:
            push(f"ERROR: elion.py exited with code {rc}")
        push("__DONE__")

    except Exception as exc:
        logger.exception("[TS] job %s failed", job_id)
        with _ts_lock:
            _ts_jobs[job_id]["status"] = "error"
        push(f"ERROR: {exc}")
        push("__DONE__")


@app.route('/vina_visualization/ts_mol_svg/<rxn_key>/<reagent_id>', methods=['GET'])
def ts_mol_svg(rxn_key: str, reagent_id: str):
    """Render a reagent's SMILES to a 2D SVG via RDKit, cached on disk under
    TS_Session/images/<rxn_key>/<reagent_id>.svg.
    Looks up the SMILES from the reaction's building-block CSVs.
    Optional query params: w, h (pixel dimensions).
    Returns 204 if rdkit unavailable or the id/SMILES can't be resolved.
    """
    logger.info("[TS:mol] request rxn_key=%s id=%s", rxn_key, reagent_id)
    if _mol is None:
        logger.warning("[TS:mol] rdkit/_mol unavailable — is rdkit installed and "
                       "ts_mol_render.py importable?")
        return ("", 204)
    try:
        w = max(60, min(400, int(request.args.get("w", 220))))
        h = max(40, min(200, int(request.args.get("h", 90))))
    except (TypeError, ValueError):
        w, h = 220, 90

    # Disk cache path: TS_Session/images/<rxn_key>/<reagent_id>_<w>x<h>.svg
    img_dir  = os.path.join(_STATE_DIR, "images", rxn_key)
    # Create the folder up-front so its existence confirms requests are arriving.
    try:
        os.makedirs(img_dir, exist_ok=True)
    except Exception as _me:
        logger.warning("[TS:mol] could not create %s: %s", img_dir, _me)
    img_path = os.path.join(img_dir, f"{reagent_id}_{w}x{h}.svg")
    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        return Response(open(img_path).read(), mimetype="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})

    # Resolve reagent_files for this reaction (accept rxn_key or short_name)
    reagent_files = _REACTION_CATALOGUE.get(rxn_key, {}).get("reagent_files", [])
    if not reagent_files:
        for k, v in _REACTION_CATALOGUE.items():
            if v.get("short_name") == rxn_key:
                reagent_files = v.get("reagent_files", [])
                break

    if not reagent_files:
        logger.warning("[TS:mol] no reagent_files for rxn_key=%s (catalogue keys: %s)",
                       rxn_key, list(_REACTION_CATALOGUE.keys()))
        return ("", 204)

    index  = _mol.build_smiles_index(_BB_BASE, reagent_files)
    smiles = index.get(reagent_id, "")
    if not smiles:
        logger.warning("[TS:mol] id=%s NOT in %d reagents for %s (sample ids: %s)",
                       reagent_id, len(index), rxn_key, list(index.keys())[:3])
        return ("", 204)

    # Render and cache to disk
    ok = _mol.render_to_disk(smiles, img_path, w, h)
    logger.info("[TS:mol] rendered id=%s smiles=%s -> %s (ok=%s)",
                reagent_id, smiles[:40], img_path, ok)
    svg = _mol.smiles_to_svg(smiles, w, h)
    if not svg:
        return ("", 204)
    return Response(svg, mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.route('/vina_visualization/ts_mol_smiles/<rxn_key>/<reagent_id>', methods=['GET'])
def ts_mol_smiles(rxn_key: str, reagent_id: str):
    """Return the SMILES for a reagent id PLUS the source building-block CSV it
    came from — file name, full path, and slot index — by checking each
    reagent_file individually (per-file, so we know which slot owns the id).
    JSON: {smiles, csv_file, csv_path, slot}.
    """
    if _mol is None:
        return jsonify({"smiles": "", "csv_file": "", "csv_path": "", "slot": -1})
    entry, reagent_files = _resolve_rxn(rxn_key)
    for slot, fname in enumerate(reagent_files or []):
        path = fname if os.path.isabs(fname) else os.path.join(_BB_BASE, fname)
        rmap = _mol.load_reagent_map(path)      # cached per path
        if reagent_id in rmap:
            return jsonify({"smiles":   rmap[reagent_id],
                            "csv_file": os.path.basename(path),
                            "csv_path": path,
                            "slot":     slot})
    return jsonify({"smiles": "", "csv_file": "", "csv_path": "", "slot": -1})


def _resolve_rxn(rxn_key: str):
    """Return (catalogue_entry, reagent_files) for a key OR short_name."""
    entry = _REACTION_CATALOGUE.get(rxn_key)
    if entry:
        return entry, entry.get("reagent_files", [])
    for k, v in _REACTION_CATALOGUE.items():
        if v.get("short_name") == rxn_key:
            return v, v.get("reagent_files", [])
    return None, []


@app.route('/vina_visualization/ts_product_smiles/<rxn_key>/<id_a>/<id_b>', methods=['GET'])
def ts_product_smiles(rxn_key: str, id_a: str, id_b: str):
    """Return the reaction PRODUCT SMILES for two building blocks under this
    reaction's SMARTS (JSON: {"smiles": "..."}). Empty string when the pair
    does not react cleanly (no/ambiguous product), so the UI shows a fallback.
    """
    if _mol is None:
        return jsonify({"smiles": ""})
    entry, reagent_files = _resolve_rxn(rxn_key)
    if not entry or not reagent_files:
        return jsonify({"smiles": ""})
    smarts = entry.get("smarts", "")
    index  = _mol.build_smiles_index(_BB_BASE, reagent_files)
    smi_a  = index.get(id_a, "")
    smi_b  = index.get(id_b, "")
    if not smarts or not smi_a or not smi_b:
        return jsonify({"smiles": ""})
    product = _mol.react_product_smiles(smarts, smi_a, smi_b)
    return jsonify({"smiles": product})


@app.route('/vina_visualization/ts_product_svg/<rxn_key>/<id_a>/<id_b>', methods=['GET'])
def ts_product_svg(rxn_key: str, id_a: str, id_b: str):
    """Render the reaction PRODUCT of two building blocks to a 2D SVG, cached on
    disk under TS_Session/images/<rxn_key>/prod_<id_a>_<id_b>_<w>x<h>.svg.
    Returns 204 when rdkit is unavailable or the pair has no clean product.
    Optional query params: w, h.
    """
    if _mol is None:
        return ("", 204)
    try:
        w = max(60, min(400, int(request.args.get("w", 220))))
        h = max(40, min(200, int(request.args.get("h", 90))))
    except (TypeError, ValueError):
        w, h = 220, 90

    img_dir = os.path.join(_STATE_DIR, "images", rxn_key)
    try:
        os.makedirs(img_dir, exist_ok=True)
    except Exception:
        pass
    img_path = os.path.join(img_dir, f"prod_{id_a}_{id_b}_{w}x{h}.svg")
    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        return Response(open(img_path).read(), mimetype="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})

    entry, reagent_files = _resolve_rxn(rxn_key)
    if not entry or not reagent_files:
        return ("", 204)
    smarts = entry.get("smarts", "")
    index  = _mol.build_smiles_index(_BB_BASE, reagent_files)
    smi_a  = index.get(id_a, "")
    smi_b  = index.get(id_b, "")
    if not smarts or not smi_a or not smi_b:
        return ("", 204)

    product = _mol.react_product_smiles(smarts, smi_a, smi_b)
    if not product:
        return ("", 204)

    _mol.render_to_disk(product, img_path, w, h)
    svg = _mol.smiles_to_svg(product, w, h)
    if not svg:
        return ("", 204)
    return Response(svg, mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.route('/vina_visualization/ts_pool_counts/<rxn_key>', methods=['GET'])
def ts_pool_counts(rxn_key: str):
    """Return the STATIC raw building-block count per reagent slot for a
    reaction — i.e. how many rows each slot's CSV file has, in catalogue order.

    JSON: {"counts": [n_slot0, n_slot1], "files": ["rxn110_1.csv", "rxn110_2.csv"]}

    This is the fixed pool size from the building-block files and does NOT
    change during a run (unlike the live "competitors" eligible count, which
    shrinks as the DisallowTracker masks reagents). The sidebar shows this so
    the displayed pool size is stable across iterations.
    """
    if _mol is None:
        return jsonify({"counts": [], "files": []})
    entry, reagent_files = _resolve_rxn(rxn_key)
    if not entry or not reagent_files:
        return jsonify({"counts": [], "files": []})
    counts = []
    for fname in reagent_files:
        path = fname if os.path.isabs(fname) else os.path.join(_BB_BASE, fname)
        # load_reagent_map is cached per path, so this is cheap on repeat calls.
        counts.append(len(_mol.load_reagent_map(path)))
    return jsonify({"counts": counts, "files": reagent_files})


def _find_model_file_in_yml(cfg: dict):
    """Walk the elion YAML config to find the CHEMBERT model_file path.
    Returns (model_file, prop_name) or (None, None).

    Structure-agnostic: recursively searches the whole config for a value that
    looks like a model checkpoint (.pt/.pth/.ckpt), preferring keys named
    model_file / model_state / model / trained_model / checkpoint, and falling
    back to ANY string value that ends in a checkpoint extension.
    """
    _MODEL_KEYS = ("model_file", "model_state", "trained_model", "checkpoint",
                   "model_path", "model")
    _EXTS = (".pt", ".pth", ".ckpt")

    def _looks_like_ckpt(val):
        return isinstance(val, str) and val.strip().lower().endswith(_EXTS)

    # First pass: prefer an explicit model-ish key whose value is a checkpoint.
    def _search_preferred(node, breadcrumb):
        if isinstance(node, dict):
            for k, val in node.items():
                if k in _MODEL_KEYS and _looks_like_ckpt(val):
                    # breadcrumb's last dict-key is the most useful "property" label
                    return val, (breadcrumb[-1] if breadcrumb else k)
            for k, val in node.items():
                found = _search_preferred(val, breadcrumb + [str(k)])
                if found[0]:
                    return found
        elif isinstance(node, list):
            for i, item in enumerate(node):
                found = _search_preferred(item, breadcrumb)
                if found[0]:
                    return found
        return None, None

    # Second pass: ANY string that looks like a checkpoint path.
    def _search_any(node, breadcrumb):
        if isinstance(node, dict):
            for k, val in node.items():
                if _looks_like_ckpt(val):
                    return val, (breadcrumb[-1] if breadcrumb else str(k))
                found = _search_any(val, breadcrumb + [str(k)])
                if found[0]:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _search_any(item, breadcrumb)
                if found[0]:
                    return found
        return None, None

    mf, prop = _search_preferred(cfg, [])
    if mf:
        return mf, prop
    return _search_any(cfg, [])


@app.route('/vina_visualization/ts_model_arch', methods=['GET'])
def ts_model_arch():
    """Inspect the .pt checkpoint configured for CHEMBERT and return an
    architecture summary for display in the UI. Loads ONLY the weights
    (torch.load), never instantiates the model, so it's cheap and safe.

    Optional ?path=… overrides the YAML-configured model file.
    JSON: {status, model_file, summary{...}, groups[...], verdict}
    """
    import yaml as _yaml
    model_file = request.args.get("path", "").strip()
    prop_name = None
    if not model_file:
        yml_path = os.path.join(_ELION_CWD, _ELION_YML)
        try:
            with open(yml_path) as f:
                cfg = _yaml.safe_load(f) or {}
            model_file, prop_name = _find_model_file_in_yml(cfg)
        except FileNotFoundError:
            return jsonify({"status": "error", "message": f"config not found: {yml_path}"}), 404
        except Exception as e:
            return jsonify({"status": "error", "message": f"could not read config: {e}"}), 500

    if not model_file:
        # Show what's actually in the YAML so we can locate the model path.
        _dbg = {}
        try:
            _dbg_yml = os.path.join(_ELION_CWD, _ELION_YML)
            with open(_dbg_yml) as f:
                _cfg = _yaml.safe_load(f) or {}
            _dbg = {
                "top_level_keys": list(_cfg.keys()) if isinstance(_cfg, dict) else [],
                "reward_function": _cfg.get("Reward_function") if isinstance(_cfg, dict) else None,
            }
        except Exception:
            pass
        return jsonify({
            "status": "error",
            "message": "No .pt/.pth/.ckpt path found anywhere in input_TS.yml. "
                       "The architecture viewer reads the model path from the config; "
                       "see 'debug' for the YAML structure, or pass ?path=/abs/model.pt.",
            "debug": _dbg,
        }), 404
    if not os.path.isfile(model_file):
        return jsonify({"status": "error", "message": f"model file not found: {model_file}"}), 404

    try:
        import torch as _torch
        _state = _torch.load(model_file, map_location="cpu")
        if isinstance(_state, dict) and "state_dict" in _state and \
                all(not _torch.is_tensor(v) for v in _state.values()):
            _state = _state["state_dict"]
        if not isinstance(_state, dict):
            return jsonify({"status": "error",
                            "message": "checkpoint is not a state_dict mapping"}), 400

        keys = list(_state.keys())
        has_bert_prefix = any(k.startswith("bert.") for k in keys)
        has_linear_head = any(k.startswith("linear.") or k == "linear.weight" for k in keys)
        n_params = 0
        n_tensors = 0
        for v in _state.values():
            if _torch.is_tensor(v):
                n_tensors += 1
                n_params += v.numel()

        # Count transformer encoder layers
        import re as _re
        layer_idxs = set()
        for k in keys:
            m = _re.search(r"transformer_encoder\.layers\.(\d+)\.", k)
            if m:
                layer_idxs.add(int(m.group(1)))
        n_layers = (max(layer_idxs) + 1) if layer_idxs else 0

        # Group keys into a compact tree: top-level module -> count + total params
        groups = {}
        for k, v in _state.items():
            top = k.split(".")[0]
            g = groups.setdefault(top, {"keys": 0, "params": 0, "sample": []})
            g["keys"] += 1
            if _torch.is_tensor(v):
                g["params"] += v.numel()
                if len(g["sample"]) < 3:
                    g["sample"].append({"key": k, "shape": list(v.shape)})
        groups_list = [
            {"name": name, "keys": info["keys"], "params": info["params"],
             "sample": info["sample"]}
            for name, info in sorted(groups.items(), key=lambda kv: -kv[1]["params"])
        ]

        # Verdict: is this a usable (fine-tuned) model or a bare backbone?
        if has_bert_prefix and has_linear_head:
            verdict = {
                "kind": "finetuned",
                "usable": True,
                "message": "Full BERT_base model (backbone + trained regression head). "
                           "Ready for binding-energy prediction.",
            }
        elif not has_bert_prefix and not has_linear_head:
            verdict = {
                "kind": "pretrained_backbone",
                "usable": False,
                "message": "Raw Smiles_BERT backbone only — no 'bert.' prefix and no "
                           "regression head (linear.*). It loads (keys are remapped, "
                           "head randomly initialized) but predictions are MEANINGLESS "
                           "until the head is trained. Use a fine-tuned model for real scores.",
            }
        else:
            verdict = {
                "kind": "partial",
                "usable": has_linear_head,
                "message": f"Checkpoint has bert_prefix={has_bert_prefix}, "
                           f"linear_head={has_linear_head}. Mixed/partial — verify before use.",
            }

        return jsonify({
            "status": "ok",
            "model_file": model_file,
            "property": prop_name,
            "summary": {
                "architecture": "BERT_base (Smiles_BERT backbone + nn.Linear(1024,1) head)",
                "tensors": n_tensors,
                "total_params": n_params,
                "transformer_layers": n_layers,
                "has_bert_prefix": has_bert_prefix,
                "has_linear_head": has_linear_head,
                "file_size_mb": round(os.path.getsize(model_file) / (1024 * 1024), 1),
            },
            "groups": groups_list,
            "verdict": verdict,
        })
    except Exception as e:
        import traceback as _tb
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}",
                        "trace": _tb.format_exc()[-800:]}), 500


@app.route('/vina_visualization/ts_active', methods=['GET'])
def ts_active():
    """Return all active TS jobs for page-reload recovery.
    Merges in-memory jobs (fresh history) with disk sessions (survives Flask restart).
    """
    seen = set()
    active = []

    # 1. In-memory jobs (most up-to-date history)
    with _ts_lock:
        for job_id, job in _ts_jobs.items():
            if job.get("status") == "running":
                seen.add(job_id)
                active.append({
                    "job_id":     job_id,
                    "key":        job.get("rxn_key", ""),
                    "short_name": job.get("short_name", job.get("rxn_key", "")),
                    "bb_names":   job.get("bb_names", []),
                    "cpu_cores":  job.get("cpu_cores", []),
                    "pid":        job.get("pid"),
                    "launch_idx": job.get("launch_idx", 0),
                    "history":    job.get("history", {"points": [], "scores": [], "reagents": {}}),
                })

    # 2. Disk sessions (survive Flask restart / network drop)
    disk_sessions = _load_sessions()
    debug_log = getattr(_load_sessions, "_last_debug", [])
    for s in disk_sessions:
        job_id = s.get("job_id")
        if not job_id or job_id in seen:
            continue  # already covered by in-memory
        seen.add(job_id)
        active.append({
            "job_id":     job_id,
            "key":        s.get("rxn_key", ""),
            "short_name": s.get("short_name", ""),
            "bb_names":   s.get("bb_names", []),
            "cpu_cores":  s.get("cpu_cores", []),
            "pid":        s.get("pid"),
            "launch_idx": s.get("launch_idx", 0),
            "history":    s.get("history", {"points": [], "scores": [], "reagents": {}}),
        })

    # Sort by launch_idx so tabs always rebuild in launch order, regardless of
    # whether jobs came from memory or disk and regardless of glob() ordering.
    active.sort(key=lambda j: j.get("launch_idx", 0))

    return jsonify({"jobs": active, "_debug": debug_log})


@app.route('/vina_visualization/ts_top5', methods=['GET'])
def ts_top5():
    """Lightweight endpoint returning ONLY each job's persisted top-5 reagents
    (highest best molecule score), for the frontend to poll periodically.
    Much smaller than /ts_active (no points/scores arrays). Reads in-memory jobs
    first (freshest), falling back to the session files on disk.

    Returns: { "jobs": [ { "job_id", "launch_idx", "rxn_key", "short_name",
                           "top5": [ {id,name,mu,std,sc,best,smiles,partner}, ... ] }, ... ] }
    """
    out = []
    seen = set()

    # In-memory jobs (freshest top5)
    with _ts_lock:
        for job_id, job in _ts_jobs.items():
            hist = job.get("history", {}) or {}
            out.append({
                "job_id":     job_id,
                "launch_idx": job.get("launch_idx", 0),
                "rxn_key":    job.get("rxn_key", ""),
                "short_name": job.get("short_name", ""),
                "status":     job.get("status", "running"),
                "top5":       hist.get("top5", []),
            })
            seen.add(job_id)

    # Disk sessions not already in memory
    try:
        for s in _load_sessions():
            jid = s.get("job_id")
            if not jid or jid in seen:
                continue
            seen.add(jid)
            hist = s.get("history", {}) or {}
            out.append({
                "job_id":     jid,
                "launch_idx": s.get("launch_idx", 0),
                "rxn_key":    s.get("rxn_key", ""),
                "short_name": s.get("short_name", ""),
                "status":     s.get("status", "running"),
                "top5":       hist.get("top5", []),
            })
    except Exception as e:
        logger.exception("[TS:top5] disk scan failed: %s", e)

    out.sort(key=lambda j: j.get("launch_idx", 0))
    # This is the server's last word before the browser takes over, so log
    # exactly what was handed across. If top5 is non-empty here and the panel is
    # still blank, the fault is in front of this line, not behind it.
    _dbg("top5", "GET /ts_top5 -> " + (
        "; ".join(f"{j['job_id'][:8]}[idx={j.get('launch_idx')},{j.get('status')}]"
                  f"top5={len(j.get('top5') or [])}" for j in out) or "NO JOBS"))
    # state_dir is returned so the frontend can say *why* the reagent panel is
    # empty. Without it, "no sessions on disk" and "the run produced nothing"
    # look identical in the browser — and the usual cause of the former is a
    # mis-resolved ELION_CWD making visualizer.output_dir unreadable.
    return jsonify({"jobs": out, "state_dir": _STATE_DIR, "debug_dir": _DEBUG_DIR})



@app.route('/vina_visualization/ts_run', methods=['POST'])
def vina_ts_run():
    """
    POST /vina_visualization/ts_run
    Body (all fields optional):
      {
        "yml_path":          str,    # path to base yml (default: input_TS.yml)
        "reaction_smarts":   str,    # legacy: single-reaction smarts override
        "reactions":         [       # multi-reaction parallel mode
          {"key": "rxn101_amide"},               # use catalogue entry
          {"key": "rxn110_suzuki",               # or fully custom
           "smarts": "...", "reagent_files": [...]}
        ],
        "num_ts_iterations": int
      }

    Launches one  python elion.py -i <yml>  per reaction in parallel daemon
    threads, each pinned to its own CPU slice.

    Returns: {"status":"started", "job_id": <first_job_id>,
              "jobs": [{"key":…, "job_id":…, "cpu_cores":…}, …]}
    (job_id at top level is kept for backwards compatibility with existing
     ts_status / ts_kill callers.)
    """
    try:
        import psutil
        total_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 4
    except ImportError:
        total_cores = os.cpu_count() or 4

    try:
        data    = request.get_json(force=True) or {}
        yml_src = data.get("yml_path", "").strip() or os.path.join(_ELION_CWD, _ELION_YML)
        iters   = data.get("num_ts_iterations")
        output_dir = data.get("output_dir", "").strip() or None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="elion_ts_")

        if not os.path.isfile(yml_src):
            return jsonify({"status": "error",
                            "message": f"yml not found: {yml_src}"}), 404

        # ── Build reaction list ───────────────────────────────────────────
        raw_reactions = data.get("reactions") or []

        # Legacy single-smarts mode
        if not raw_reactions and data.get("reaction_smarts", "").strip():
            raw_reactions = [{"smarts": data["reaction_smarts"].strip()}]

        # Default: run the yml as-is (no reaction override)
        if not raw_reactions:
            raw_reactions = [{}]

        reactions = []
        for i, r in enumerate(raw_reactions):
            key = r.get("key", f"rxn{i}")
            if key in _REACTION_CATALOGUE and "smarts" not in r:
                cat = _REACTION_CATALOGUE[key]
                smarts = cat["smarts"]
                reagent_files = cat["reagent_files"]
            else:
                smarts = r.get("smarts", "")
                reagent_files = r.get("reagent_files", [])
            reactions.append({"key": key, "smarts": smarts,
                               "reagent_files": reagent_files})

        # ── Distribute CPU cores evenly across reactions ───────────────────
        cores_per_rxn = max(1, total_cores // len(reactions))

        jobs = []
        for i, rxn in enumerate(reactions):
            cpu_slice = list(range(i * cores_per_rxn,
                                   min((i + 1) * cores_per_rxn, total_cores)))
            if not cpu_slice:
                cpu_slice = None

            # Patch yml only when we have something to override
            if rxn["smarts"] or rxn["reagent_files"]:
                yml_to_run, results_path = _patch_yml_for_reaction(
                    yml_src, tmp_dir, rxn["key"],
                    rxn["smarts"], rxn["reagent_files"], iters, i,
                    output_dir=output_dir)
            else:
                # No reaction override — just force log_level=DEBUG + iters
                results_path = None
                yml_to_run = os.path.join(tmp_dir, f"input_TS_{rxn['key']}.yml")
                shutil.copy2(yml_src, yml_to_run)
                content = open(yml_to_run).read()
                if re.search(r'log_level\s*:', content):
                    content = re.sub(r'(log_level\s*:\s*)\S+', r'\g<1>DEBUG', content)
                else:
                    content = content.rstrip() + '\n    log_level: DEBUG\n'
                if iters:
                    content = re.sub(r'(num_ts_iterations\s*:\s*)\d+',
                                     lambda m: m.group(1) + str(int(iters)), content)
                open(yml_to_run, 'w').write(content)

            job_id = str(uuid.uuid4())
            short_name = _REACTION_CATALOGUE.get(rxn["key"], {}).get("short_name", rxn["key"])
            bb_names   = [f.replace(".csv","") for f in (rxn.get("reagent_files") or
                          _REACTION_CATALOGUE.get(rxn["key"],{}).get("reagent_files",[]))]
            with _ts_lock:
                _ts_jobs[job_id] = {
                    "status":       "running",
                    "queue":        queue.Queue(),
                    "yml":          yml_to_run,
                    "pid":          None,
                    "returncode":   None,
                    "rxn_key":      rxn["key"],
                    "short_name":   short_name,
                    "bb_names":     bb_names,
                    "cpu_cores":    cpu_slice,
                    "results_path": results_path,
                    "launch_idx":   i,   # preserve launch order for reconnect tab mapping
                }

            t = threading.Thread(
                target=_run_ts_job,
                args=(job_id, yml_to_run, {"TS_RXN_KEY": rxn["key"]}, cpu_slice, results_path),
                daemon=True,
            )
            t.start()
            jobs.append({
                "key":        rxn["key"],
                "job_id":     job_id,
                "cpu_cores":  cpu_slice,
                "short_name": short_name,
                "bb_names":   bb_names,
            })
            logger.info("[TS] job %s started for reaction %s on cores %s",
                        job_id, rxn["key"], cpu_slice)

        return jsonify({
            "status":  "started",
            "job_id":  jobs[0]["job_id"],   # legacy compat
            "jobs":    jobs,
        })

    except Exception as exc:
        logger.exception("[TS] ts_run error")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/vina_visualization/ts_status/<job_id>', methods=['GET'])
def vina_ts_status(job_id: str):
    """
    GET /vina_visualization/ts_status/<job_id>
    Server-Sent Events stream.
    Each event:  data: <log line>\\n\\n
    Terminal:    data: __DONE__\\n\\n
    Keep-alive:  : keep-alive\\n\\n   (every 15 s while idle)
    """
    if job_id not in _ts_jobs:
        return jsonify({"status": "error", "message": "Unknown job_id"}), 404

    job = _ts_jobs[job_id]
    q   = job["queue"]

    def generate():
        # Replay stored debug lines immediately on connect so late-connecting
        # clients (reconnect after page open) can see startup diagnostics
        hist = job.get("history", {})
        for i, sl in enumerate(hist.get("_dbg_first_stdout", []), 1):
            yield f"data: [DEBUG:stdout#{i}] {sl!r}\n\n"
        for i, sl in enumerate(hist.get("_dbg_first_stderr", []), 1):
            if "[evaluate]" in sl:
                yield f"data: [STDERR:evaluate!] {sl!r}\n\n"
            else:
                yield f"data: [STDERR:#{i}] {sl!r}\n\n"
        lines_seen = hist.get("_dbg_lines_seen", 0)
        if lines_seen > 0:
            yield (f"data: [DEBUG:replay] at connect: lines_seen={lines_seen} "
                   f"evaluate_seen={hist.get('_dbg_evaluate_seen',0)} "
                   f"gate={hist.get('_dbg_gate_opened',False)} "
                   f"scores={hist.get('_dbg_score_added',0)}\n\n")
        while True:
            try:
                line = q.get(timeout=15)
                # Escape SSE special chars — newlines inside data would break the frame
                safe = line.replace("\n", " ").replace("\r", "")
                yield f"data: {safe}\n\n"
                if line == "__DONE__":
                    break
            except queue.Empty:
                yield ": keep-alive\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route('/vina_visualization/ts_kill/<job_id>', methods=['POST'])
def vina_ts_kill(job_id: str):
    """
    POST /vina_visualization/ts_kill/<job_id>
    Sends SIGTERM to the elion.py subprocess so the user can cancel a run.
    """
    import signal
    if job_id not in _ts_jobs:
        return jsonify({"status": "error", "message": "Unknown job_id"}), 404

    pid = _ts_jobs[job_id].get("pid")
    if not pid:
        return jsonify({"status": "error", "message": "Process not started yet"}), 400

    try:
        os.kill(pid, signal.SIGTERM)
        with _ts_lock:
            _ts_jobs[job_id]["status"] = "killed"
        _delete_session(job_id)
        logger.info("[TS] job %s killed (pid %s)", job_id, pid)
        return jsonify({"status": "killed", "pid": pid})
    except ProcessLookupError:
        return jsonify({"status": "already_done"})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/vina_visualization/ts_kill_all', methods=['POST'])
def vina_ts_kill_all():
    """POST: SIGTERM all currently running elion.py subprocesses."""
    killed, already_done, errors = [], [], []
    with _ts_lock:
        job_ids = list(_ts_jobs.keys())

    for job_id in job_ids:
        job = _ts_jobs.get(job_id, {})
        pid = job.get("pid")
        if not pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            with _ts_lock:
                if job_id in _ts_jobs:
                    _ts_jobs[job_id]["status"] = "killed"
            _delete_session(job_id)
            killed.append({"job_id": job_id, "pid": pid,
                           "key": job.get("rxn_key", "")})
            logger.info("[TS] kill_all: job %s (pid %s) killed", job_id, pid)
        except ProcessLookupError:
            already_done.append(pid)
        except Exception as exc:
            errors.append({"pid": pid, "error": str(exc)})

    return jsonify({"status": "ok", "killed": killed,
                    "already_done": already_done, "errors": errors})


@app.route('/vina_visualization/ts_jobs_status', methods=['GET'])
def vina_ts_jobs_status():
    """GET: return all known TS jobs (for UI to show active/stale runs)."""
    with _ts_lock:
        jobs = [
            {
                "job_id":  jid,
                "rxn_key": job.get("rxn_key", ""),
                "status":  job.get("status", "unknown"),
                "pid":     job.get("pid"),
            }
            for jid, job in _ts_jobs.items()
        ]
    return jsonify({"jobs": jobs})

@app.route('/vina_visualization/ts_meta', methods=['GET'])
def vina_ts_meta():
    """
    GET /vina_visualization/ts_meta?dir=<output_dir>
    Scans the directory for TS result CSVs and returns per-file stats.
    Recognises two filename patterns:
      <name>_<timestamp>_mean<X.XXX>_std<X.XXX>.csv  (stats already in name)
      <name>_<timestamp>_tmp.csv                       (still running, skip)
      <name>_*.csv                                     (legacy, compute stats live)
    Returns:
      { "files": [ { "filename", "reaction", "timestamp", "mean", "std", "n" }, ... ] }
    """
    import re as _re, csv as _csv, os as _os, glob as _glob
    output_dir = request.args.get('dir', '').strip()
    if not output_dir or not _os.path.isdir(output_dir):
        return jsonify({"files": [], "error": f"dir not found: {output_dir}"}), 200

    files = []
    pat_named = _re.compile(
        r'^(?P<reaction>.+?)_(?P<ts>\d{8}_\d{6})_mean(?P<mean>[\d.]+)_std(?P<std>[\d.]+)\.csv$'
    )

    for fpath in sorted(_glob.glob(_os.path.join(output_dir, '*.csv'))):
        fname = _os.path.basename(fpath)
        if '_tmp.csv' in fname:
            continue   # still writing

        m = pat_named.match(fname)
        if m:
            # Stats encoded in filename — fast path, no CSV read needed
            try:
                mean_v = float(m.group('mean'))
                std_v  = float(m.group('std'))
                # Count rows for n
                n = 0
                with open(fpath, newline='') as f:
                    reader = _csv.DictReader(f)
                    for _ in reader:
                        n += 1
                files.append({
                    "filename":  fname,
                    "path":      fpath,
                    "reaction":  m.group('reaction'),
                    "timestamp": m.group('ts'),
                    "mean":      round(mean_v, 4),
                    "std":       round(std_v,  4),
                    "n":         n,
                })
            except Exception:
                pass
        else:
            # Legacy filename — compute stats from CSV content
            try:
                scores = []
                with open(fpath, newline='') as f:
                    reader = _csv.DictReader(f)
                    score_col = next(
                        (c for c in (reader.fieldnames or [])
                         if c.lower() in ('score','total_score','reward')),
                        None
                    )
                    if score_col:
                        for row in reader:
                            try:
                                scores.append(float(row[score_col]))
                            except (ValueError, KeyError):
                                pass
                if scores:
                    import statistics as _st
                    rxn_m = _re.match(r'^(.+?)_(\d{8})', fname)
                    reaction = rxn_m.group(1) if rxn_m else fname.replace('.csv','')
                    ts_str   = rxn_m.group(2) if rxn_m else ''
                    files.append({
                        "filename":  fname,
                        "path":      fpath,
                        "reaction":  reaction,
                        "timestamp": ts_str,
                        "mean":      round(_st.mean(scores), 4),
                        "std":       round(_st.pstdev(scores), 4),
                        "n":         len(scores),
                    })
            except Exception:
                pass

    # Sort by timestamp desc (newest first)
    files.sort(key=lambda f: f.get('timestamp',''), reverse=True)
    return jsonify({"files": files, "dir": output_dir})


@app.route('/vina_visualization/ts_download', methods=['GET'])
def vina_ts_download():
    """GET: serve a TS result CSV for download."""
    from flask import send_file
    fpath = request.args.get('path', '').strip()
    if not fpath or not os.path.isfile(fpath):
        return jsonify({"error": "File not found"}), 404
    return send_file(fpath, as_attachment=True,
                     download_name=os.path.basename(fpath),
                     mimetype='text/csv')


@app.route('/vina_visualization/ts_warmup_status', methods=['GET'])
def vina_ts_warmup_status():
    """GET: returns warmup checkpoint status for each reaction."""
    import json as _json, glob as _glob
    result = {}
    for rxn_key, cat in _REACTION_CATALOGUE.items():
        short_name = cat.get("short_name", rxn_key)
        pattern    = os.path.join(_WARMUP_DIR, f"{short_name}_*_warmup.json")
        ckpts      = sorted(_glob.glob(pattern), reverse=True)
        if ckpts:
            latest = ckpts[0]
            try:
                with open(latest) as f:
                    data = _json.load(f)
                result[rxn_key] = {
                    "cached": True,
                    "latest": latest,
                    "all": ckpts,
                    "timestamp": data.get("timestamp", ""),
                    "n_reagents": data.get("n_reagents", 0),
                    "prior_mean": data.get("prior_mean"),
                    "prior_std":  data.get("prior_std"),
                }
            except Exception:
                result[rxn_key] = {"cached": False}
        else:
            result[rxn_key] = {"cached": False}
    return jsonify({"warmup_dir": _WARMUP_DIR, "reactions": result})


@app.route('/vina_visualization/ts_warmup_clear', methods=['POST'])
def vina_ts_warmup_clear():
    """POST {rxn_key: str | 'all'}: delete warmup cache for a reaction or all."""
    data    = request.get_json(force=True) or {}
    rxn_key = data.get("rxn_key", "all")
    cleared = []
    if rxn_key == "all":
        keys = list(_REACTION_CATALOGUE.keys())
    else:
        keys = [rxn_key]
    for k in keys:
        p = _warmup_cache_path(k)
        if os.path.isfile(p):
            try:
                os.remove(p)
                cleared.append(k)
            except Exception as e:
                logger.warning("[TS] could not clear warmup cache %s: %s", p, e)
    return jsonify({"cleared": cleared})


@app.route('/vina_visualization/ts_config', methods=['GET'])
def vina_ts_config():
    """
    GET /vina_visualization/ts_config
    Reads input_TS.yml and returns the TS generator config as JSON
    for display in the modal sidebar.
    """
    import yaml as _yaml
    yml_path = os.path.join(_ELION_CWD, _ELION_YML)
    try:
        with open(yml_path) as f:
            cfg = _yaml.safe_load(f)
        ts = cfg.get('generator', {}).get('TS', {})

        # Default value of the OUTPUT DIR box, from the engine yml.
        #
        # NOTE this is NOT `visualizer.output_dir`. That one is where the UI
        # writes its own dashboard state (TS_Session / Warmup_TS / debug); this
        # one is where a RUN writes its result CSVs, and it is what the user
        # types over. Two different directories, deliberately separate keys.
        #
        # `visualizer.default_output_dir` wins; otherwise it is derived from
        # `generator.TS.results_filename`, whose directory is by definition the
        # place this engine already writes results — so the box is correct
        # without anyone adding a key.
        _vis_sec  = cfg.get('visualizer') or {}
        _explicit = _vis_sec.get('default_output_dir') if isinstance(_vis_sec, dict) else None
        if isinstance(_explicit, str) and _explicit.strip():
            _default_out = os.path.expanduser(_explicit.strip())
        else:
            # _abs_results_path folds in generator.TS.results_base first — a
            # relative results_filename would otherwise leave the bare fragment
            # ("results_TS") in the box instead of a usable directory.
            _abs = _abs_results_path(ts.get('results_filename'))
            _default_out = os.path.dirname(_abs) if _abs else ''

        return jsonify({
            'status':      'ok',
            'default_output_dir': _default_out,
            # Seeds the Import-database path box. `visualizer.bb_scan_dir` in
            # the engine yml wins; otherwise the engine's own reagent-library
            # root, so the common case is one click rather than a paste.
            # Separate from `bb_base` because the directory you SCAN is often a
            # superset of the one the engine actually draws reagents from.
            'bb_base':     _BB_BASE,
            'bb_scan_dir': (os.path.expanduser(_vis_sec['bb_scan_dir'].strip())
                            if isinstance(_vis_sec, dict)
                            and isinstance(_vis_sec.get('bb_scan_dir'), str)
                            and _vis_sec['bb_scan_dir'].strip()
                            else _BB_BASE),
            'log_level':   ts.get('log_level', 'INFO'),
            'ts_mode':     ts.get('ts_mode', '—'),
            'iterations':  ts.get('num_ts_iterations', '—'),
            'warmup':      ts.get('num_warmup_trials', '—'),
            'batch':       ts.get('eval_batch_size', '—'),
            'smarts':      ts.get('reaction_smarts', '—'),
            'reagents':    ts.get('reagent_file_list', []),
            'results':     ts.get('results_filename', '—'),
            'reactions':      list(_REACTION_CATALOGUE.keys()),   # ← new
            'warmup_cached':  {k: _warmup_exists(k) for k in _REACTION_CATALOGUE},
        })
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': f'Not found: {yml_path}'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    import glob, csv as _csv, re as _re, statistics as _stats
    from datetime import datetime as _dt

    out_dir = request.args.get('dir', '').strip()
    if not out_dir:
        return jsonify({'status': 'error', 'message': 'dir param required'}), 400
    if not os.path.isdir(out_dir):
        return jsonify({'status': 'error', 'message': f'Not a directory: {out_dir}'}), 404

    results = []

    # Match completed files: <name>_<YYYYMMDD_HHMMSS>_mean<X.XXX>_std<X.XXX>.csv
    pat_done = _re.compile(
        r'^(?P<rxn>.+?)_(?P<ts>\d{8}_\d{6})_mean(?P<mean>[\d.]+)_std(?P<std>[\d.]+)\.csv$'
    )
    # Match in-progress: <name>_<YYYYMMDD_HHMMSS>_tmp.csv
    pat_tmp  = _re.compile(
        r'^(?P<rxn>.+?)_(?P<ts>\d{8}_\d{6})_tmp\.csv$'
    )

    for fpath in sorted(glob.glob(os.path.join(out_dir, '*.csv'))):
        fname = os.path.basename(fpath)

        m = pat_done.match(fname)
        if m:
            # Read CSV to get sample size
            n = 0
            try:
                with open(fpath, newline='') as f:
                    reader = _csv.reader(f)
                    rows = list(reader)
                    n = max(0, len(rows) - 1)  # subtract header
            except Exception:
                pass
            results.append({
                'reaction':    m.group('rxn'),
                'timestamp':   m.group('ts'),
                'mean':        float(m.group('mean')),
                'std':         float(m.group('std')),
                'n':           n,
                'filename':    fname,
                'in_progress': False,
            })
            continue

        m = pat_tmp.match(fname)
        if m:
            # In-progress: compute live stats from file
            scores = []
            try:
                with open(fpath, newline='') as f:
                    reader = _csv.DictReader(f)
                    for candidate in ('score', 'total_score', 'reward', 'Score'):
                        if candidate in (reader.fieldnames or []):
                            for row in reader:
                                try: scores.append(float(row[candidate]))
                                except (ValueError, KeyError): pass
                            break
            except Exception:
                pass
            results.append({
                'reaction':    m.group('rxn'),
                'timestamp':   m.group('ts'),
                'mean':        round(_stats.mean(scores), 4) if len(scores) > 1 else (scores[0] if scores else None),
                'std':         round(_stats.pstdev(scores), 4) if len(scores) > 1 else 0,
                'n':           len(scores),
                'filename':    fname,
                'in_progress': True,
            })

    # Sort: completed before in-progress, then newest timestamp first
    results.sort(key=lambda r: (r['in_progress'], r['timestamp']), reverse=True)
    return jsonify({'status': 'ok', 'dir': out_dir, 'files': results})

@app.route('/vina_visualization/ts_cpu', methods=['GET'])
def vina_ts_cpu():
    """
    GET /vina_visualization/ts_cpu  (Server-Sent Events, ~1 s interval)
    Streams per-core CPU%, system stats, and process table for app.py + descendants.
    """
    def generate():
        import json, time
        try:
            import psutil
        except ImportError:
            yield 'data: {"error":"psutil not installed"}\n\n'
            return

        APP_SCRIPT = "app.py"
        SELF_PID   = os.getpid()
        _proc_cache: dict = {}

        def _fmt_bytes(b):
            for unit, thresh in (("g",1<<30),("m",1<<20),("k",1<<10)):
                if b >= thresh: return f"{b/thresh:.1f}{unit}"
            return f"{b}b"

        def _fmt_time(s):
            m = int(s // 60); return f"{m}:{s%60:05.2f}"

        def _cmd(p):
            try:
                parts = p.cmdline()
                for i,part in enumerate(parts):
                    if part.endswith(".py"): return " ".join(parts[i:])[-45:]
                return " ".join(parts)[-45:]
            except: return p.name()

        def _get_roots():
            roots = {}
            # Always include self (the Flask process serving this request)
            try:
                sp = psutil.Process(SELF_PID)
                p  = sp
                while True:
                    try:
                        pp = p.parent()
                        if pp is None or pp.pid <= 1: break
                        pp_cmd = " ".join(pp.cmdline() or [])
                        if APP_SCRIPT in pp_cmd or "flask" in pp_cmd.lower(): p = pp
                        else: break
                    except: break
                roots[p.pid] = p
            except: pass
            # Scan for app.py in cmdlines
            for p in psutil.process_iter(["pid","cmdline"]):
                try:
                    if APP_SCRIPT in " ".join(p.info["cmdline"] or []):
                        roots[p.pid] = p
                except: pass
            return list(roots.values())

        def _collect():
            with _ts_lock:
                dead = [j for j,d in _ts_jobs.items()
                        if d.get("status") in ("done","error","killed") and
                        not psutil.pid_exists(d.get("pid",0))]
                for j in dead: del _ts_jobs[j]

            all_pids, candidates = set(), []
            for root in _get_roots():
                try:
                    for proc in [root] + root.children(recursive=True):
                        if proc.pid not in all_pids:
                            all_pids.add(proc.pid); candidates.append(proc)
                except: pass

            with _ts_lock:
                for jid, job in list(_ts_jobs.items()):
                    pid = job.get("pid")
                    if pid and pid not in all_pids:
                        try:
                            p = psutil.Process(pid); candidates.append(p); all_pids.add(pid)
                            for c in p.children(recursive=True):
                                if c.pid not in all_pids: candidates.append(c); all_pids.add(c.pid)
                        except: pass

            pid_to_rxn = {}; pid_to_job = {}
            with _ts_lock:
                for jid, job in _ts_jobs.items():
                    pid = job.get("pid"); rxn = job.get("rxn_key","")
                    if pid: pid_to_rxn[pid] = rxn; pid_to_job[pid] = rxn
                    if pid:
                        try:
                            for c in psutil.Process(pid).children(recursive=True):
                                pid_to_job[c.pid] = rxn
                        except: pass

            # Update cache: prime new entries
            cur = {p.pid for p in candidates}
            for pid in list(_proc_cache):
                if pid not in cur: del _proc_cache[pid]
            for p in candidates:
                if p.pid not in _proc_cache:
                    _proc_cache[p.pid] = p
                    try: p.cpu_percent(interval=None)
                    except: pass

            rows = []; mt = psutil.virtual_memory().total
            for p in candidates:
                try:
                    cp = _proc_cache.get(p.pid, p)
                    with cp.oneshot():
                        cpu = round(cp.cpu_percent(interval=None), 1)
                        mi  = cp.memory_info()
                        mem = round(mi.rss / mt * 100, 1)
                        st  = cp.status()[0].upper()
                        nm  = cp.name()
                        ppid = cp.ppid()
                        try: cnum = cp.cpu_num()
                        except: cnum = None
                        try: et = cp.cpu_times(); ec = et.user + et.system
                        except: ec = 0
                    try: aff = cp.cpu_affinity()
                    except: aff = []
                    rows.append({
                        "pid": p.pid, "ppid": ppid, "name": nm, "cmd": _cmd(p),
                        "cpu": cpu, "cpu_num": cnum, "cpu_affinity": aff,
                        "mem": mem, "virt": _fmt_bytes(mi.vms), "res": _fmt_bytes(mi.rss),
                        "time": _fmt_time(ec), "status": st,
                        "rxn_key": pid_to_rxn.get(p.pid),
                        "job_tag":  pid_to_job.get(p.pid),
                        "is_elion_root": p.pid in pid_to_rxn,
                    })
                except: pass
            rows.sort(key=lambda r: r["pid"])
            return rows

        # Prime cpu_percent counters before first loop (non-blocking baseline)
        psutil.cpu_percent(percpu=True)
        time.sleep(0.5)

        while True:
            per_core = psutil.cpu_percent(percpu=True)
            total    = sum(per_core)/len(per_core) if per_core else 0
            procs    = _collect()
            load_avg = [round(x,2) for x in psutil.getloadavg()]
            vm       = psutil.virtual_memory()
            mem_info = {"used": _fmt_bytes(vm.used), "total": _fmt_bytes(vm.total), "pct": round(vm.percent,1)}
            yield f"data: {json.dumps({'cores':[round(c,1) for c in per_core],'total':round(total,1),'procs':procs,'load_avg':load_avg,'mem':mem_info})}\n\n"
            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.route('/vina_visualization/ts_gpu', methods=['GET'])
def vina_ts_gpu():
    """
    GET /vina_visualization/ts_gpu  (Server-Sent Events, ~1 s interval)
    Streams per-device NVIDIA GPU utilisation, memory, temperature, power and
    the compute processes on each device.

    WHY THIS EXISTS
    ---------------
    `ts_ui.js` has opened an EventSource on this path since the TS frontend was
    split, and the route was never written — so every TS open logged
    `ts_gpu 404 (NOT FOUND)` in the console and the Monitor tab's GPU panel
    stayed blank with no explanation. The frontend already handles
    `{"available": false, "error": ...}`, so the honest fix is to answer.

    Payload (matches `_tsGpuRender`):
        {"available": bool, "error": str,
         "gpus": [{"index", "name", "util", "mem_used_str", "mem_total_str",
                   "mem_pct", "temp", "power", "power_limit",
                   "procs": [{"pid", "name", "type", "mem", "job_tag"}]}]}

    Reads pynvml when installed and falls back to parsing `nvidia-smi`, so a
    machine with drivers but no Python bindings still gets a populated panel.
    A host with neither reports `available: false` once per second rather than
    erroring — the stream stays open so the panel recovers if a GPU appears.
    """
    def _fmt_mib(mib):
        if mib is None:
            return None
        return f"{mib/1024:.1f}G" if mib >= 1024 else f"{int(mib)}M"

    def _job_tag(pid):
        """Label a PID that belongs to one of this server's TS jobs."""
        try:
            with _ts_lock:
                for job in _ts_jobs.values():
                    proc = job.get("proc")
                    if proc is not None and getattr(proc, "pid", None) == pid:
                        return job.get("short_name") or job.get("rxn_key") or "job"
        except Exception:
            pass
        return None

    def _via_pynvml():
        import pynvml
        pynvml.nvmlInit()
        try:
            gpus = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)

                def _try(fn, *a):
                    try:
                        return fn(*a)
                    except Exception:
                        return None

                name = _try(pynvml.nvmlDeviceGetName, h)
                if isinstance(name, bytes):
                    name = name.decode()
                mem  = _try(pynvml.nvmlDeviceGetMemoryInfo, h)
                util = _try(pynvml.nvmlDeviceGetUtilizationRates, h)
                pw   = _try(pynvml.nvmlDeviceGetPowerUsage, h)
                pwl  = _try(pynvml.nvmlDeviceGetEnforcedPowerLimit, h)
                used_mib  = mem.used  / (1024 ** 2) if mem else None
                total_mib = mem.total / (1024 ** 2) if mem else None

                procs = []
                for kind, fn in (("C", pynvml.nvmlDeviceGetComputeRunningProcesses),
                                 ("G", pynvml.nvmlDeviceGetGraphicsRunningProcesses)):
                    for p in (_try(fn, h) or []):
                        pname = _try(pynvml.nvmlSystemGetProcessName, p.pid)
                        if isinstance(pname, bytes):
                            pname = pname.decode()
                        procs.append({
                            "pid":  p.pid,
                            "name": (pname or "?").split("/")[-1][:80],
                            "type": kind,
                            "mem":  _fmt_mib((p.usedGpuMemory or 0) / (1024 ** 2))
                                    if getattr(p, "usedGpuMemory", None) else None,
                            "job_tag": _job_tag(p.pid),
                        })

                gpus.append({
                    "index": i,
                    "name":  name or f"GPU {i}",
                    "util":  util.gpu if util else None,
                    "mem_used_str":  _fmt_mib(used_mib),
                    "mem_total_str": _fmt_mib(total_mib),
                    "mem_pct": round(100 * used_mib / total_mib, 1) if (used_mib and total_mib) else 0,
                    "temp":  _try(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
                    "power": round(pw / 1000) if pw is not None else None,
                    "power_limit": round(pwl / 1000) if pwl is not None else None,
                    "procs": procs,
                })
            return gpus
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _via_smi():
        """Fallback: parse nvidia-smi. Drivers present, bindings absent."""
        import subprocess as _sp
        fields = ("index,name,utilization.gpu,memory.used,memory.total,"
                  "temperature.gpu,power.draw,power.limit")
        out = _sp.run(["nvidia-smi", f"--query-gpu={fields}",
                       "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or "nvidia-smi failed").strip()[:200])

        def _num(tok, cast=float):
            tok = tok.strip()
            try:
                return cast(tok)
            except ValueError:
                return None

        gpus = []
        for line in out.stdout.strip().splitlines():
            c = line.split(",")
            if len(c) < 8:
                continue
            used, total = _num(c[3]), _num(c[4])
            gpus.append({
                "index": _num(c[0], int) or 0,
                "name":  c[1].strip(),
                "util":  _num(c[2], int),
                "mem_used_str":  _fmt_mib(used),
                "mem_total_str": _fmt_mib(total),
                "mem_pct": round(100 * used / total, 1) if (used and total) else 0,
                "temp":  _num(c[5], int),
                "power": round(_num(c[6])) if _num(c[6]) is not None else None,
                "power_limit": round(_num(c[7])) if _num(c[7]) is not None else None,
                "procs": [],
            })

        pout = _sp.run(["nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5)
        if pout.returncode == 0 and gpus:
            by_index = {g["index"]: g for g in gpus}
            for line in pout.stdout.strip().splitlines():
                c = line.split(",")
                if len(c) < 4:
                    continue
                pid = _num(c[1], int)
                # nvidia-smi reports a UUID here, not an index; attribute to the
                # first device rather than dropping the row.
                target = by_index.get(0) or gpus[0]
                target["procs"].append({
                    "pid":  pid,
                    "name": c[2].strip().split("/")[-1][:80],
                    "type": "C",
                    "mem":  _fmt_mib(_num(c[3])),
                    "job_tag": _job_tag(pid) if pid else None,
                })
        return gpus

    def generate():
        import json, time
        while True:
            try:
                gpus = _via_pynvml()
                payload = {"available": bool(gpus), "gpus": gpus}
                if not gpus:
                    payload["error"] = "NVML reports no devices."
            except Exception as nvml_exc:
                try:
                    gpus = _via_smi()
                    payload = {"available": bool(gpus), "gpus": gpus}
                    if not gpus:
                        payload["error"] = "nvidia-smi reported no devices."
                except FileNotFoundError:
                    payload = {"available": False, "gpus": [],
                               "error": "No NVIDIA GPU detected "
                                        "(neither pynvml nor nvidia-smi is available)."}
                except Exception as smi_exc:
                    payload = {"available": False, "gpus": [],
                               "error": f"GPU probe failed: {smi_exc} "
                                        f"(pynvml: {nvml_exc})"}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/vina_visualization/ts_debug_client', methods=['POST'])
def vina_ts_debug_client():
    """
    POST /vina_visualization/ts_debug_client   {"tag": str, "data": {...}}

    Appends one line to <debug>/client.log.

    WHY THIS EXISTS
    ---------------
    The reagent-panel chain ends in the browser: /ts_top5 → `_ts._jobBars` →
    `_ts._activeBars` → `_tsRenderTsBars` → rows in `#tsTsBars`. The server can
    prove it handed over a populated top5 (see the `top5` log) but cannot see
    which of the four steps after that dropped it. This puts both halves in the
    same directory, in order, so one `tail -f` shows the whole path.

    Deliberately unauthenticated and best-effort: it writes a bounded log line
    and always returns 200, so a debug failure can never break a live run.
    """
    if not _DEBUG_ENABLED:
        return jsonify({"ok": False, "reason": "debug disabled"})
    try:
        body = request.get_json(silent=True) or {}
        tag  = str(body.get("tag", "?"))[:40]
        data = json.dumps(body.get("data", {}), default=str)[:600]
        _dbg("client", f"{tag} {data}")
    except Exception as e:                                    # pragma: no cover
        _dbg("client", f"malformed client debug post: {e}")
    return jsonify({"ok": True})


@app.route('/vina_visualization/ts_debug_info', methods=['GET'])
def vina_ts_debug_info():
    """GET /vina_visualization/ts_debug_info — where the trace is and what is in it."""
    files = []
    try:
        for name in sorted(os.listdir(_DEBUG_DIR)):
            path = os.path.join(_DEBUG_DIR, name)
            if os.path.isfile(path):
                files.append({"name": name, "bytes": os.path.getsize(path)})
    except Exception as e:
        return jsonify({"enabled": _DEBUG_ENABLED, "debug_dir": _DEBUG_DIR,
                        "error": str(e), "files": []})
    return jsonify({
        "enabled":    _DEBUG_ENABLED,
        "debug_dir":  _DEBUG_DIR,
        "output_dir": _OUTPUT_DIR,
        "state_dir":  _STATE_DIR,
        "elion_cwd":  _ELION_CWD,
        "files":      files,
    })


# ══ Building-block database scan ═══════════════════════════════════════════
_BB_SCAN_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "bb_scan.py")

# Per-scan directory of duplicate records, read back a page at a time by
# /ts_scan_dupes. Under the UI's own output root, not the scanned directory:
# the library being scanned is the user's data and must not be written into.
_BB_SCAN_ROOT = os.path.join(_OUTPUT_DIR, "bb_scan")
_BB_SCAN_KEEP = 5          # scans retained; older ones are pruned on each new run


def _bb_scan_dir(scan_id: str) -> str:
    """Resolve a scan id to its directory, refusing anything path-like.

    `scan_id` arrives from the browser. Without this check a crafted value could
    walk out of the scan root and read arbitrary JSONL off the disk.
    """
    if not scan_id or not re.fullmatch(r"[0-9a-f]{8,32}", scan_id):
        return ""
    return os.path.join(_BB_SCAN_ROOT, scan_id)


def _bb_prune_scans() -> None:
    """Keep the most recent _BB_SCAN_KEEP scan directories."""
    try:
        entries = [os.path.join(_BB_SCAN_ROOT, d) for d in os.listdir(_BB_SCAN_ROOT)]
        dirs = sorted((d for d in entries if os.path.isdir(d)),
                      key=os.path.getmtime, reverse=True)
        for old in dirs[_BB_SCAN_KEEP:]:
            shutil.rmtree(old, ignore_errors=True)
    except FileNotFoundError:
        pass
    except Exception as exc:                                  # pragma: no cover
        logger.warning("[TS:bbscan] prune failed: %s", exc)


@app.route('/vina_visualization/ts_scan_dupes', methods=['GET'])
def vina_ts_scan_dupes():
    """
    GET /vina_visualization/ts_scan_dupes?scan=<id>&file=<idx>&offset=&limit=

    One page of a file's duplicate records.

    A single file in a real library can hold 10k+ duplicates, which is why these
    live on disk rather than riding the SSE stream: the scan writes them once,
    the browser reads a slice when the user asks. Returns the exact total so the
    pager can size itself without loading anything.
    """
    scan_dir = _bb_scan_dir((request.args.get("scan") or "").strip())
    if not scan_dir:
        return jsonify({"error": "bad scan id", "rows": [], "total": 0}), 400
    try:
        idx    = int(request.args.get("file", 0))
        offset = max(0, int(request.args.get("offset", 0)))
        limit  = max(1, min(200, int(request.args.get("limit", 5))))
    except (TypeError, ValueError):
        return jsonify({"error": "bad parameters", "rows": [], "total": 0}), 400

    path = os.path.join(scan_dir, f"f{int(idx)}.jsonl")
    if not os.path.isfile(path):
        return jsonify({"error": "no records for that file — the scan may have "
                                 "been pruned; re-run it",
                        "rows": [], "total": 0}), 404

    rows = []
    total = 0
    try:
        # Streamed, not read whole: a 5000-record file is ~1 MB and there is no
        # reason to hold it to hand back five rows.
        with open(path) as fh:
            for i, line in enumerate(fh):
                total += 1
                if offset <= i < offset + limit:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError as exc:
        return jsonify({"error": str(exc), "rows": [], "total": 0}), 500

    return jsonify({"rows": rows, "total": total, "offset": offset, "limit": limit})


@app.route('/vina_visualization/ts_scan_bb', methods=['GET'])
def vina_ts_scan_bb():
    """
    GET /vina_visualization/ts_scan_bb?path=<dir>   (Server-Sent Events)

    Scans every CSV under <dir> and reports, per file and per reaction, how many
    building blocks are eligible as the FIRST reagent and how many as the SECOND
    — i.e. how many match each reactant template of that reaction's SMARTS.

    Streams the scanner's JSON lines straight through as SSE `data:` frames, so
    a library that takes minutes reports per-file progress instead of blocking on
    one response. Terminates with `__DONE__`, matching the sentinel convention
    the rest of this app's streams use.

    The work runs in a SUBPROCESS (uiapp/core/bb_scan.py, invoked by path — see
    that file's docstring). RDKit substructure matching is C++ and can segfault;
    in-process that would take Flask down and destroy every running TS job. Out
    of process the worst case is a non-zero exit code reported to the browser.
    """
    raw = (request.args.get("path") or "").strip()
    if not raw:
        return Response("data: " + json.dumps(
            {"type": "error", "message": "No path given."}) + "\n\ndata: __DONE__\n\n",
            mimetype="text/event-stream")

    root = os.path.abspath(os.path.expanduser(raw))

    def generate():
        import subprocess as _sp
        import tempfile as _tf

        _dbg("bbscan", f"scan requested path={root!r}")
        if not os.path.isdir(root):
            msg = (f"No such path: {root}" if not os.path.exists(root)
                   else f"Not a directory: {root}")
            yield "data: " + json.dumps({"type": "error", "message": msg}) + "\n\n"
            yield "data: __DONE__\n\n"
            return

        # uuid4, not a hash of path+clock: `time` is not imported at module
        # scope here, and two scans of the same directory in the same second
        # must not collide onto one details directory.
        scan_id = uuid.uuid4().hex[:16]
        details_dir = os.path.join(_BB_SCAN_ROOT, scan_id)
        _bb_prune_scans()

        cat_fh = _tf.NamedTemporaryFile("w", suffix=".json", delete=False)
        try:
            # The catalogue is passed in rather than imported by the scanner so
            # _REACTION_CATALOGUE above stays the single source of truth for the
            # SMARTS: the dropdown, the run command and this scan cannot disagree.
            json.dump(_REACTION_CATALOGUE, cat_fh)
            cat_fh.close()

            cmd = [_sys.executable, _BB_SCAN_SCRIPT,
                   "--path", root, "--catalogue", cat_fh.name,
                   "--details-dir", details_dir]
            yield "data: " + json.dumps({"type": "scan_id", "scan_id": scan_id}) + "\n\n"
            _dbg("bbscan", f"exec {' '.join(cmd)}")
            proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE,
                             text=True, bufsize=1)
            n = 0
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                n += 1
                yield f"data: {line}\n\n"
            proc.wait()
            err = (proc.stderr.read() or "").strip()
            _dbg("bbscan", f"exit={proc.returncode} lines={n} stderr={err[:200]!r}")
            if proc.returncode != 0 and n == 0:
                # Non-zero with no JSON emitted means it died before it could
                # report — a crash, a missing interpreter, an import failure.
                yield "data: " + json.dumps({
                    "type": "error",
                    "message": (f"Scanner exited {proc.returncode}. "
                                f"{err[-400:] or 'No output.'}")}) + "\n\n"
        except Exception as exc:                              # pragma: no cover
            logger.exception("[TS:bbscan] failed")
            yield "data: " + json.dumps({"type": "error", "message": str(exc)}) + "\n\n"
        finally:
            try:
                os.unlink(cat_fh.name)
            except Exception:
                pass
            yield "data: __DONE__\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/vina_visualization/ts_smiles_svg', methods=['GET'])
def ts_smiles_svg():
    """
    GET /vina_visualization/ts_smiles_svg?smi=<smiles>&w=&h=

    Render an ARBITRARY SMILES to 2D SVG.

    `ts_mol_svg/<rxn_key>/<reagent_id>` resolves an id through the engine's own
    building-block CSVs, so it can only draw molecules that are already in the
    reagent index. The duplicate report draws molecules from whatever directory
    the user scanned — files that need not be in that index at all — so it needs
    a route keyed on the structure itself.

    Cached on disk under TS_Session/images/_smiles/<hash>_<w>x<h>.svg: the same
    molecule is requested once per duplicate row, and the report can show dozens.
    204 (not 404) when RDKit is unavailable or the SMILES will not parse, so a
    single bad row renders as a blank tile instead of a broken-image icon.
    """
    if _mol is None:
        return ("", 204)
    smi = (request.args.get("smi") or "").strip()
    if not smi or len(smi) > 1000:
        return ("", 204)
    try:
        w = max(40, min(400, int(request.args.get("w", 96))))
        h = max(24, min(200, int(request.args.get("h", 48))))
    except (TypeError, ValueError):
        w, h = 96, 48

    import hashlib as _hl
    key = _hl.blake2b(f"{smi}|{w}x{h}".encode(), digest_size=12).hexdigest()
    img_dir = os.path.join(_STATE_DIR, "images", "_smiles")
    try:
        os.makedirs(img_dir, exist_ok=True)
    except Exception:
        img_dir = None
    img_path = os.path.join(img_dir, f"{key}.svg") if img_dir else None

    if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        return Response(open(img_path).read(), mimetype="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})
    try:
        svg = _mol.smiles_to_svg(smi, width=w, height=h)
    except Exception as exc:
        logger.warning("[TS:mol] smiles_to_svg failed for %r: %s", smi[:60], exc)
        return ("", 204)
    # Type-check, not truthiness: a renderer that returns anything other than
    # SVG text must not be served as image/svg+xml. (Caught by the smoke test,
    # where _mol is a stub whose every call returns a truthy mock.)
    if not isinstance(svg, str) or "<svg" not in svg:
        return ("", 204)
    if img_path:
        try:
            with open(img_path, "w") as fh:
                fh.write(svg)
        except Exception:
            pass
    return Response(svg, mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})