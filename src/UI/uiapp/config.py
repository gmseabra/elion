# =============================================================================
# uiapp/config.py
# -----------------------------------------------------------------------------
# Single source of truth for every filesystem path used by the UI application.
#
# Design rules
#   * In-repo paths are computed RELATIVE to the repository root (the parent of
#     the ``uiapp`` package). Nothing is hard-coded to a user's home directory.
#   * Genuinely-external resources (model weights, sibling projects, cluster
#     scratch dirs) default to a sensible repo-relative location but can be
#     overridden with an environment variable, so the app is portable.
#
# Import this module instead of re-deriving paths anywhere else.
# =============================================================================

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Anchors ──────────────────────────────────────────────────────────────────
PKG_DIR: Path = Path(__file__).resolve().parent          # .../UI/uiapp
REPO_ROOT: Path = PKG_DIR.parent                         # .../UI


def _env_path(var: str, default: Path) -> Path:
    """Return ``$var`` as a Path if set and non-empty, else *default*."""
    val = os.environ.get(var, "").strip()
    return Path(val).expanduser() if val else default


def resolve(path_str: str, base: Path = REPO_ROOT) -> str:
    """Resolve *path_str* to an absolute path.

    Absolute inputs are returned unchanged; relative inputs are resolved
    against *base* (the repo root by default). Used to turn the relative paths
    stored in ``config/input_routes.yml`` into absolute paths at load time.
    """
    if not path_str:
        return path_str
    p = Path(path_str).expanduser()
    return str(p if p.is_absolute() else (base / p))


# ── Configuration file ───────────────────────────────────────────────────────
CONFIG_DIR: Path = REPO_ROOT / "config"
INPUT_ROUTES_YML: str = str(_env_path("UI_INPUT_ROUTES_YML", CONFIG_DIR / "input_routes.yml"))

# ── Frontend (Flask template / static roots) ─────────────────────────────────
WEB_DIR: Path = REPO_ROOT / "web"
TEMPLATES_DIR: str = str(WEB_DIR / "templates")
STATIC_DIR: str = str(WEB_DIR / "static")

# ── Vina docking engine (vendored under engines/) ────────────────────────────
VINA_BASE: str = str(_env_path("UI_VINA_DIR", REPO_ROOT / "engines" / "vina"))
VINA_BIN: str = os.path.join(VINA_BASE, "vina")
VINA_LOG: str = os.path.join(VINA_BASE, "vina_non_cache.log")

# ── Runtime data ─────────────────────────────────────────────────────────────
DATA_DIR: Path = REPO_ROOT / "data"
CONVERTED_ROOT: str = str(_env_path("UI_CONVERTED_PDBQT", DATA_DIR / "converted_pdbqt"))

# ── Action knowledge base ────────────────────────────────────────────────────
#   Main KB (human-authored, read to build the FAISS index) ships in the package.
#   Shadow KB + logs (auto-generated, auditable) live under data/ at runtime.
ROUTER_DIR: Path = PKG_DIR / "router"
UI_ACTION_KB_DIR: str = str(ROUTER_DIR / "ui_action_kb")
UI_ACTION_INDEX: str = str(ROUTER_DIR / "ui_action_router.index")

KB_SHADOW_DIR: str = str(_env_path("UI_KB_SHADOW_DIR", DATA_DIR / "kb_shadow"))
ATTN_ACTION_KB_PATH: str = os.path.join(KB_SHADOW_DIR, "attn_action_kb.md")
VINA_ACTION_KB_PATH: str = os.path.join(KB_SHADOW_DIR, "vina_action_kb.md")
AUTOLEARN_LOG_PATH: str = os.path.join(KB_SHADOW_DIR, "autolearn.log")
COT_MAIN_LOG_PATH: str = os.path.join(KB_SHADOW_DIR, "cot_main.log")

# ── Sentence-transformer encoder (external model directory) ──────────────────
ENCODER_PATH: str = str(_env_path("UI_ENCODER_PATH", REPO_ROOT.parent / "LLM" / "all-MiniLM-L6-v2"))

# ── ChemBERT checkpoints (external weights; override via env) ─────────────────
CHEMBERT_BASE: str = "uiapp.CHEMBERT"
_CHEMBERT_CKPT_DIR: Path = _env_path("UI_CHEMBERT_CKPT_DIR", PKG_DIR / "CHEMBERT")
DEFAULT_FINETUNED: str = str(_env_path("UI_CHEMBERT_FINETUNED", _CHEMBERT_CKPT_DIR / "Finetuned_model_5.pt"))
DEFAULT_PRETRAINED: str = str(_env_path("UI_CHEMBERT_PRETRAINED", _CHEMBERT_CKPT_DIR / "pretrained_model.pt"))

# ── Elion engine (external sibling project driven by the TS routes) ──────────
# The engine directory is the one containing ELION_YML. Where it sits relative
# to this repo depends on how the two were checked out, so rather than guess one
# layout we probe the known ones and take the first that actually has the file.
#
# This matters more than it looks: `/vina_visualization/ts_config` and
# `/vina_visualization/ts_run` both answer **HTTP 404** when the yml is missing.
# In the browser that surfaces as "HTTP 404: NOT FOUND" and an empty Reactions
# picker — i.e. a path misconfiguration is indistinguishable from a missing
# route unless you read the response body. Probing here is what stops the app
# from booting into that state on a layout we could have detected.
ELION_YML: str = os.environ.get("ELION_TS_YML", "input_TS.yml")   # relative to ELION_CWD

def elion_candidates(repo_root: Path) -> list[Path]:
    """Directories to probe for the engine, in priority order."""
    return [
        # UI/ and elion/ are siblings: <x>/src/UI and <x>/src/elion.
        repo_root.parent / "elion",
        # UI/ sits beside a separate elion *repository* whose engine is src/elion.
        repo_root.parent / "elion" / "src" / "elion",
        # UI/ is one level deeper than the elion repo.
        repo_root.parent.parent / "elion" / "src" / "elion",
        repo_root.parent.parent / "elion",
        # The engine vendored inside this repo.
        repo_root / "elion",
    ]


def resolve_elion_cwd(repo_root: Path = REPO_ROOT, yml: str = "") -> Path:
    """First candidate directory containing *yml*, else the historical default.

    `$ELION_CWD` always wins and is never probed — an explicit setting must be
    honoured even when it is wrong, so the resulting error names the path the
    operator chose rather than silently substituting another one.

    The fallback is the pre-existing default (``<parent>/elion/src/elion``), so
    a checkout with no engine installed resolves exactly where it always did.
    """
    yml = yml or ELION_YML
    explicit = os.environ.get("ELION_CWD", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    for cand in elion_candidates(repo_root):
        try:
            if (cand / yml).is_file():
                return cand
        except OSError:                                   # pragma: no cover
            continue
    return repo_root.parent / "elion" / "src" / "elion"


ELION_CWD: str = str(resolve_elion_cwd())
ELION_YML_PATH: str = os.path.join(ELION_CWD, ELION_YML)
ELION_FOUND: bool = os.path.isfile(ELION_YML_PATH)
ELION_VENV: str = os.environ.get("ELION_VENV", "") or sys.executable


# ── UI server bind address ───────────────────────────────────────────────────
# The port lives in the engine's `input_TS.yml` under `visualizer:`, next to
# `output_dir`, so one file configures the whole run. Precedence:
#     $UI_HOST / $UI_PORT   →   visualizer.host / visualizer.port   →   default
# Anything unreadable, missing or out of range falls through to the next level;
# a config mistake must never stop the server from starting.
DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 5000


def visualizer_section(yml_path: str = "") -> dict:
    """The ``visualizer:`` mapping from the engine yml, or ``{}``.

    Every failure mode — file absent, malformed YAML, no such section, section
    is not a mapping — collapses to ``{}``. Callers supply their own defaults.
    """
    try:
        import yaml as _yaml
        with open(yml_path or ELION_YML_PATH) as fh:
            cfg = _yaml.safe_load(fh) or {}
        section = cfg.get("visualizer")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def resolve_bind(yml_path: str = "") -> tuple[str, int, str]:
    """Return ``(host, port, source)``; *source* names where the port came from."""
    vis = visualizer_section(yml_path)

    host = os.environ.get("UI_HOST", "").strip()
    host_src = "$UI_HOST"
    if not host:
        raw = vis.get("host")
        if isinstance(raw, str) and raw.strip():
            host, host_src = raw.strip(), "visualizer.host"
        else:
            host, host_src = DEFAULT_HOST, "default"

    port, port_src = None, "default"
    raw_env = os.environ.get("UI_PORT", "").strip()
    for value, src in ((raw_env, "$UI_PORT"), (vis.get("port"), "visualizer.port")):
        if value in (None, ""):
            continue
        try:
            candidate = int(str(value).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= candidate <= 65535:
            port, port_src = candidate, src
            break
    if port is None:
        port = DEFAULT_PORT

    return host, port, f"{port_src} (host from {host_src})"


UI_HOST, UI_PORT, UI_BIND_SOURCE = resolve_bind()

# ── DeepAtom (external project; must be configured to use those routes) ───────
DEEPATOM_SCRIPT: str = os.environ.get("DEEPATOM_SCRIPT", "")
DEEPATOM_DATA_DIR: str = os.environ.get("DEEPATOM_DATA_DIR", "")
DEEPATOM_TEST_TYPE: str = os.environ.get("DEEPATOM_TEST_TYPE", "vs")

# ── Attention-visualization sibling (added to sys.path by shared.py) ─────────
ATTN_BASE: str = str(_env_path("UI_ATTN_BASE", REPO_ROOT.parent / "Elion-AGI-Ecosystem" / "attention_visualization"))

# ── Triton kernel cache ──────────────────────────────────────────────────────
TRITON_CACHE_DIR: str = os.environ.get("TRITON_CACHE_DIR", "") or str(Path.home() / ".cache" / "triton")


# ─────────────────────────────────────────────────────────────────────────────
# Sessions (uiapp/routes/session_routes.py)
# SQLite-backed chat history, keyed on (client, chat_type).
# ─────────────────────────────────────────────────────────────────────────────
SESSIONS_DB: str = str(_env_path("UI_SESSIONS_DB", DATA_DIR / "elion_sessions.db"))

# ─────────────────────────────────────────────────────────────────────────────
# Chain-of-thought logs (uiapp/routes/cot_routes.py)
# ─────────────────────────────────────────────────────────────────────────────
COT_LOG_DIR: str = str(_env_path("UI_COT_LOG_DIR", KB_SHADOW_DIR))
COT_ROUTES_LOG_PATH: str = os.path.join(COT_LOG_DIR, "cot_main.log")

# ─────────────────────────────────────────────────────────────────────────────
# MGLTools / AutoDockTools (uiapp/routes/shared.py, tools_routes.py)
# The prepare_*4.py scripts run as subprocesses under an MGLTools interpreter.
# ─────────────────────────────────────────────────────────────────────────────
MGLTOOLS_DIR: str = str(_env_path("UI_MGLTOOLS_DIR", REPO_ROOT / "engines" / "vina" / "mgltools"))
PREPARE_LIGAND4: str = os.path.join(MGLTOOLS_DIR, "prepare_ligand4.py")
PREPARE_RECEPTOR4: str = os.path.join(MGLTOOLS_DIR, "prepare_receptor4.py")

# ─────────────────────────────────────────────────────────────────────────────
# Pose generation (uiapp/routes/pose_routes.py)
# Scratch + upload roots for the SMILES→3D pose tool. Every one of these is a
# WRITE target for request-supplied names, so keep them inside the repo.
# ─────────────────────────────────────────────────────────────────────────────
POSE_ROOT: str = str(_env_path("UI_POSE_ROOT", DATA_DIR / "pose"))
POSE_UPLOAD_DIR: str = str(_env_path("UI_POSE_UPLOAD_DIR", DATA_DIR / "pose" / "uploads"))
POSE_UPLOADS_DB: str = str(_env_path("UI_POSE_UPLOADS_DB", DATA_DIR / "pose" / "pose_uploads.db"))
POSE_DEBUG_DIR: str = str(_env_path("UI_POSE_DEBUG_DIR", DATA_DIR / "pose" / "debug"))
# Fallback when input_routes.yml has no pose.gign_script. Blank = the GIGN
# scorer is unavailable and the route says so, rather than shelling out to a
# path that does not exist on this host.
POSE_GIGN_SCRIPT: str = os.environ.get("UI_POSE_GIGN_SCRIPT", "")

# ─────────────────────────────────────────────────────────────────────────────
# Thompson-Sampling session + warm-up state (uiapp/routes/ts_routes.py)
# Overridable by `visualizer.output_dir` inside input_routes.yml; this is the
# fallback when that key is absent.
# ─────────────────────────────────────────────────────────────────────────────
TS_OUTPUT_DIR: str = str(_env_path("UI_TS_OUTPUT_DIR", DATA_DIR / "ts"))
TS_SESSION_DIR: str = os.path.join(TS_OUTPUT_DIR, "TS_Session")
TS_WARMUP_DIR: str = os.path.join(TS_OUTPUT_DIR, "Warmup_TS")

# ─────────────────────────────────────────────────────────────────────────────
# Reasoning — EXTERNAL app, not a feature of this server.
# The in-app reasoning gateway (blueprint + Claude client + modal) was removed;
# the sidebar button is now a plain link that opens this URL in a new tab.
# It is rendered into hub.html by hub_routes.hub(), so repointing it is an env
# change, not a template edit. Nothing here validates or reaches the URL — if
# nothing is listening on that port the new tab simply fails to connect.
# ─────────────────────────────────────────────────────────────────────────────
REASONING_URL: str = os.environ.get("UI_REASONING_URL", "http://localhost:5001/")

# ─────────────────────────────────────────────────────────────────────────────
# Elion engine sub-paths used by the TS + DeepAtom routes
# All derived from ELION_CWD so a single override repoints the whole engine.
# ─────────────────────────────────────────────────────────────────────────────
_ELION: Path = Path(ELION_CWD)
TS_ENGINE_DIR: str = str(_env_path("UI_TS_ENGINE_DIR", _ELION / "generators" / "TS"))
TS_BB_BASE: str = str(_env_path(
    "UI_TS_BB_BASE",
    _ELION / "generators" / "TS" / "data" / "Building_Blocks" /
    "BBandSMARTS-eXplore_1xx-5xx_unified"))

DEEPATOM_ROOT: str = str(_env_path("DEEPATOM_ROOT", _ELION / "properties" / "deepatom"))
DEEPATOM_SCRIPTS_DIR: str = os.path.join(DEEPATOM_ROOT, "model_split_data")
