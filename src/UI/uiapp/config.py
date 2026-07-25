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
ELION_CWD: str = str(_env_path("ELION_CWD", REPO_ROOT.parent / "elion" / "src" / "elion"))
ELION_YML: str = os.environ.get("ELION_TS_YML", "input_TS.yml")   # relative to ELION_CWD
ELION_VENV: str = os.environ.get("ELION_VENV", "") or sys.executable

# ── DeepAtom (external project; must be configured to use those routes) ───────
DEEPATOM_SCRIPT: str = os.environ.get("DEEPATOM_SCRIPT", "")
DEEPATOM_DATA_DIR: str = os.environ.get("DEEPATOM_DATA_DIR", "")
DEEPATOM_TEST_TYPE: str = os.environ.get("DEEPATOM_TEST_TYPE", "vs")

# ── Attention-visualization sibling (added to sys.path by shared.py) ─────────
ATTN_BASE: str = str(_env_path("UI_ATTN_BASE", REPO_ROOT.parent / "Elion-AGI-Ecosystem" / "attention_visualization"))

# ── Triton kernel cache ──────────────────────────────────────────────────────
TRITON_CACHE_DIR: str = os.environ.get("TRITON_CACHE_DIR", "") or str(Path.home() / ".cache" / "triton")
