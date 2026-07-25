# =============================================================================
# routes/_shared.py
# Shared state, paths, logger, model classes, and helper functions.
# Imported by every route module — keep imports lean here.
# =============================================================================

import os
import sys as _sys
import logging
import threading
import uuid
import queue
from pathlib import Path as _Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# All paths are defined centrally in uiapp.config (relative to the repo root,
# with environment-variable overrides). They are re-exported here under the
# names the route modules already import, so nothing downstream needs to change.
from uiapp import config as _cfg

# Anchor paths (kept for backward-compat; some routes import _VIZ directly).
_HERE = _Path(__file__).resolve().parent          # .../uiapp/routes/
_NAS  = _cfg.PKG_DIR                               # .../uiapp/
_VIZ  = _cfg.REPO_ROOT                             # repo root
_ROOT = _cfg.REPO_ROOT.parent                      # parent of the repo

# Attention-visualization sibling on sys.path (harmless if absent).
_ATTN_BASE = _cfg.ATTN_BASE
if _ATTN_BASE not in _sys.path:
    _sys.path.insert(0, _ATTN_BASE)

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("elion.routes")

# ── Model paths ───────────────────────────────────────────────────────────────
DEFAULT_FINETUNED  = _cfg.DEFAULT_FINETUNED
DEFAULT_PRETRAINED = _cfg.DEFAULT_PRETRAINED
CHEMBERT_BASE      = _cfg.CHEMBERT_BASE

# ── Vina paths ────────────────────────────────────────────────────────────────
VINA_BASE = _cfg.VINA_BASE
VINA_BIN  = _cfg.VINA_BIN
VINA_LOG  = _cfg.VINA_LOG

# ── Persistent storage ────────────────────────────────────────────────────────
CONVERTED_ROOT = _cfg.CONVERTED_ROOT

# ── Config yml ────────────────────────────────────────────────────────────────
_INPUT_ROUTES_YML = _cfg.INPUT_ROUTES_YML

# ── Action KB paths ───────────────────────────────────────────────────────────
ATTN_ACTION_KB_PATH = _cfg.ATTN_ACTION_KB_PATH
os.makedirs(os.path.dirname(ATTN_ACTION_KB_PATH), exist_ok=True)

VINA_ACTION_KB_PATH = _cfg.VINA_ACTION_KB_PATH

# ── TS paths ──────────────────────────────────────────────────────────────────
_ELION_CWD  = _cfg.ELION_CWD
_ELION_YML  = _cfg.ELION_YML
_ELION_VENV = _cfg.ELION_VENV

# ── Shared job stores ─────────────────────────────────────────────────────────
# Attention finetune jobs
_attn_finetune_jobs: dict[str, dict] = {}
_attn_finetune_lock = threading.Lock()

# Vina finetune jobs
_vina_finetune_jobs: dict[str, dict] = {}
_vina_finetune_lock = threading.Lock()

# Vina dock progress queue (byte-by-byte from subprocess stdout)
_vina_progress_q: queue.Queue = queue.Queue()

# Thompson Sampling jobs
_ts_jobs: dict[str, dict] = {}
_ts_lock = threading.Lock()