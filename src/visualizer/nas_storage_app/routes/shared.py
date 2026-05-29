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

# ── Anchor paths ──────────────────────────────────────────────────────────────
# _shared.py lives at  .../visualizer/nas_storage_app/routes/_shared.py
# so _HERE → routes/, _NAS → nas_storage_app/, _VIZ → visualizer/
_HERE = _Path(__file__).resolve().parent          # .../routes/
_NAS  = _HERE.parent                              # .../nas_storage_app/
_VIZ  = _NAS.parent                              # .../visualizer/
_ROOT = _VIZ.parent                              # .../elion/src/

_ATTN_BASE = str(_ROOT / "Elion-AGI-Ecosystem" / "attention_visualization")
if _ATTN_BASE not in _sys.path:
    _sys.path.insert(0, _ATTN_BASE)

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("elion.routes")

# ── Model paths ───────────────────────────────────────────────────────────────
DEFAULT_FINETUNED  = str(_ROOT / "Elion-AGI-Ecosystem" / "attention_visualization" /
                         "nas_storage_app" / "CHEMBERT" / "Finetuned_model_5.pt")
DEFAULT_PRETRAINED = str(_ROOT / "Elion-AGI-Ecosystem" / "attention_visualization" /
                         "nas_storage_app" / "CHEMBERT" / "pretrained_model.pt")

CHEMBERT_BASE = "nas_storage_app.CHEMBERT"

# ── Vina paths ────────────────────────────────────────────────────────────────
VINA_BASE = str(_VIZ / "vina")
VINA_BIN  = str(_VIZ / "vina" / "vina")
VINA_LOG  = str(_VIZ / "vina" / "vina_non_cache.log")

# ── Persistent storage ────────────────────────────────────────────────────────
CONVERTED_ROOT = str(_VIZ / "nas_storage_app" / "converted_pdbqt")

# ── Config yml ────────────────────────────────────────────────────────────────
_INPUT_ROUTES_YML = str(_VIZ / "input_routes.yml")

# ── Action KB paths ───────────────────────────────────────────────────────────
ATTN_ACTION_KB_PATH = str(_ROOT / "Elion-AGI-Ecosystem" / "attention_visualization" /
                          "nas_storage_app" / ".qwen" / "attn_action_kb.md")
os.makedirs(os.path.dirname(ATTN_ACTION_KB_PATH), exist_ok=True)

VINA_ACTION_KB_PATH = str(_ROOT / "Elion-AGI-Ecosystem" / "vina_visualization" /
                          "nas_storage_app" / ".qwen" / "vina_action_kb.md")

# ── TS paths ──────────────────────────────────────────────────────────────────
_ELION_CWD  = "/home/huangzihang/repos/elion/src/elion"
_ELION_YML  = "input_TS.yml"
_ELION_VENV = "/home/huangzihang/repos/envs/elion-app/bin/python"

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