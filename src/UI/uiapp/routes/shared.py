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


# ── MGLTools / AutoDockTools PDBQT preparation ────────────────────────────────
# The two prepare_*4.py scripts (Valdes-Tresanco Py3 fork of AutoDockTools) are
# invoked as SUBPROCESSES — they import AutoDockTools/MolKit, which only exist
# in the MGLTools env, so they can never be imported into this app.
# Location is config-driven: UI_MGLTOOLS_DIR, default <repo>/engines/vina/mgltools
MGLTOOLS_DIR      = _cfg.MGLTOOLS_DIR
PREPARE_LIGAND4   = _cfg.PREPARE_LIGAND4
PREPARE_RECEPTOR4 = _cfg.PREPARE_RECEPTOR4


def _detect_mgltools_python() -> str | None:
    """
    Locate an interpreter that can run the prepare_*4.py scripts (i.e. one that
    can import AutoDockTools + MolKit). Resolution order:

      1. $MGLTOOLS_PYTHON or $ADT_PYTHON  (explicit override — recommended)
      2. `pythonsh` on PATH               (classic MGLTools launcher)
      3. common MGLTools install locations
      4. this app's own python IF it can import AutoDockTools
         (true when the pip-installable `AutoDockTools_py3` fork is in the env)

    Returns the interpreter path, or None if nothing suitable is found.
    """
    import shutil, subprocess

    # 1. explicit env override
    for var in ("MGLTOOLS_PYTHON", "ADT_PYTHON"):
        p = os.environ.get(var)
        if p and _Path(p).is_file():
            return p

    # 2. pythonsh on PATH
    psh = shutil.which("pythonsh")
    if psh:
        return psh

    # 3. common install locations
    for cand in (
        "/opt/mgltools/bin/pythonsh",
        "/usr/local/mgltools/bin/pythonsh",
        str(_Path.home() / "mgltools" / "bin" / "pythonsh"),
    ):
        if _Path(cand).is_file():
            return cand

    # 4. current python, only if AutoDockTools is importable here
    try:
        r = subprocess.run(
            [_sys.executable, "-c", "import AutoDockTools, MolKit"],
            capture_output=True, timeout=15,
        )
        if r.returncode == 0:
            return _sys.executable
    except Exception:
        pass

    return None


# Resolved once at import; may be None if MGLTools isn't installed on this host.
MGLTOOLS_PYTHON = _detect_mgltools_python()


# ==============================================================================
# PDBQT sanitizer — shared by the converter (tools_routes) and the docking route
# (vina_dock_routes). Single source of truth so both layers behave identically.
# ==============================================================================
#
# Vina rejects TITLE / REMARK / COMPND / USER / MODEL / etc. in both receptors
# and ligands and then reports best affinity = None:
#   Receptor: "PDBQT parsing error: Unknown or inappropriate tag found in
#              rigid receptor"
#   Ligand:   "PDBQT parsing error: Unknown or inappropriate tag found in
#              flex residue or ligand.  > TITLE     LG-104"
#
# obabel (and MGLTools prepare_ligand4) copy the source PDB's TITLE/REMARK
# header — including Schrödinger/Maestro "REMARK 888 WRITTEN BY MAESTRO" and
# "TITLE  <name>" lines — straight into the PDBQT output. This strips every
# record that is not on the whitelist for the given molecule type.

# Records Vina accepts in a rigid receptor (no torsion-tree tags)
_RECEPTOR_ALLOWED = ('ATOM  ', 'HETATM', 'TER', 'END')

# Records Vina accepts in a ligand / flex residue (torsion tree, no END)
_LIGAND_ALLOWED = ('ATOM  ', 'HETATM', 'ROOT', 'ENDROOT',
                   'TORSDOF', 'BRANCH', 'ENDBRANCH')


def sanitize_pdbqt(pdbqt_path: str, mol_type: str = "receptor") -> int:
    """
    In-place cleanup of a PDBQT file produced by obabel / RDKit / MGLTools /
    Glide. Removes every record not legal for `mol_type`, and guarantees the
    chain ID (PDB column 21) is never blank (defaults to 'A').

    Returns the number of disallowed lines that were stripped.
    """
    allowed = _LIGAND_ALLOWED if mol_type == "ligand" else _RECEPTOR_ALLOWED

    clean: list[str] = []
    stripped = 0
    with open(pdbqt_path) as fh:
        for line in fh:
            s = line.rstrip('\n').rstrip('\r')

            if not any(s.startswith(p) for p in allowed):
                stripped += 1
                continue

            # Never leave a blank chain ID on a coordinate line (PDB col 21)
            if s.startswith(('ATOM  ', 'HETATM')):
                s = s.ljust(80)
                if s[21] == ' ':
                    s = s[:21] + 'A' + s[22:]

            clean.append(s + '\n')

    # Receptors must terminate with END; ligands terminate with TORSDOF
    if mol_type == "receptor":
        if clean and not clean[-1].rstrip().startswith('END'):
            clean.append('END\n')

    _Path(pdbqt_path).write_text(''.join(clean))
    return stripped


def ensure_vina_safe_pdbqt(path: str, mol_type: str = "ligand") -> dict:
    """
    Idempotent pre-flight check — call this on BOTH the receptor and ligand
    path immediately before running Vina, regardless of how the file was
    produced (this converter, a stale route, Glide/MGLTools, a manual copy).

    Cheap no-op when the file is already clean (one read, zero writes).

    Returns {"was_dirty": bool, "stripped_lines": int, "path": str}.
    """
    allowed = _LIGAND_ALLOWED if mol_type == "ligand" else _RECEPTOR_ALLOWED

    with open(path) as fh:
        lines = fh.readlines()

    bad = sum(
        1 for line in lines
        if not any(line.rstrip('\n').rstrip('\r').startswith(p) for p in allowed)
    )

    if bad == 0:
        return {"was_dirty": False, "stripped_lines": 0, "path": path}

    logger.warning(
        "[ensure_vina_safe_pdbqt] %s (%s) had %d disallowed record(s) "
        "(e.g. TITLE/REMARK) that would make Vina report 'Unknown or "
        "inappropriate tag' and best=None — self-healing before docking.",
        path, mol_type, bad,
    )
    sanitize_pdbqt(path, mol_type)
    return {"was_dirty": True, "stripped_lines": bad, "path": path}
