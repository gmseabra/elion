# ==============================================================================
# deepatom_saliency.py — CNN input-gradient saliency for DeepAtom
#
# Flask route:
#   POST /vina_visualization/deepatom_saliency
#
# Given a compound ID + data_dir, this route:
#   1. Loads the 3D voxel .npz produced by make_grid_mp.py
#      shape: (24, 32, 32, 32)  — 24 atom-type channels, 32³ spatial
#   2. Loads the ShuffleNetV3 model weights
#   3. Runs a forward pass with requires_grad on the input
#   4. Backpropagates the predicted pK score → input gradient
#   5. Reduces gradient over the 24 channels (L2 norm per voxel)
#   6. Reads atom xyz coords from the matching .atomtypes file
#   7. For each atom, looks up its voxel and assigns the gradient magnitude
#      as its "importance" score (analogous to Vina's this_e per-atom energy)
#   8. Returns JSON ready for Plotly 3D scatter
#
# The .atomtypes file format (from arpeggio):
#   atom_name  x  y  z  atom_type  charge  ...
# The .npz grid origin is stored as a key; if missing it falls back to
# computing it from the atom coords centre.
# ==============================================================================

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

import numpy as np
from flask import jsonify, request
from nas_storage_app import app

logger = logging.getLogger(__name__)

# ── Paths (mirrors routes.py constants) ──────────────────────────────────────
_DEEPATOM_ROOT = Path(
    "/blue/lic/huangzihang/repos/elion/src/elion/properties/deepatom"
)
_MODEL_SPLIT_DIR = _DEEPATOM_ROOT / "model_split_data" / "02_pytorch"
_DEFAULT_MODEL_DIR = _DEEPATOM_ROOT / "model_split_data" / "weight" / "ZccE"

# Grid config (matches make_grid_mp.py defaults)
_NUM_GRID    = 32
_VOXEL_SIZE  = 1.0   # Å per voxel
_INPUT_FEAT  = 24


# ==============================================================================
# Model loader (cached per process)
# ==============================================================================
_model_cache: dict = {}


def _load_model(model_dir: str):
    """Load ShuffleNetV3 from the first .tar or .pk file in model_dir."""
    if model_dir in _model_cache:
        return _model_cache[model_dir]

    # Add the pytorch source dir to sys.path so imports work
    pytorch_dir = str(_MODEL_SPLIT_DIR)
    if pytorch_dir not in sys.path:
        sys.path.insert(0, pytorch_dir)

    import torch
    from shufflenet_v3 import ShuffleNetV3

    model = ShuffleNetV3(
        input_channel=_INPUT_FEAT,
        dropout_prob=0.0,
        width_multiplier=2.0,
    )
    model = torch.nn.DataParallel(model, device_ids=[])
    model.eval()

    model_path = Path(model_dir)
    candidates = sorted(model_path.glob("*.tar")) + sorted(model_path.glob("*.pk"))
    if not candidates:
        raise FileNotFoundError(f"No model weights (.tar/.pk) found in {model_dir}")

    restore_file = str(candidates[0])
    device = torch.device("cpu")

    if restore_file.endswith(".pk"):
        model.load_state_dict(torch.load(restore_file, map_location=device))
    else:
        ckpt = torch.load(restore_file, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    logger.info("[deepatom_saliency] Loaded model from %s", restore_file)
    _model_cache[model_dir] = model
    return model


# ==============================================================================
# .atomtypes parser
# ==============================================================================

def _parse_atomtypes(atomtypes_path: str) -> list[dict]:
    """
    Parse an arpeggio .atomtypes file.
    Format (space-separated):
        atom_name  res_name  chain  res_num  x  y  z  atom_type  ...
    Returns list of {name, x, y, z, atom_type}.
    """
    atoms = []
    with open(atomtypes_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                # Try columns 4,5,6 first (common arpeggio format)
                x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                atype   = parts[7] if len(parts) > 7 else '?'
                name    = parts[0]
            except (ValueError, IndexError):
                # Fallback: first three floats in the line
                floats = []
                for p in parts:
                    try:
                        floats.append(float(p))
                    except ValueError:
                        pass
                if len(floats) < 3:
                    continue
                x, y, z = floats[0], floats[1], floats[2]
                name    = parts[0]
                atype   = '?'
            atoms.append({"name": name, "x": x, "y": y, "z": z,
                          "atom_type": atype})
    return atoms


# ==============================================================================
# Core saliency computation
# ==============================================================================

def _compute_saliency(npz_path: str, atomtypes_path: str,
                      model_dir: str) -> list[dict]:
    """
    Returns a list of per-atom dicts:
      { name, x, y, z, atom_type, importance, importance_norm }
    """
    import torch

    # ── 1. Load voxel grid ────────────────────────────────────────────────────
    npz = np.load(npz_path, allow_pickle=True)

    # The grid may be stored under various key names
    grid = None
    for key in ("data", "grid", "pocket", "feature", npz.files[0]):
        if key in npz.files:
            grid = npz[key]
            break
    if grid is None:
        grid = npz[npz.files[0]]

    # Normalise shape to (24, 32, 32, 32)
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim == 4 and grid.shape[0] == _INPUT_FEAT:
        pass  # already (C, D, H, W)
    elif grid.ndim == 4 and grid.shape[-1] == _INPUT_FEAT:
        grid = grid.transpose(3, 0, 1, 2)   # (D,H,W,C) → (C,D,H,W)
    elif grid.ndim == 5:
        # (1, C, D, H, W)
        grid = grid[0]

    # Grid origin: stored in npz or computed from atom coords
    origin = None
    for ok in ("origin", "grid_origin", "center"):
        if ok in npz.files:
            origin = np.asarray(npz[ok], dtype=np.float32)
            break

    # ── 2. Load atoms ─────────────────────────────────────────────────────────
    atoms = _parse_atomtypes(atomtypes_path)
    if not atoms:
        raise ValueError(f"No atoms parsed from {atomtypes_path}")

    if origin is None:
        # Derive origin: centre of atom cloud minus half the box
        xs = [a["x"] for a in atoms]
        ys = [a["y"] for a in atoms]
        zs = [a["z"] for a in atoms]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        cz = (max(zs) + min(zs)) / 2
        half = (_NUM_GRID * _VOXEL_SIZE) / 2
        origin = np.array([cx - half, cy - half, cz - half], dtype=np.float32)

    # ── 3. Forward + gradient ─────────────────────────────────────────────────
    model = _load_model(model_dir)
    model.eval()

    x = torch.tensor(grid[np.newaxis], dtype=torch.float32, requires_grad=True)
    score = model(x)
    # score shape varies (OutBlock4 in eval: scalar per sample)
    if score.dim() > 1:
        score = score.mean()
    else:
        score = score.squeeze()

    score.backward()

    grad = x.grad.detach().cpu().numpy()[0]   # (24, 32, 32, 32)
    # L2 norm over channels → importance volume (32, 32, 32)
    importance_vol = np.sqrt((grad ** 2).sum(axis=0))

    # ── 4. Map atom xyz → voxel index → importance ───────────────────────────
    max_imp = importance_vol.max() or 1.0

    result = []
    for a in atoms:
        vi = int(round((a["x"] - origin[0]) / _VOXEL_SIZE))
        vj = int(round((a["y"] - origin[1]) / _VOXEL_SIZE))
        vk = int(round((a["z"] - origin[2]) / _VOXEL_SIZE))

        # Clamp to grid bounds
        vi = max(0, min(_NUM_GRID - 1, vi))
        vj = max(0, min(_NUM_GRID - 1, vj))
        vk = max(0, min(_NUM_GRID - 1, vk))

        imp = float(importance_vol[vi, vj, vk])
        result.append({
            "name":           a["name"],
            "x":              round(a["x"], 4),
            "y":              round(a["y"], 4),
            "z":              round(a["z"], 4),
            "atom_type":      a["atom_type"],
            "importance":     round(imp, 6),
            "importance_norm": round(imp / max_imp, 6),
        })

    # Sort by importance descending
    result.sort(key=lambda r: r["importance"], reverse=True)
    return result


# ==============================================================================
# Flask route
# ==============================================================================

@app.route("/vina_visualization/deepatom_saliency", methods=["POST"])
def deepatom_saliency():
    """
    POST /vina_visualization/deepatom_saliency
    Body: {
        "compound_id": "BM-1-57",
        "data_dir":    "/path/to/ZccE",
        "model_dir":   "/path/to/weight/ZccE"   // optional
    }

    Returns JSON:
    {
        "status":      "success",
        "compound_id": "BM-1-57",
        "pred_pk":     7.23,          // from the forward pass
        "atoms": [
            { "name":"C1", "x":…, "y":…, "z":…,
              "atom_type":"C", "importance":0.042, "importance_norm":0.87 },
            …
        ]
    }
    """
    try:
        body        = request.get_json(force=True) or {}
        compound_id = (body.get("compound_id") or "").strip()
        data_dir    = (body.get("data_dir")    or "").strip()
        model_dir   = (body.get("model_dir")   or str(_DEFAULT_MODEL_DIR)).strip()

        if not compound_id:
            return jsonify({"status": "error",
                            "message": "compound_id is required"}), 400
        if not data_dir:
            return jsonify({"status": "error",
                            "message": "data_dir is required"}), 400

        data_path = Path(data_dir)
        if not data_path.is_dir():
            return jsonify({"status": "error",
                            "message": f"data_dir not found: {data_dir}"}), 400

        # ── Find the .npz file ────────────────────────────────────────────────
        # Pipeline writes to:  <temp>/vs/ZccE/<batch>/3d_32_24_pcmax/<compound_id>.npz
        npz_candidates = list(data_path.rglob(f"{compound_id}.npz"))
        if not npz_candidates:
            # Try without extension match — some runs include augmented suffix
            npz_candidates = list(data_path.rglob(f"{compound_id}*.npz"))
        if not npz_candidates:
            return jsonify({"status": "error",
                            "message": f"No .npz found for {compound_id} under {data_dir}"}), 404

        npz_path = str(npz_candidates[0])

        # ── Find the .atomtypes file ──────────────────────────────────────────
        at_candidates = list(data_path.rglob(f"{compound_id}.atomtypes"))
        if not at_candidates:
            at_candidates = list(data_path.rglob(f"{compound_id}*.atomtypes"))
        if not at_candidates:
            return jsonify({"status": "error",
                            "message": f"No .atomtypes found for {compound_id} under {data_dir}"}), 404

        atomtypes_path = str(at_candidates[0])

        # ── Compute saliency ──────────────────────────────────────────────────
        atoms = _compute_saliency(npz_path, atomtypes_path, model_dir)

        # Re-run forward to get the predicted pK for display
        import torch
        import numpy as _np
        npz   = _np.load(npz_path, allow_pickle=True)
        grid  = _np.asarray(npz[npz.files[0]], dtype=_np.float32)
        if grid.ndim == 4 and grid.shape[-1] == _INPUT_FEAT:
            grid = grid.transpose(3, 0, 1, 2)
        elif grid.ndim == 5:
            grid = grid[0]
        model = _load_model(model_dir)
        with torch.no_grad():
            x     = torch.tensor(grid[_np.newaxis], dtype=torch.float32)
            score = model(x)
            pred  = float(score.mean().item())

        return jsonify({
            "status":      "success",
            "compound_id": compound_id,
            "pred_pk":     round(pred, 4),
            "npz_path":    npz_path,
            "atoms":       atoms,
        })

    except Exception as exc:
        logger.exception("[deepatom_saliency] error")
        return jsonify({"status": "error", "message": str(exc)}), 500