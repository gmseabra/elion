"""
routes/vina.py — Vina Docking Visualizer blueprint.
All routes served under /vina_visualization/* (url_prefix in __init__.py).
"""

import os, re, json, logging, threading, queue, subprocess, shutil, tempfile
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, Response, stream_with_context

from elion_config import (
    DEFAULT_FINETUNED, DEFAULT_PRETRAINED, CHEMBERT_BASE,
    VINA_BASE, VINA_BIN, VINA_LOG, CONVERTED_ROOT,
    VINA_GUIDED_PROMPT, VINA_SYSTEM_PROMPT,
)
from elion_shared import (
    load_chembert_model, keyword_route_vina, detect_emotion,
    _vina_chat_history, _vina_progress_q, _finetune_jobs, _get_qwen,
    pdb_to_pdbqt_receptor, pdb_to_pdbqt_ligand, autodock_type,
)

logger = logging.getLogger(__name__)
bp = Blueprint('vina', __name__)

import threading as _t2

# ==============================================================================
# AdjacencyWeightVisualizer
# ==============================================================================

class AdjacencyWeightVisualizer:
    """
    Wraps a loaded BERT_base model.  Extracts weight_a from Adjacency_embedding,
    runs inference for the predicted binding score (only meaningful for finetuned
    checkpoints), and packages everything as JSON for Plotly.
    """

    def __init__(self, full_model, device, model_path: str, model_tag: str):
        self.model      = full_model   # BERT_base (nn.Module, already on device)
        self.device     = device
        self.model_path = model_path
        self.model_tag  = model_tag    # 'finetuned' | 'pretrained'
        try:
            self.adj_emb = self.model.bert.embedding.adj
        except AttributeError:
            raise RuntimeError("Model was not loaded with adj=True.")

    # ── weight_a ──────────────────────────────────────────────────────────────

    def _weight_a(self):
        return self.adj_emb.weight_a.detach().cpu().numpy()

    def _atom_scores(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        w    = self._weight_a()
        n    = mol.GetNumAtoms()
        raw  = w[1: n + 1]
        vmin, vmax = raw.min(), raw.max()
        norm = (raw - vmin) / (vmax - vmin + 1e-8)
        return raw, norm

    # ── 3-D conformer ─────────────────────────────────────────────────────────

    @staticmethod
    def _get_3d_coords(smiles):
        from rdkit.Chem import rdDepictor
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        mol_h = Chem.AddHs(mol)
        ok    = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        if ok == 0:
            AllChem.MMFFOptimizeMolecule(mol_h)
            mol_h  = Chem.RemoveHs(mol_h)
            conf   = mol_h.GetConformer()
            coords = np.array([conf.GetAtomPosition(i)
                               for i in range(mol_h.GetNumAtoms())])
            return mol_h, coords
        mol2d = Chem.MolFromSmiles(smiles)
        rdDepictor.Compute2DCoords(mol2d)
        conf   = mol2d.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y, 0.0]
                           for i in range(mol2d.GetNumAtoms())])
        return mol2d, coords

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_score(self, smiles):
        """
        Forward pass → scalar binding affinity (kcal/mol).
        Returns None with a warning if the model is pretrained-only
        (output layer weights are random, so the number is meaningless).
        """
        if self.model_tag == "pretrained":
            return None   # random linear head — don't show as a real score

        from nas_storage_app.CHEMBERT.chembert import SMILES_Dataset, Vocab
        from torch.utils.data import DataLoader
        vocab   = Vocab()
        dataset = SMILES_Dataset([smiles], vocab=vocab)
        loader  = DataLoader(dataset, batch_size=1, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                inp   = batch["smiles_bert_input"].to(self.device)
                amask = batch["smiles_bert_adj_mask"].float().to(self.device)
                amat  = batch["smiles_bert_adjmat"].float().to(self.device)
                pos   = torch.arange(256).repeat(inp.size(0), 1).to(self.device)
                out   = self.model(inp, pos, adj_mask=amask, adj_mat=amat)
                return float(out[:, 0, 0].item())

    # ── payload ───────────────────────────────────────────────────────────────

    def build_3d_payload(self, smiles):
        raw, norm   = self._atom_scores(smiles)
        mol, coords = self._get_3d_coords(smiles)
        if mol is None:
            raise ValueError(f"Could not generate 3-D structure for: {smiles}")

        try:
            score = self.predict_score(smiles)
        except Exception as e:
            logger.warning(f"Score prediction failed for {smiles}: {e}")
            score = None

        def _coolwarm_hex(t):
            t = float(np.clip(t, 0, 1))
            if t < 0.5:
                s = t * 2
                r = int(59  + s * (221 - 59))
                g = int(76  + s * (220 - 76))
                b = int(192 + s * (220 - 192))
            else:
                s = (t - 0.5) * 2
                r = int(221 + s * (180 - 221))
                g = int(220 + s * (4   - 220))
                b = int(220 + s * (38  - 220))
            return f"#{r:02x}{g:02x}{b:02x}"

        atoms = []
        for i in range(mol.GetNumAtoms()):
            x, y, z = coords[i]
            atoms.append({
                "idx":         i,
                "symbol":      mol.GetAtomWithIdx(i).GetSymbol(),
                "x":           round(float(x), 4),
                "y":           round(float(y), 4),
                "z":           round(float(z), 4),
                "weight_raw":  round(float(raw[i]),  5),
                "weight_norm": round(float(norm[i]), 5),
                "color":       _coolwarm_hex(norm[i]),
            })

        bonds = [
            {"begin": b.GetBeginAtomIdx(),
             "end":   b.GetEndAtomIdx(),
             "order": int(b.GetBondTypeAsDouble())}
            for b in mol.GetBonds()
        ]

        w_all    = self._weight_a()
        n_atoms  = mol.GetNumAtoms()

        # Atom-space slice: w_atom[i] = w_all[i+1], so atom idx i == bar x i.
        # This eliminates the <start>-token offset that caused pos/idx mismatch.
        w_atom       = w_all[1: n_atoms + 1]
        top_atom_idx = np.argsort(np.abs(w_atom))[::-1][:50].tolist()
        atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]

        return {
            "smiles":          smiles,
            "predicted_score": score,
            "model_tag":       self.model_tag,
            "model_path":      self.model_path,
            "atoms":           atoms,
            "bonds":           bonds,
            "weight_vector": {
                # atom-space: index i == atom idx i == 3D-hover idx i
                "values":       [round(float(v), 5) for v in w_atom],
                "top_indices":  top_atom_idx,
                "atom_symbols": atom_symbols,
                "n_atoms":      n_atoms,
                # full raw array (seq positions 0..255) still available
                "raw_values":   [round(float(v), 5) for v in w_all],
                "max_len":      int(len(w_all)),
            },
        }


# ==============================================================================
# Singleton cache  (keyed by model_path so switching is instant on repeat)
# ==============================================================================
_viz_cache: dict[str, AdjacencyWeightVisualizer] = {}

def _get_viz(model_path: str) -> AdjacencyWeightVisualizer:
    global _viz_cache
    model_path = model_path.strip()
    if model_path not in _viz_cache:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        full_model, device, tag = _load_chembert(model_path)
        _viz_cache[model_path]  = AdjacencyWeightVisualizer(
            full_model, device, model_path, tag
        )
        logger.info(f"Loaded {tag} CHEM-BERT from {model_path}")
    return _viz_cache[model_path]


# ==============================================================================
# Routes
# ==============================================================================

VINA_ACTION_KB_PATH = "/blue/lic/huangzihang/repos/Elion-AGI-Ecosystem/vina_visualization/nas_storage_app/.qwen/vina_action_kb.md"
os.makedirs(os.path.dirname(VINA_ACTION_KB_PATH), exist_ok=True)

DEFAULT_FINETUNED   = "/blue/lic/huangzihang/repos/Elion-AGI-Ecosystem/attention_visualization/nas_storage_app/CHEMBERT/Finetuned_model_5.pt"
DEFAULT_PRETRAINED  = "/blue/lic/huangzihang/repos/Elion-AGI-Ecosystem/attention_visualization/nas_storage_app/CHEMBERT/pretrained_model.pt"

CHEMBERT_BASE = "nas_storage_app.CHEMBERT"   # import root for chembert modules


# ==============================================================================
# Smart model loader
# ==============================================================================
# The two checkpoint formats differ structurally:
#
#   pretrained_model.pt   → saved as bare Smiles_BERT
#                           keys start with  "embedding."  "transformer_encoder."
#                           NO "bert." prefix, NO "linear." key
#
#   Finetuned_model_5.pt  → saved as BERT_base(Smiles_BERT, Linear(1024,1))
#                           keys start with  "bert.embedding."  "bert.transformer_encoder."
#                           HAS "linear.weight" key
#
# Strategy: inspect the first key of the checkpoint.  If it starts with "bert.",
# load via chembert_model (which wraps in BERT_base).  Otherwise load the raw
# Smiles_BERT state dict directly into a freshly constructed BERT_base.

def _load_chembert(model_path: str):
    """
    Returns a loaded BERT_base model + device, regardless of whether
    model_path points to a pretrained or finetuned checkpoint.
    Also returns a string tag: 'finetuned' | 'pretrained'.
    """
    from nas_storage_app.CHEMBERT.model   import Smiles_BERT, BERT_base
    from nas_storage_app.CHEMBERT.chembert import Vocab

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state  = torch.load(model_path, map_location=device)

    # Detect format by inspecting first key
    first_key = next(iter(state))
    is_finetuned = first_key.startswith("bert.")

    vocab = Vocab()

    smiles_bert = Smiles_BERT(
        len(vocab),
        max_len=256, nhead=16,
        feature_dim=1024, feedforward_dim=1024,
        nlayers=8, adj=True, dropout_rate=0,
    )

    if is_finetuned:
        # Checkpoint already has BERT_base structure → load directly
        output_layer = nn.Linear(1024, 1)
        full_model   = BERT_base(smiles_bert, output_layer)
        full_model.load_state_dict(state)
        tag = "finetuned"
    else:
        # Checkpoint is bare Smiles_BERT → load into smiles_bert, then wrap
        smiles_bert.load_state_dict(state)
        output_layer = nn.Linear(1024, 1)   # random head (no regression labels)
        full_model   = BERT_base(smiles_bert, output_layer)
        tag = "pretrained"

    full_model.to(device)
    full_model.eval()
    return full_model, device, tag


# ==============================================================================
# AdjacencyWeightVisualizer
# ==============================================================================

class AdjacencyWeightVisualizer:
    """
    Wraps a loaded BERT_base model.  Extracts weight_a from Adjacency_embedding,
    runs inference for the predicted binding score (only meaningful for finetuned
    checkpoints), and packages everything as JSON for Plotly.
    """

    def __init__(self, full_model, device, model_path: str, model_tag: str):
        self.model      = full_model   # BERT_base (nn.Module, already on device)
        self.device     = device
        self.model_path = model_path
        self.model_tag  = model_tag    # 'finetuned' | 'pretrained'
        try:
            self.adj_emb = self.model.bert.embedding.adj
        except AttributeError:
            raise RuntimeError("Model was not loaded with adj=True.")

    # ── weight_a ──────────────────────────────────────────────────────────────

    def _weight_a(self):
        return self.adj_emb.weight_a.detach().cpu().numpy()

    def _atom_scores(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        w    = self._weight_a()
        n    = mol.GetNumAtoms()
        raw  = w[1: n + 1]
        vmin, vmax = raw.min(), raw.max()
        norm = (raw - vmin) / (vmax - vmin + 1e-8)
        return raw, norm

    # ── 3-D conformer ─────────────────────────────────────────────────────────

    @staticmethod
    def _get_3d_coords(smiles):
        from rdkit.Chem import rdDepictor
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        mol_h = Chem.AddHs(mol)
        ok    = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        if ok == 0:
            AllChem.MMFFOptimizeMolecule(mol_h)
            mol_h  = Chem.RemoveHs(mol_h)
            conf   = mol_h.GetConformer()
            coords = np.array([conf.GetAtomPosition(i)
                               for i in range(mol_h.GetNumAtoms())])
            return mol_h, coords
        mol2d = Chem.MolFromSmiles(smiles)
        rdDepictor.Compute2DCoords(mol2d)
        conf   = mol2d.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y, 0.0]
                           for i in range(mol2d.GetNumAtoms())])
        return mol2d, coords

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_score(self, smiles):
        """
        Forward pass → scalar binding affinity (kcal/mol).
        Returns None with a warning if the model is pretrained-only
        (output layer weights are random, so the number is meaningless).
        """
        if self.model_tag == "pretrained":
            return None   # random linear head — don't show as a real score

        from nas_storage_app.CHEMBERT.chembert import SMILES_Dataset, Vocab
        from torch.utils.data import DataLoader
        vocab   = Vocab()
        dataset = SMILES_Dataset([smiles], vocab=vocab)
        loader  = DataLoader(dataset, batch_size=1, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                inp   = batch["smiles_bert_input"].to(self.device)
                amask = batch["smiles_bert_adj_mask"].float().to(self.device)
                amat  = batch["smiles_bert_adjmat"].float().to(self.device)
                pos   = torch.arange(256).repeat(inp.size(0), 1).to(self.device)
                out   = self.model(inp, pos, adj_mask=amask, adj_mat=amat)
                return float(out[:, 0, 0].item())

    # ── payload ───────────────────────────────────────────────────────────────

    def build_3d_payload(self, smiles):
        raw, norm   = self._atom_scores(smiles)
        mol, coords = self._get_3d_coords(smiles)
        if mol is None:
            raise ValueError(f"Could not generate 3-D structure for: {smiles}")

        try:
            score = self.predict_score(smiles)
        except Exception as e:
            logger.warning(f"Score prediction failed for {smiles}: {e}")
            score = None

        def _coolwarm_hex(t):
            t = float(np.clip(t, 0, 1))
            if t < 0.5:
                s = t * 2
                r = int(59  + s * (221 - 59))
                g = int(76  + s * (220 - 76))
                b = int(192 + s * (220 - 192))
            else:
                s = (t - 0.5) * 2
                r = int(221 + s * (180 - 221))
                g = int(220 + s * (4   - 220))
                b = int(220 + s * (38  - 220))
            return f"#{r:02x}{g:02x}{b:02x}"

        atoms = []
        for i in range(mol.GetNumAtoms()):
            x, y, z = coords[i]
            atoms.append({
                "idx":         i,
                "symbol":      mol.GetAtomWithIdx(i).GetSymbol(),
                "x":           round(float(x), 4),
                "y":           round(float(y), 4),
                "z":           round(float(z), 4),
                "weight_raw":  round(float(raw[i]),  5),
                "weight_norm": round(float(norm[i]), 5),
                "color":       _coolwarm_hex(norm[i]),
            })

        bonds = [
            {"begin": b.GetBeginAtomIdx(),
             "end":   b.GetEndAtomIdx(),
             "order": int(b.GetBondTypeAsDouble())}
            for b in mol.GetBonds()
        ]

        w_all    = self._weight_a()
        n_atoms  = mol.GetNumAtoms()

        # Atom-space slice: w_atom[i] = w_all[i+1], so atom idx i == bar x i.
        # This eliminates the <start>-token offset that caused pos/idx mismatch.
        w_atom       = w_all[1: n_atoms + 1]
        top_atom_idx = np.argsort(np.abs(w_atom))[::-1][:50].tolist()
        atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]

        return {
            "smiles":          smiles,
            "predicted_score": score,
            "model_tag":       self.model_tag,
            "model_path":      self.model_path,
            "atoms":           atoms,
            "bonds":           bonds,
            "weight_vector": {
                # atom-space: index i == atom idx i == 3D-hover idx i
                "values":       [round(float(v), 5) for v in w_atom],
                "top_indices":  top_atom_idx,
                "atom_symbols": atom_symbols,
                "n_atoms":      n_atoms,
                # full raw array (seq positions 0..255) still available
                "raw_values":   [round(float(v), 5) for v in w_all],
                "max_len":      int(len(w_all)),
            },
        }


# ==============================================================================
# Singleton cache  (keyed by model_path so switching is instant on repeat)
# ==============================================================================
_viz_cache: dict[str, AdjacencyWeightVisualizer] = {}

def _get_viz(model_path: str) -> AdjacencyWeightVisualizer:
    global _viz_cache
    model_path = model_path.strip()
    if model_path not in _viz_cache:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        full_model, device, tag = _load_chembert(model_path)
        _viz_cache[model_path]  = AdjacencyWeightVisualizer(
            full_model, device, model_path, tag
        )
        logger.info(f"Loaded {tag} CHEM-BERT from {model_path}")
    return _viz_cache[model_path]


# ==============================================================================
# Routes
# ==============================================================================



# ==============================================================================
# ── HUB LANDING PAGE ──────────────────────────────────────────────────────────
# ==============================================================================



# ══ VINA VISUALIZER  /vina_visualization/* ════════════════════════════════════
# ==============================================================================
@bp.route('/')
def vina_home():
    return render_template('hub.html')


@bp.route('/chembert_models', methods=['GET'])
def vina_chembert_models():
    """
    GET /chembert_models
    Returns the two preset paths so the front-end can populate a selector.
    """
    return jsonify({
        "status": "success",
        "presets": [
            {"label": "Finetuned (Finetuned_model_5.pt)", "path": DEFAULT_FINETUNED,  "tag": "finetuned"},
            {"label": "Pretrained (pretrained_model.pt)",  "path": DEFAULT_PRETRAINED, "tag": "pretrained"},
        ]
    })


@bp.route('/adj_3d_viz', methods=['POST'])
def vina_adj_3d_viz():
    """
    POST /adj_3d_viz
    Body: { "smiles": "<SMILES>", "model_path": "<optional path>" }

    Returns:
    {
      "status":          "success",
      "smiles":          str,
      "predicted_score": float | null,
      "model_tag":       "finetuned" | "pretrained",
      "model_path":      str,
      "atoms":           [...],
      "bonds":           [...],
      "weight_vector":   {...}
    }
    """
    try:
        data       = request.get_json(force=True) or {}
        smiles     = data.get("smiles", "").strip()
        model_path = data.get("model_path", DEFAULT_FINETUNED).strip() or DEFAULT_FINETUNED

        if not smiles:
            return jsonify({"status": "error", "message": "No SMILES provided."}), 400
        if Chem.MolFromSmiles(smiles) is None:
            return jsonify({"status": "error",
                            "message": f"Invalid SMILES: {smiles}"}), 400

        viz               = _get_viz(model_path)
        payload           = viz.build_3d_payload(smiles)
        payload["status"] = "success"
        return jsonify(payload)

    except FileNotFoundError as exc:
        logger.error(f"adj_3d_viz model not found: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:
        logger.error(f"adj_3d_viz error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@bp.route('/chembert_compare', methods=['POST'])
def vina_chembert_compare():
    """
    POST /chembert_compare
    Body: { "smiles_a": str, "smiles_b": str, "model_path": str (optional) }

    Returns both payloads in one round-trip for side-by-side rendering.
    """
    try:
        data       = request.get_json(force=True) or {}
        smiles_a   = data.get("smiles_a", "").strip()
        smiles_b   = data.get("smiles_b", "").strip()
        model_path = data.get("model_path", DEFAULT_FINETUNED).strip() or DEFAULT_FINETUNED

        errors = {}
        if not smiles_a:
            errors["smiles_a"] = "No SMILES provided."
        elif Chem.MolFromSmiles(smiles_a) is None:
            errors["smiles_a"] = f"Invalid SMILES: {smiles_a}"
        if not smiles_b:
            errors["smiles_b"] = "No SMILES provided."
        elif Chem.MolFromSmiles(smiles_b) is None:
            errors["smiles_b"] = f"Invalid SMILES: {smiles_b}"
        if errors:
            return jsonify({"status": "error", "errors": errors}), 400

        viz       = _get_viz(model_path)
        payload_a = viz.build_3d_payload(smiles_a)
        payload_b = viz.build_3d_payload(smiles_b)

        return jsonify({"status": "success", "a": payload_a, "b": payload_b})

    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:
        logger.error(f"chembert_compare error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500

# ==============================================================================
# Fine-tuning routes
# ==============================================================================

# Job registry: job_id -> {"status": "running"|"done"|"error", "queue": Queue}
_finetune_jobs: dict[str, dict] = {}
_finetune_lock = threading.Lock()


def _run_finetune_job(job_id: str, smiles_file: str, pretrained_model: str,
                      max_time: int, max_epochs: int, task: str):
    """
    Runs nas_storage_app/finetune_CHEMBERT.py as a subprocess.
    Interface: finetune_CHEMBERT.py [-m MODEL] [-t MAX_TIME] smiles_file
    """
    import subprocess, sys as _sys
    q = _finetune_jobs[job_id]["queue"]
    def _push(line: str): q.put(line)

    _script   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "finetune_CHEMBERT.py")
    _viz_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        if not os.path.isfile(_script):
            raise FileNotFoundError(f"finetune_CHEMBERT.py not found at {_script}")

        cmd = [_sys.executable, _script]
        if pretrained_model:
            cmd += ["-m", pretrained_model]
        if max_time:
            cmd += ["-t", str(max_time)]
        cmd += [smiles_file]   # positional — must be last

        _env = os.environ.copy()
        _env["PYTHONPATH"] = _viz_root + os.pathsep + _env.get("PYTHONPATH", "")

        logger.info("[Finetune] cmd: %s", " ".join(cmd))
        _push(f"$ {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=_viz_root, env=_env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line: _push(line)
        proc.wait()

        if proc.returncode == 0:
            with _finetune_lock: _finetune_jobs[job_id]["status"] = "done"
        else:
            with _finetune_lock: _finetune_jobs[job_id]["status"] = "error"
            _push(f"ERROR: process exited with code {proc.returncode}")
        _push("__DONE__")

    except Exception as exc:
        logger.error("[Finetune] job %s failed: %s", job_id, exc)
        with _finetune_lock: _finetune_jobs[job_id]["status"] = "error"
        _push(f"ERROR: {exc}")
        _push("__DONE__")


@bp.route('/finetune_chembert', methods=['POST'])
def vina_finetune_chembert():
    """
    POST /finetune_chembert
    Body: {
        "smiles_file":       str,   # path to CSV with SMILES,LABELS
        "pretrained_model":  str,   # optional, defaults to DEFAULT_PRETRAINED
        "max_time":          int,   # minutes, default 720
        "max_epochs":        int,   # default 15
        "task":              str    # "regression" | "classification", default "regression"
    }
    Returns: { "status": "started", "job_id": str }
    """
    try:
        data            = request.get_json(force=True) or {}
        smiles_file     = data.get("smiles_file", "").strip()
        pretrained_model = (data.get("pretrained_model") or "").strip()  # use exactly what the UI sends; no hidden default
        max_time        = int(data.get("max_time", 720))
        max_epochs      = int(data.get("max_epochs", 15))
        task            = data.get("task", "regression").strip()

        if not smiles_file:
            return jsonify({"status": "error", "message": "No smiles_file provided."}), 400
        if not Path(smiles_file).is_file():
            return jsonify({"status": "error", "message": f"File not found: {smiles_file}"}), 404
        if task not in ("regression", "classification"):
            return jsonify({"status": "error", "message": "task must be 'regression' or 'classification'"}), 400

        job_id = str(uuid.uuid4())
        with _finetune_lock:
            _finetune_jobs[job_id] = {
                "status": "running",
                "queue":  queue.Queue(),
                "params": {
                    "smiles_file":      smiles_file,
                    "pretrained_model": pretrained_model,
                    "max_time":         max_time,
                    "max_epochs":       max_epochs,
                    "task":             task,
                },
            }

        t = threading.Thread(
            target=_run_finetune_job,
            args=(job_id, smiles_file, pretrained_model, max_time, max_epochs, task),
            daemon=True,
        )
        t.start()

        return jsonify({"status": "started", "job_id": job_id})

    except Exception as exc:
        logger.error(f"finetune_chembert error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@bp.route('/finetune_status/<job_id>', methods=['GET'])
def vina_finetune_status(job_id: str):
    """
    GET /finetune_status/<job_id>
    Server-Sent Events stream: each event is a log line.
    Sends "data: __DONE__\\n\\n" when the job finishes.
    """
    if job_id not in _finetune_jobs:
        return jsonify({"status": "error", "message": "Unknown job_id"}), 404

    job = _finetune_jobs[job_id]
    q   = job["queue"]

    def generate():
        while True:
            try:
                line = q.get(timeout=30)
                yield f"data: {line}\n\n"
                if line == "__DONE__":
                    break
            except queue.Empty:
                # Send a keep-alive comment
                yield ": keep-alive\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ==============================================================================
# Prepare SMILES training data  (AGI_get_smile_fine_tune_ChemBERT logic)
# ==============================================================================

@bp.route('/prepare_smiles', methods=['POST'])
def vina_prepare_smiles():
    """
    POST /prepare_smiles
    Body: {
        "subset_path":    str,   # CSV with Ligand_ID + Affinity columns
        "reference_path": str,   # CSV with Name + SMILES columns
        "output_path":    str    # destination .smi / .csv path
    }

    Merges subset_path (docking scores) with reference_path (SMILES) on
    Ligand_ID == Name, writes SMILES,LABELS to output_path, and returns a
    preview of the first 10 rows plus row count.
    """
    try:
        import pandas as pd

        data           = request.get_json(force=True) or {}
        subset_path    = data.get("subset_path",    "").strip()
        reference_path = data.get("reference_path", "").strip()
        output_path    = data.get("output_path",    "").strip()

        # ── Validate inputs ────────────────────────────────────────────────────
        missing = [k for k, v in [
            ("subset_path",    subset_path),
            ("reference_path", reference_path),
            ("output_path",    output_path),
        ] if not v]
        if missing:
            return jsonify({"status": "error",
                            "message": f"Missing fields: {', '.join(missing)}"}), 400

        for label, p in [("subset_path",    subset_path),
                         ("reference_path", reference_path)]:
            if not Path(p).is_file():
                return jsonify({"status": "error",
                                "message": f"File not found ({label}): {p}"}), 404

        # ── Load & merge ───────────────────────────────────────────────────────
        subset_df    = pd.read_csv(subset_path)
        reference_df = pd.read_csv(reference_path)

        required_subset = {"Ligand_ID", "Affinity"}
        required_ref    = {"Name", "SMILES"}
        missing_s = required_subset - set(subset_df.columns)
        missing_r = required_ref    - set(reference_df.columns)
        if missing_s:
            return jsonify({"status": "error",
                            "message": f"subset_path missing columns: {missing_s}"}), 400
        if missing_r:
            return jsonify({"status": "error",
                            "message": f"reference_path missing columns: {missing_r}"}), 400

        merged_df = subset_df.merge(
            reference_df[["Name", "SMILES"]],
            left_on="Ligand_ID",
            right_on="Name",
            how="inner",
        )

        final_df = (merged_df[["SMILES", "Affinity"]]
                    .rename(columns={"Affinity": "LABELS"}))

        # ── Write output ───────────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)

        logger.info(f"prepare_smiles: wrote {len(final_df)} rows to {output_path}")

        preview = final_df.head(10).to_dict(orient="records")

        return jsonify({
            "status":       "success",
            "total_rows":   len(final_df),
            "output_path":  output_path,
            "preview":      preview,
        })

    except Exception as exc:
        logger.error(f"prepare_smiles error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500

# ==============================================================================
# Vina Score Decomposition Visualizer
# ==============================================================================

import re
import math
from collections import defaultdict

_XS_META = {
    0:("C_H",True,False,False), 1:("C_P",True,False,False),
    2:("N_P",False,False,False), 3:("N_D",False,True,False),
    4:("N_A",False,False,True), 5:("N_DA",False,True,True),
    6:("O_P",False,False,False), 7:("O_D",False,True,False),
    8:("O_A",False,False,True), 9:("O_DA",False,True,True),
    10:("S_P",False,False,False), 11:("P_P",False,False,False),
    12:("F_H",False,False,False), 13:("Cl_H",False,False,False),
    14:("Br_H",False,False,False), 15:("I_H",False,False,False),
    16:("Met",False,False,False),
}
_XS_RADIUS = {
    0:1.9,1:1.9,2:1.8,3:1.8,4:1.8,5:1.8,
    6:1.7,7:1.7,8:1.7,9:1.7,
    10:2.0,11:2.1,12:1.5,13:1.8,14:2.0,15:2.2,16:1.2,
}
_VINA_W = {
    "gauss1":-0.035579,"gauss2":-0.005156,"repulsion":0.840245,
    "hydrophobic":-0.035069,"hbond":-0.587439,
}

def _parse_vina_log(log_text):
    pair_re  = re.compile(
        r'\[non_cache::eval pair\]\s+lig_atom=(\d+)\(xs=(\d+)\)\s+'
        r'rec_atom=(\d+)\(xs=(\d+)\)\s+opt\(ri\+rj\)=([\d.]+)\s+'
        r'r=([\d.]+)\s+s=r-opt=([-\d.]+)\s+pair_e=([-\d.]+)'
    )
    atom_re  = re.compile(
        r'\[non_cache::eval lig_atom=(\d+)\(xs=(\d+)\)\]\s+'
        r'this_e\(after curl\)=([-\d.]+)\s+out_of_bounds_penalty=([-\d.]+)'
    )
    total_re = re.compile(r'\[non_cache::eval TOTAL\] e=([-\d.]+)')
    mode_re  = re.compile(r'\[mode (\d+) energy assembly\]')
    ev_re    = re.compile(r'total \(E_vina\)\s*=\s*([-\d.]+)')
    conf_re  = re.compile(r'conf_independent \(Nrot penalty\)\s*=\s*([-\d.]+)')

    lig_atoms, lig_order, total_e = {}, [], None
    mode_energies, cur_mode = {}, None

    for line in log_text.splitlines():
        m = mode_re.search(line)
        if m: cur_mode = int(m.group(1)); mode_energies[cur_mode] = {}; continue
        if cur_mode:
            m = ev_re.search(line)
            if m: mode_energies[cur_mode]['total_e'] = float(m.group(1)); continue
            m = conf_re.search(line)
            if m: mode_energies[cur_mode]['nrot_penalty'] = float(m.group(1)); continue

        m = pair_re.search(line)
        if m:
            li,lxs,ri,rxs = int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4))
            opt,r,s,pe    = float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8))
            if li not in lig_atoms:
                lig_atoms[li] = {"xs":lxs,"pairs":[],"this_e":0.0,"out_of_bounds":0.0}
                lig_order.append(li)
            lig_atoms[li]["pairs"].append({"rec_idx":ri,"rec_xs":rxs,"opt":opt,"r":r,"s":s,"pair_e":pe})
            continue

        m = atom_re.search(line)
        if m:
            li,lxs,this_e,oob = int(m.group(1)),int(m.group(2)),float(m.group(3)),float(m.group(4))
            if li not in lig_atoms:
                lig_atoms[li] = {"xs":lxs,"pairs":[],"this_e":0.0,"out_of_bounds":0.0}
                lig_order.append(li)
            lig_atoms[li]["this_e"]       = this_e
            lig_atoms[li]["out_of_bounds"] = oob
            continue

        m = total_re.search(line)
        if m: total_e = float(m.group(1))

    return lig_atoms, total_e, lig_order, mode_energies


def _term_breakdown(xs1, xs2, r):
    opt = _XS_RADIUS.get(xs1,1.8) + _XS_RADIUS.get(xs2,1.8)
    s   = r - opt
    g1  = math.exp(-((s/0.5)**2))
    g2  = math.exp(-(((s-3.0)/2.0)**2))
    rp  = s*s if s < 0 else 0.0
    m1  = _XS_META.get(xs1,("?",False,False,False))
    m2  = _XS_META.get(xs2,("?",False,False,False))
    hp  = (1.0 if s<=0.5 else 0.0 if s>=1.5 else 1.0-(s-0.5)) if (m1[1] and m2[1]) else 0.0
    hb_ok = (m1[2] and m2[3]) or (m1[3] and m2[2])
    hb  = (1.0 if s<=-0.7 else 0.0 if s>=0.0 else 1.0-s/(-0.7)) if hb_ok else 0.0
    w   = _VINA_W
    return {
        "s": round(s,5),
        "gauss1_raw": round(g1,5),    "gauss2_raw": round(g2,5),
        "repulsion_raw": round(rp,5), "hydrophobic_raw": round(hp,5), "hbond_raw": round(hb,5),
        "gauss1_w": round(w["gauss1"]*g1,6),       "gauss2_w": round(w["gauss2"]*g2,6),
        "repulsion_w": round(w["repulsion"]*rp,6),  "hydrophobic_w": round(w["hydrophobic"]*hp,6),
        "hbond_w": round(w["hbond"]*hb,6),
        "pair_e_formula": round(w["gauss1"]*g1+w["gauss2"]*g2+w["repulsion"]*rp
                                +w["hydrophobic"]*hp+w["hbond"]*hb, 6),
    }


def _load_pdbqt_all(path):
    """Load ALL atoms from PDBQT MODEL 1 (including H, for index alignment)."""
    atoms = []
    try:
        with open(path) as fh:
            in_model = has_model = False
            for line in fh:
                if line.startswith("MODEL"):  has_model = in_model = True; continue
                if line.startswith("ENDMDL"): break
                if not has_model: in_model = True
                if not in_model:  continue
                if line[:4] not in ("ATOM","HEAT"): continue
                try:
                    atoms.append({
                        "serial":  int(line[6:11]),
                        "name":    line[12:16].strip(),
                        "resname": line[17:20].strip(),
                        "resseq":  line[22:26].strip(),
                        "chain":   line[21].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "atype":   line[77:].strip(),
                    })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        logger.warning(f"_load_pdbqt_all({path}): {e}")
    return atoms


def _load_pdbqt_heavy(path):
    """Load heavy atoms only (H/HD excluded) — used for vina_check_file."""
    return [a for a in _load_pdbqt_all(path) if a["atype"] not in ("H","HD")]


def _load_rec_heavy(path):
    """Load receptor heavy atoms for protein view.

    CRITICAL: idx stored here must match the rec_atom index in the Vina log,
    which is the position in m.grid_atoms[] — i.e. the ALL-atom line index
    (H atoms included in the count, just skipped for scoring).
    """
    atoms = []
    all_atom_idx = 0   # counts every ATOM/HETATM line including H
    try:
        with open(path) as fh:
            for line in fh:
                if line[:4] not in ("ATOM","HEAT"): continue
                atype = line[77:].strip()
                is_h  = atype in ("HD", "H")
                try:
                    rec = {
                        "serial":      int(line[6:11]),
                        "name":        line[12:16].strip(),
                        "resname":     line[17:20].strip(),
                        "resseq":      line[22:26].strip(),
                        "chain":       line[21].strip(),
                        "x":           float(line[30:38]),
                        "y":           float(line[38:46]),
                        "z":           float(line[46:54]),
                        "atype":       atype,
                        "all_atom_idx": all_atom_idx,  # ← Vina's rec_atom index
                    }
                    if not is_h:
                        atoms.append(rec)
                    all_atom_idx += 1
                except (ValueError, IndexError):
                    all_atom_idx += 1
                    continue
    except Exception as e:
        logger.warning(f"_load_rec_heavy({path}): {e}")
    return atoms


def _build_response(lig_atoms_raw, total_e, lig_order, mode_energies,
                    lig_path, rec_path, log_text):
    """Shared JSON builder for vina_viz and vina_dock."""
    lig_atoms_all = _load_pdbqt_all(lig_path)   if lig_path  else []
    rec_atoms_all = _load_rec_heavy(rec_path)    if rec_path  else []

    # Build lookup: all_atom_idx → atom record (matches m.grid_atoms[] indexing)
    rec_by_idx = {a['all_atom_idx']: a for a in rec_atoms_all}

    result_atoms = []
    for li in lig_order:
        info  = lig_atoms_raw[li]
        xs    = info["xs"]
        pairs = info["pairs"]

        pairs_sorted = sorted(pairs, key=lambda p: abs(p["pair_e"]), reverse=True)
        top_pairs = []
        for p in pairs_sorted[:15]:
            terms    = _term_breakdown(xs, p["rec_xs"], p["r"])
            rec_atom = rec_by_idx.get(p["rec_idx"])
            top_pairs.append({
                "rec_idx":      p["rec_idx"],
                "rec_xs":       p["rec_xs"],
                "rec_xs_label": _XS_META.get(p["rec_xs"],("?",))[0],
                "rec_atom":     rec_atom,
                "opt":  p["opt"], "r": p["r"], "s": p["s"],
                "pair_e": p["pair_e"], "terms": terms,
            })

        lig_atom = lig_atoms_all[li] if li < len(lig_atoms_all) else None
        result_atoms.append({
            "lig_idx":       li,
            "xs":            xs,
            "xs_label":      _XS_META.get(xs,("?",))[0],
            "this_e":        info["this_e"],
            "out_of_bounds": info["out_of_bounds"],
            "n_pairs":       len(pairs),
            "pair_sum":      round(sum(p["pair_e"] for p in pairs), 6),
            "top_pairs":     top_pairs,
            "lig_atom":      lig_atom,
        })

    result_atoms.sort(key=lambda a: a["this_e"])

    return jsonify({
        "status":        "success",
        "total_e":       total_e,
        "n_lig_atoms":   len(result_atoms),
        "lig_atoms":     result_atoms,
        "mode_energies": mode_energies,
        "log_preview":   log_text[:2000],
    })


@bp.route('/vina_viz', methods=['POST'])
def vina_viz():
    """
    POST /vina_viz
    Body: { "log_text": str, "ligand_path": str (opt), "receptor_path": str (opt) }
    Parse a captured non_cache::eval log and return structured JSON.
    """
    try:
        data     = request.get_json(force=True) or {}
        log_text = data.get("log_text","").strip()
        lig_path = data.get("ligand_path","").strip()
        rec_path = data.get("receptor_path","").strip()

        if not log_text:
            return jsonify({"status":"error","message":"log_text is required"}), 400

        lig_atoms_raw, total_e, lig_order, mode_energies = _parse_vina_log(log_text)

        if not lig_atoms_raw:
            return jsonify({"status":"error",
                            "message":"No [non_cache::eval pair] lines found in log_text"}), 400

        return _build_response(lig_atoms_raw, total_e, lig_order, mode_energies,
                               lig_path, rec_path, log_text)
    except Exception as exc:
        logger.error(f"vina_viz error: {exc}", exc_info=True)
        return jsonify({"status":"error","message":str(exc)}), 500


@bp.route('/vina_check_file', methods=['POST'])
def vina_check_file():
    """Check if a PDBQT file exists and count its heavy atoms."""
    try:
        data  = request.get_json(force=True) or {}
        path  = data.get('path','').strip()
        if not path or not Path(path).is_file():
            return jsonify({'exists': False, 'n_atoms': 0})
        atoms = _load_pdbqt_heavy(path)
        return jsonify({'exists': True, 'n_atoms': len(atoms)})
    except Exception as exc:
        return jsonify({'exists': False, 'n_atoms': 0, 'error': str(exc)})


@bp.route('/vina_dock', methods=['POST'])
def vina_dock():
    """
    POST /vina_dock
    Runs AutoDock Vina, streams stdout progress via SSE, writes a log file,
    parses the non_cache::eval output, and returns structured score data.

    Body: { "receptor_path": str, "ligand_path": str }
    """
    import subprocess, re, datetime
    try:
        data     = request.get_json(force=True) or {}
        rec_path = data.get('receptor_path', '').strip()
        lig_path = data.get('ligand_path',   '').strip()

        for label, p in [('receptor_path', rec_path), ('ligand_path', lig_path)]:
            if not p:
                return jsonify({'status': 'error', 'message': f'{label} is required'}), 400
            if not Path(p).is_file():
                return jsonify({'status': 'error', 'message': f'File not found: {p}'}), 404

        LOG_FILE  = VINA_LOG

        # Output alongside the ligand file, named after it
        lig_stem  = Path(lig_path).stem
        OUT_LIG   = str(Path(lig_path).parent / f"{lig_stem}_out.pdbqt")

        cmd = [
            VINA_BIN,
            '--receptor',     rec_path,
            '--ligand',       lig_path,
            '--center_x',     '-25.7',
            '--center_y',     '0.22',
            '--center_z',     '28.39',
            '--size_x',       '20',
            '--size_y',       '20',
            '--size_z',       '20',
            '--exhaustiveness', '8',
            '--cpu',          str(os.cpu_count() or 4),
            '--out',          OUT_LIG,
        ]

        logger.info(f'vina_dock: running {" ".join(cmd)}')

        # ── Run with real-time line capture ──────────────────────────────────
        # Clear the progress queue from any previous run
        while not _vina_progress_q.empty():
            try: _vina_progress_q.get_nowait()
            except: pass

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0,   # unbuffered — essential for live progress bar
        )

        lines = []
        import threading as _threading

        def _stream_stdout():
            """
            Read stdout byte-by-byte.
            Bar chars (* | -) are batched: a background flush thread emits
            the accumulated bar buffer every 150ms so the browser isn't
            flooded with thousands of individual SSE events.
            Regular text lines are pushed whole on newline.
            """
            import time as _time
            import threading as _t2

            buf      = b""
            bar_buf  = {"stars": "", "sep": ""}   # shared mutable dict
            bar_lock = _t2.Lock()
            done_ev  = _t2.Event()
            in_progress_bar = False   # True only after first * — guards |/- capture

            BAR_CHARS = {b'*', b'|', b'-'}

            def _flush_bar():
                """Push accumulated bar chars every 150ms."""
                while not done_ev.is_set():
                    _time.sleep(0.15)
                    with bar_lock:
                        if bar_buf["stars"]:
                            _vina_progress_q.put("__BARCH__stars:" + bar_buf["stars"])
                        if bar_buf["sep"]:
                            _vina_progress_q.put("__BARCH__sep:" + bar_buf["sep"])

            flush_t = _t2.Thread(target=_flush_bar, daemon=True)
            flush_t.start()

            with open(LOG_FILE, 'wb') as lf:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                header = f'# Vina dock run {ts}\n# CMD: {" ".join(cmd)}\n\n'.encode()
                lf.write(header)
                while True:
                    ch = proc.stdout.read(1)
                    if not ch:
                        break
                    lf.write(ch)
                    lf.flush()

                    if ch in (b'\n', b'\r'):
                        line = buf.decode('utf-8', errors='replace').strip()
                        if line:
                            lines.append(line + '\n')
                            _vina_progress_q.put(line)
                        buf = b""
                        in_progress_bar = False  # reset bar context on newline

                    elif ch == b'*':
                        in_progress_bar = True
                        with bar_lock:
                            bar_buf["stars"] += "*"
                        buf = b""   # reset text buf — bar started

                    elif ch in (b'|', b'-') and in_progress_bar:
                        # Only capture as bar chars when inside the progress bar
                        with bar_lock:
                            bar_buf["sep"] += ch.decode()
                        buf = b""

                    else:
                        buf += ch

                # Flush remaining text
                if buf:
                    line = buf.decode('utf-8', errors='replace').strip()
                    if line:
                        lines.append(line + '\n')
                        _vina_progress_q.put(line)

            # Stop flush thread and do final push
            done_ev.set()
            flush_t.join(timeout=0.5)
            with bar_lock:
                if bar_buf["stars"]:
                    _vina_progress_q.put("__BARCH__stars:" + bar_buf["stars"])
                if bar_buf["sep"]:
                    _vina_progress_q.put("__BARCH__sep:" + bar_buf["sep"])

        t = _threading.Thread(target=_stream_stdout, daemon=True)
        t.start()
        t.join()
        proc.wait()
        _vina_progress_q.put('__DONE__')

        log_text = ''.join(lines)
        logger.info(f'vina_dock: finished. Log written to {LOG_FILE} ({len(lines)} lines)')

        # ── Read the actual log file for clean extraction ─────────────────────
        # The streamed `lines` buffer strips bar chars (|/-/*) and may miss content.
        # Read the file directly to get the complete, unmodified output.
        try:
            with open(LOG_FILE, 'r', errors='replace') as _lf:
                full_log = _lf.read()
        except Exception:
            full_log = log_text   # fallback to streamed buffer

        # ── Extract the tail block: energy assembly + mode table ──────────────
        # The user-facing content starts from [num_tors_div...] or [mode 1 energy...]
        # and ends at the last line of the mode table. Read from file tail.
        tail_block = None
        tail_lines = full_log.splitlines()

        # Find the start of the energy assembly / tors block (last occurrence)
        start_idx = None
        for i in range(len(tail_lines) - 1, -1, -1):
            if '[num_tors_div' in tail_lines[i] or '[mode 1 energy' in tail_lines[i]:
                start_idx = i
                break

        if start_idx is not None:
            tail_block = '\n'.join(tail_lines[start_idx:]).strip()

        # ── Extract mode table ────────────────────────────────────────────────
        mode_table = None
        tbl_lines  = []
        in_table   = False
        for raw_line in tail_lines:
            if re.search(r'mode\s+\|\s+affinity', raw_line):
                in_table = True
                tbl_lines = [raw_line]
                continue
            if in_table:
                if raw_line.strip() == '' and tbl_lines:
                    break
                tbl_lines.append(raw_line)
        if tbl_lines:
            mode_table = '\n'.join(tbl_lines)

        # ── Extract best affinity from the mode table ─────────────────────────
        best_affinity = None
        for raw_line in tail_lines:
            m = re.search(r'^\s*1\s+([-\d.]+)', raw_line)
            if m:
                best_affinity = float(m.group(1))
                break

        # ── Count non_cache pair lines (for UI feedback) ──────────────────────
        n_pair_lines = sum(1 for l in lines if '[non_cache::eval pair]' in l)
        logger.info(f'vina_dock: {n_pair_lines} pair lines, best_affinity={best_affinity}')

        # ── Parse non_cache::eval log ─────────────────────────────────────────
        lig_atoms_raw, total_e, lig_order, mode_energies = _parse_vina_log(log_text)

        if not lig_atoms_raw:
            return jsonify({
                'status':        'success',
                'best_affinity': best_affinity,
                'mode_table':    mode_table,
                'tail_block':    tail_block,
                'log_file':      LOG_FILE,
                'n_pair_lines':  n_pair_lines,
                'total_e':       None,
                'lig_atoms':     None,
                'message':       (
                    f'Docking complete (best={best_affinity} kcal/mol). '
                    f'Log saved to {LOG_FILE}.'
                ),
            })

        # ── Build full response ───────────────────────────────────────────────
        resp      = _build_response(lig_atoms_raw, total_e, lig_order, mode_energies,
                                    OUT_LIG, rec_path, log_text)
        resp_data = resp.get_json()
        resp_data['best_affinity'] = best_affinity
        resp_data['log_file']      = LOG_FILE
        resp_data['n_pair_lines']  = n_pair_lines
        resp_data['mode_table']    = mode_table
        resp_data['tail_block']    = tail_block
        # Return only the compact summary lines — skip per-pair term breakdowns
        # (gauss/repulsion/hbond/hydrophobic) which flood the console.
        _SKIP = ('[gauss ', '[repulsion ', '[hydrophobic ', '[hbond ',
                 '[non_cache::eval pair]')
        nc_lines = [l for l in lines if not any(l.lstrip().startswith(p) for p in _SKIP)]
        resp_data['log_text'] = ''.join(nc_lines)
        return jsonify(resp_data)

    except Exception as exc:
        logger.error(f'vina_dock error: {exc}', exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@bp.route('/vina_dock_progress', methods=['GET'])
def vina_dock_progress_ep():
    """
    GET /vina_dock_progress
    SSE stream reading from _vina_progress_q — populated byte-by-byte from
    Vina's live stdout. Captures the *** progress bar in real time.
    """
    import re as _re2

    _SKIP = (
        # vLLM verbose energy terms
        '[gauss ', '[repulsion ', '[hydrophobic ', '[hbond ',
        '[non_cache', '[eval_interacting', '[pair]', '[num_tors',
        'after_curl=', 'running_e=',
        # Energy assembly lines
        'lig_grids ', 'inter_pairs ', 'inter = ', 'flex_grids ',
        'intra_pairs ', 'lig_intra ', 'intra = ', 'intramolecular',
        'e_in = ', 'conf_independent', 'total (E_vina)',
        'x (interaction', 'num_tors ', 'raw weight ',
        'weight = ', 'denom = ', 'result = ',
        # Per-atom coordinate dump lines
        'rec_in_box=', 'clipped=', 'oob_penalty=',
        # Curl accumulator lines
        '(before curl', '(= lig_grids',
        # RMSD table separator
        'dist from best', 'rmsd u.b.', 'rmsd l.b.',
        # Optimizer iteration lines e.g. "15.575(Δ=0.008Å)]"
        '(Δ=',
        # Energy assembly noise
        '(raw+1)', 'num_tors/5', '[mode ',
    )

    _SEP_RE = re.compile(r'^[-|]{6,}')   # Vina grid separator line

    def generate():
        idle = 0
        while idle < 60:
            try:
                item = _vina_progress_q.get(timeout=0.5)
                idle = 0
                stripped = item.strip()
                if item == "__DONE__":
                    yield "data: __DONE__\n\n"
                    return
                # Batched bar update: __BARCH__stars:<all_stars> or __BARCH__sep:<sep_chars>
                if item.startswith("__BARCH__"):
                    yield "data: " + item + "\n\n"
                    continue

                # Percentage line: "0%  10  20 ... 100%"
                import re as _re3
                _PCT_RE = _re3.compile(r"^\d+%")
                if _PCT_RE.match(stripped):
                    yield "data: __BAR__pct" + stripped + "\n\n"
                    continue
                # Skip if any SKIP pattern appears anywhere in the line
                # (coordinate lines start with numbers, not the pattern)
                if any(p in stripped for p in _SKIP):
                    continue
                # Skip Vina grid separator lines (long ----||||--- sequences)
                if _SEP_RE.match(stripped):
                    continue
                # Skip bare float-only lines (coordinate accumulator values)
                import re as _re_skip
                if _re_skip.match(r'^-?\d+\.\d+$', stripped):
                    continue
                # Skip lines that are ONLY numbers/spaces (partial coordinate lines)
                # BUT keep result table rows like "1  -9.3082  0.0000  0.0000"
                # Result rows have mode number + negative affinity — recognise by pattern
                is_result_row = bool(_re_skip.match(r'^\d+\s+-\d+\.\d', stripped))
                if (not is_result_row and stripped
                        and all(c in '0123456789.-,() ' for c in stripped)
                        and len(stripped) < 40):
                    continue
                # Restore mode result table lines (keep "mode |" header and "N  -X.X  ..." rows)
                is_mode_table = (stripped.startswith('mode |') or
                                 bool(_re_skip.match(r'^\d+\s+-\d+\.', stripped)))
                if "[non_cache::eval pair]" in stripped:
                    m = _re2.search(
                        r"lig_atom=(\d+).*?pair_e=([-\d.]+).*?rec_in_box=(\w+)",
                        stripped)
                    if m:
                        item = ("[pair] atom=" + m.group(1) + "  "
                                "e=" + f"{float(m.group(2)):+.5f}" + "  "
                                "in_box=" + m.group(3))
                    else:
                        item = stripped[:120]
                safe = item.replace("\n", " ").replace("\r", "")
                if safe.strip():
                    yield "data: " + safe + "\n\n"
            except _queue_module.Empty:
                idle += 0.5
                yield ": ping\n\n"
            except Exception as exc:
                yield "data: Error: " + str(exc) + "\n\n"
                break
        yield "data: __DONE__\n\n"
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@bp.route('/vina_parse_log', methods=['POST'])
def vina_parse_log():
    """
    POST /vina_parse_log
    Reads vina_non_cache.log, finds best-pose [non_cache::eval pair] lines,
    builds a payload in the SAME shape as adj_3d_viz (atoms, bonds, weight_vector)
    so the frontend can call _render3D, _renderBar, _renderTable unchanged.

    Body: { "receptor_path": str (opt), "ligand_path": str (opt) }

    weight_vector.values[i]  = this_e for atom idx i  (sum of pair_e)
    weight_vector.top_indices = top 50 by |this_e|
    atoms[i].weight_raw      = this_e (raw kcal/mol)
    atoms[i].weight_norm     = normalised 0..1 for colour/size
    """
    import re as _re
    LOG_FILE = VINA_LOG
    try:
        data     = request.get_json(force=True) or {}
        lig_path = data.get('ligand_path',   '').strip()
        rec_path = data.get('receptor_path', '').strip()

        if not Path(LOG_FILE).is_file():
            return jsonify({'status': 'error',
                            'message': f'Log file not found: {LOG_FILE}'}), 404

        with open(LOG_FILE) as fh:
            log_text = fh.read()

        # ── Parse: extract ALL non_cache pairs, split by mode ────────────────
        # Strategy: find each "[non_cache::eval lig_atom=N]" block for mode 1
        # (the best pose is the FIRST complete block, ending at [non_cache::eval TOTAL])

        pair_re = _re.compile(
            r'\[non_cache::eval pair\]\s+lig_atom=(\d+)\(xs=(\d+)\)\s+'
            r'rec_atom=(\d+)\(xs=(\d+)\)\s+opt\(ri\+rj\)=([\d.]+)\s+'
            r'r=([\d.]+)\s+'
            r'(?:r_true=[\d.]+\s+)?'
            r's=r-opt=([-\d.]+)\s+pair_e=([-\d.]+)'
        )
        atom_re = _re.compile(
            r'\[non_cache::eval lig_atom=(\d+)\(xs=(\d+)\)\]\s+'
            r'this_e\(after curl\)=([-\d.]+)\s+out_of_bounds_penalty=([-\d.]+)'
        )
        total_re = _re.compile(r'\[non_cache::eval TOTAL\] e=([-\d.]+)')
        mode_re  = _re.compile(r'\[mode (\d+) energy assembly\]')
        aff_re   = _re.compile(r'^\s*1\s+([-\d.]+)', _re.MULTILINE)

        # Split log into sections by [mode N] markers
        # Best pose = first complete non_cache block (before or within mode 1)
        lines = log_text.splitlines()

        # Find the FIRST occurrence of [non_cache::eval TOTAL]
        # and take all pair/atom lines before it
        best_pairs:  dict[int, dict] = {}
        best_order:  list[int]       = []
        best_total:  float | None    = None
        best_mode:   int             = 1

        collecting = False
        for line in lines:
            # Start collecting when we see the first pair line
            m = pair_re.search(line)
            if m:
                collecting = True
                li  = int(m.group(1))
                lxs = int(m.group(2))
                ri  = int(m.group(3))
                rxs = int(m.group(4))
                opt,r,s,pe = float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8))
                if li not in best_pairs:
                    best_pairs[li] = {'xs':lxs,'pairs':[],'this_e':0.0}
                    best_order.append(li)
                best_pairs[li]['pairs'].append({'rec_idx':ri,'rec_xs':rxs,'opt':opt,'r':r,'s':s,'pair_e':pe})
                continue

            if collecting:
                m = atom_re.search(line)
                if m:
                    li = int(m.group(1))
                    if li not in best_pairs:
                        best_pairs[li] = {'xs':int(m.group(2)),'pairs':[],'this_e':0.0}
                        best_order.append(li)
                    best_pairs[li]['this_e'] = float(m.group(3))
                    continue

                m = total_re.search(line)
                if m:
                    best_total = float(m.group(1))
                    break    # stop after first complete block

        if not best_pairs:
            return jsonify({'status': 'error',
                            'message': 'No [non_cache::eval pair] lines found in log. '
                                       'Make sure the instrumented Vina binary was used.'}), 400

        # Best affinity from mode table
        aff_m = aff_re.search(log_text)
        best_affinity = float(aff_m.group(1)) if aff_m else None

        # ── Load ligand PDBQT for 3D coords + bonds ───────────────────────────
        lig_atoms_all = _load_pdbqt_all(lig_path) if lig_path else []
        rec_atoms_all = _load_rec_heavy(rec_path)  if rec_path  else []

        # Build this_e per atom idx
        # best_pairs keys = lig_atom indices in non_cache (= PDBQT line order incl H)
        this_e_map: dict[int, float] = {li: info['this_e'] for li, info in best_pairs.items()}

        # Map to HEAVY atoms only (matching PDBQT ordering)
        # non_cache skips H (t1 >= n), so this_e only exists for heavy atoms
        # lig_atoms_all[i].atype gives the type; we need i → this_e
        heavy_atoms_indexed = [
            (i, a) for i, a in enumerate(lig_atoms_all)
            if a['atype'] not in ('H', 'HD')
        ]

        # ── Compute this_e for each heavy atom (from log) ─────────────────────
        # Non_cache iterates ALL movable atoms (incl H) but H are skipped.
        # The lig_atom index in the log == i in VINA_FOR(i, m.num_movable_atoms())
        # which iterates lig_atoms_all in PDBQT order.
        # So this_e_map[i] directly corresponds to lig_atoms_all[i].

        # Build this_e_by_pdbqt_idx: for each heavy atom's pdbqt idx → this_e
        values_raw = []
        for pdbqt_i, atom in heavy_atoms_indexed:
            te = this_e_map.get(pdbqt_i, 0.0)
            values_raw.append(te)

        if not values_raw:
            return jsonify({'status': 'error', 'message': 'No heavy atoms found in ligand PDBQT'}), 400

        # Normalise: most negative (most favourable) → 1 (red = most important),
        # least negative / zero / positive → 0 (blue = least important).
        # Vina scores: lower is better, so we INVERT the usual direction.
        min_e = min(values_raw)
        max_e = max(values_raw)
        rng   = max_e - min_e if max_e != min_e else 1.0

        def _coolwarm_hex(t):
            t = max(0.0, min(1.0, t))
            if t < 0.5:
                s = t * 2
                r = int(59  + s * (221 - 59))
                g = int(76  + s * (220 - 76))
                b = int(192 + s * (220 - 192))
            else:
                s = (t - 0.5) * 2
                r = int(221 + s * (180 - 221))
                g = int(220 + s * (4   - 220))
                b = int(220 + s * (38  - 220))
            return f'#{r:02x}{g:02x}{b:02x}'

        # ── Build atoms list (weight_a shape) ─────────────────────────────────
        atoms_out = []
        symbols   = []
        values_normalised = []

        for local_i, (pdbqt_i, atom) in enumerate(heavy_atoms_indexed):
            te   = this_e_map.get(pdbqt_i, 0.0)
            # Invert: most negative this_e → norm=1 (most important/large/red),
            # near-zero or positive → norm=0 (least important/small/blue)
            norm = (max_e - te) / rng
            atoms_out.append({
                'idx':         local_i,          # sequential heavy-atom index
                'symbol':      atom['name'],      # e.g. "O1", "C6"
                'x':           round(atom['x'], 4),
                'y':           round(atom['y'], 4),
                'z':           round(atom['z'], 4),
                'weight_raw':  round(te, 5),      # this_e in kcal/mol
                'weight_norm': round(norm, 5),    # 0..1 for colour
                'color':       _coolwarm_hex(norm),
            })
            symbols.append(atom['name'])
            values_normalised.append(round(norm, 5))

        n_atoms = len(atoms_out)

        # top 50 by |this_e|
        top_50 = sorted(range(n_atoms), key=lambda i: abs(values_raw[i]), reverse=True)[:50]

        weight_vector = {
            'values':       [round(v, 5) for v in values_raw],
            'top_indices':  top_50,
            'atom_symbols': symbols,
            'n_atoms':      n_atoms,
        }

        # ── Build bonds (PDBQT has no bond info → infer from BRANCH records) ──
        # Simple fallback: no bonds (3D still looks good with atoms only)
        # For real bonds, parse BRANCH/ENDBRANCH + ROOT connectivity
        bonds = _infer_bonds_from_pdbqt(lig_path) if lig_path else []

        # ── pairs_by_lig_atom: local_heavy_idx → [{rec_idx, pair_e}] ────────────
        pairs_by_lig = {}
        for local_i, (pdbqt_i, _) in enumerate(heavy_atoms_indexed):
            if pdbqt_i in best_pairs:
                pairs_by_lig[local_i] = [
                    {'rec_idx': p['rec_idx'], 'pair_e': round(p['pair_e'], 5)}
                    for p in best_pairs[pdbqt_i]['pairs']
                ]

        # ── rec_atoms: receptor atom coords + labels for protein view ─────────
        rec_atoms_out = [
            {
                'idx':     a['all_atom_idx'],  # ← matches m.grid_atoms[] index in Vina log
                'name':    a['name'],
                'resname': a['resname'],
                'resseq':  a['resseq'],
                'x':       round(a['x'], 4),
                'y':       round(a['y'], 4),
                'z':       round(a['z'], 4),
                'atype':   a['atype'],
            }
            for a in rec_atoms_all
        ]

        return jsonify({
            'status':            'success',
            'atoms':             atoms_out,
            'bonds':             bonds,
            'weight_vector':     weight_vector,
            'total_e':           best_total,
            'best_affinity':     best_affinity,
            'best_mode':         best_mode,
            'n_lig_atoms':       n_atoms,
            'log_file':          LOG_FILE,
            'pairs_by_lig_atom': pairs_by_lig,
            'rec_atoms':         rec_atoms_out,
        })

    except Exception as exc:
        logger.error(f'vina_parse_log error: {exc}', exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


def _infer_bonds_from_pdbqt(path: str) -> list:
    """
    Infer bonds from PDBQT BRANCH/ENDBRANCH + ROOT structure.
    Returns list of {begin: local_heavy_idx, end: local_heavy_idx, order: 1}.
    Uses simple distance-based bond detection as fallback.
    """
    import math as _math
    atoms = [a for a in _load_pdbqt_all(path) if a['atype'] not in ('H','HD')]
    bonds = []
    n = len(atoms)
    # Distance threshold per element pair (simplified)
    for i in range(n):
        for j in range(i+1, n):
            dx = atoms[i]['x']-atoms[j]['x']
            dy = atoms[i]['y']-atoms[j]['y']
            dz = atoms[i]['z']-atoms[j]['z']
            d  = _math.sqrt(dx*dx+dy*dy+dz*dz)
            # Typical covalent bond lengths: C-C~1.54, C-N~1.47, C-O~1.43, C-F~1.35
            # Use generous 1.8 Å cutoff for all heavy-heavy bonds
            if d < 1.85:
                bonds.append({'begin': i, 'end': j, 'order': 1})
    return bonds


# ==============================================================================
# ── QWEN2.5-14B CHAT ROUTES ───────────────────────────────────────────────────
# Model runs in serve_qwen.sh on port 8001 — no weights loaded here.
# ==============================================================================


# ==============================================================================
# ── CoT UI Action Routing ─────────────────────────────────────────────────────
# ==============================================================================

# Dedicated CoT log
import logging as _logging
_cot_logger = _logging.getLogger("elion.vina.cot")

# cot_main.log — receives ALL CoT calls from routes.py (vina + attn routing)
# autolearn.log is written by kb_auto_learner._write_cot_log (feature-gap subset only)
_COT_MAIN_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".qwen", "cot_main.log"
)
os.makedirs(os.path.dirname(_COT_MAIN_LOG), exist_ok=True)
_cot_handler = _logging.FileHandler(_COT_MAIN_LOG)
_cot_handler.setFormatter(_logging.Formatter(
    "\n" + "=" * 60 + "\n"
    "[%(asctime)s]  source=routes\n"
    "%(message)s\n"
    + "─" * 60
))
_cot_logger.addHandler(_cot_handler)
_cot_logger.setLevel(_logging.DEBUG)
_cot_logger.propagate = False  # don't bubble up to root/autolearn.log

_vina_kb_cache: str | None = None

# Live stdout queue for vina_dock_progress SSE stream
import queue as _queue_module
_vina_progress_q: _queue_module.Queue = _queue_module.Queue()

def _load_vina_kb() -> str:
    global _vina_kb_cache
    if _vina_kb_cache is not None:
        return _vina_kb_cache
    try:
        with open(VINA_ACTION_KB_PATH, "r", encoding="utf-8") as f:
            _vina_kb_cache = f.read()
        logger.info("[CoT] vina_action_kb.md loaded (%d chars)", len(_vina_kb_cache))
    except Exception as e:
        logger.warning("[CoT] Could not load vina KB: %s", e)
        _vina_kb_cache = ""
    return _vina_kb_cache


def _keyword_route_action(user_message: str) -> dict | None:
    """
    RAG-powered UI action router for the Vina Docking visualizer.
    Delegates to ElionUIRouter; falls back gracefully if not available.
    """
    try:
        from elion_ui_router import route_ui_action as _rag_route
        return _rag_route(user_message, tool_hint="vina")
    except Exception as _e:
        logger.warning("[UIRouter] RAG unavailable for vina, using legacy keywords: %s", _e)

    # ── Legacy keyword fallback (kept for safety) ─────────────────────────────
    u = user_message.lower()
    if any(w in u for w in ["run docking", "start docking", "dock this", "perform docking",
                              "now what", "what now", "next step", "then", "and then"]):
        return {"action": "vina_dock_guided", "confidence": "high",
                "reason": "legacy: dock/progression keyword"}
    if any(w in u for w in ["load ligand", "ligand path", "ligand file"]):
        return {"action": "load_ligand", "confidence": "medium",
                "reason": "legacy: ligand keyword"}
    if any(w in u for w in ["load receptor", "receptor path", "protein file"]):
        return {"action": "load_receptor", "confidence": "medium",
                "reason": "legacy: receptor keyword"}
    if any(w in u for w in ["open chembert", "chembert", "attention"]):
        return {"action": "open_visualizer", "confidence": "medium",
                "reason": "legacy: chembert keyword"}
    return None


def _cot_route_action(user_message: str, coach_response: str, history: list) -> dict | None:
    """
    DeepSeek-V4-style two-stage CoT routing for Vina UI actions.
    Stage 1: Qwen thinks through intent using the KB.
    Stage 2: Parses JSON conclusion → ui_action.
    """
    import json, re as _re

    kb = _load_vina_kb()
    if not kb:
        return None

    history_summary = ""
    for role, text in history[-3:]:
        history_summary += f"  {role.upper()}: {text[:200].replace(chr(10), ' ')}\n"

    cot_prompt = (
        "<|im_start|>system\n"
        "You are Elion's UI routing agent for a molecular docking visualizer. "
        "Decide which UI action (if any) to trigger based on the conversation. "
        "Think step-by-step, then output one raw JSON line.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"## UI Action Knowledge Base\n{kb}\n\n"
        f"## Recent history\n{history_summary}\n"
        f"## User message\n{user_message}\n\n"
        f"## Elion response\n{coach_response[:400]}\n\n"
        "Think step-by-step about user intent. "
        "After your reasoning write exactly --- on its own line then raw JSON:\n"
        '{"action": "none", "confidence": "high", "reason": "..."}\n'
        "<|im_end|>\n"
        "<|im_start|>think\n"
    )

    try:
        out  = qwen_compat(cot_prompt, COT_ROUTING_PARAMS)
        text = out[0].outputs[0].text.strip()
        _cot_logger.info(
            "USER: %s\n\nPROMPT (tail 600):\n...%s\n\nRAW OUTPUT:\n%s",
            user_message, cot_prompt[-600:], text
        )
        logger.info("[CoT] vina routing output (first 400): %s", text[:400])

        matches = _re.findall(r'\{[^{}]+\}', text)
        if not matches:
            return None
        decision   = json.loads(matches[-1])
        action     = decision.get("action", "none")
        confidence = decision.get("confidence", "low")
        reason     = decision.get("reason", "")
        logger.info("[CoT] → action=%s confidence=%s reason=%s", action, confidence, reason)

        valid = {"open_visualizer","open_voxel_inspector","toggle_grid_box",
                 "run_visualization","run_docking","switch_ligand_view",
                 "switch_protein_view","load_ligand","load_receptor",
                 "vina_dock_guided","guide_pdb_conversion","none"}
        if action in valid and confidence in ("high", "medium") and action != "none":
            return {"type": "ui_action", "action": action,
                    "confidence": confidence, "reason": reason}
        return None
    except Exception as e:
        logger.warning("[CoT] vina routing failed: %s", e)
        return None

VINA_SYSTEM_PROMPT = """You are Elion, an expert AI assistant for computational drug discovery.
You are embedded in a Vina molecular docking visualizer. You help researchers understand:
- AutoDock Vina docking results and binding energies (kcal/mol)
- Molecular interactions between ligands and protein receptors
- SMILES notation, molecular properties, and drug-likeness
- ChemBERT attention scores and what they reveal about binding sites
- How to interpret 3D molecular visualizations and interaction energies
- Next steps in the drug discovery pipeline

Be concise, scientific, and practical. When given docking scores or SMILES, analyze them directly.
Negative binding energies (e.g. -8.5 kcal/mol) indicate stronger binding. Below -7 is considered
good, below -9 is very strong."""

VINA_GUIDED_PROMPT = (
    "You are Elion, a guided assistant inside the Elion Vina Visualizer.\n\n"
    "STRICT RULES — follow every one:\n"
    "1. Reply in MAXIMUM 1-2 short sentences. No markdown headers. No bullet lists. No step numbers.\n"
    "2. Give exactly ONE concrete UI action. Name the specific button or field.\n"
    "3. If the receptor and ligand paths are already filled in (visible in context), "
    "tell the user to verify the paths and click the Vina Dock button.\n"
    "4. Detect where the user is in the workflow from the conversation history, "
    "then give ONLY the single immediate next micro-step.\n"
    "5. NEVER explain concepts, list alternatives, or describe future steps.\n"
    "6. FIRST-TIME USER DETECTION: If the user mentions having a .pdb file (not .pdbqt), "
    "or asks how to get started, or seems unfamiliar with the workflow, "
    "tell them to use the PDB → PDBQT converter tool first via the Ask Elion menu. "
    "Say something like: \"Since you have a .pdb file, let me guide you to convert it — "
    "click the glowing Ask Elion button above to open the tool menu.\"\n"
    "7. POST-CONVERSION: If [Just converted] appears in context, the file was already saved "
    "and the path auto-filled. Tell the user which field still needs filling, then guide them "
    "to click Vina Dock. Do NOT ask them to convert again.\n\n"
    "Workflow order: 1→Convert .pdb to .pdbqt (if needed) 2→Load receptor 3→Load ligand "
    "4→Click Vina Dock 5→Click Visualize\n\n"
    "Good responses (copy this style):\n"
    '"If those two paths look correct, click the Vina Dock button to start docking."\n'
    '"Enter your ligand .pdbqt path in the LIGAND field, then click Load."\n'
    '"Click the Visualize button to render the per-atom energy decomposition."\n'
    '"Since you have a .pdb file, let me guide you to convert it — '
    'click the glowing Ask Elion button above to open the tool menu."\n'
)

_vina_chat_history = []   # in-memory session history


@bp.route('/chat', methods=['POST'])
def vina_chat_ep():
    """
    Non-streaming chat endpoint (fallback).
    POST { "message": str, "context": { smiles, score, mode_energies, ... } }
    """
    try:
        data    = request.get_json(force=True) or {}
        message = (data.get('message') or '').strip()
        context = data.get('context', {})
        if not message:
            return jsonify({"status": "error", "message": "Empty message"}), 400

        # Build context block from current visualization state
        ctx_lines = []
        if context.get('smiles'):
            ctx_lines.append(f"Current ligand SMILES: {context['smiles']}")
        if context.get('score') is not None:
            ctx_lines.append(f"Best docking score: {context['score']} kcal/mol")
        if context.get('mode_energies'):
            energies = context['mode_energies'][:5]
            ctx_lines.append(f"Top binding modes (kcal/mol): {', '.join(str(e) for e in energies)}")
        if context.get('ligand_path'):
            ctx_lines.append(f"Ligand file: {context['ligand_path']}")
        if context.get('receptor_path'):
            ctx_lines.append(f"Receptor file: {context['receptor_path']}")
        if context.get('n_atoms'):
            ctx_lines.append(f"Ligand heavy atoms: {context['n_atoms']}")

        ctx_block = ("\n[Current docking session context]\n" + "\n".join(ctx_lines)) if ctx_lines else ""

        # Build ChatML prompt with history
        parts = [f"<|im_start|>system\n{VINA_SYSTEM_PROMPT}<|im_end|>"]
        for role, content in _vina_chat_history[-6:]:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{ctx_block}\n\n{message}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)

        out      = qwen_compat(prompt, COACH_PARAMS)
        response = out[0].outputs[0].text.strip()

        _vina_chat_history.append(("user",      message))
        _vina_chat_history.append(("assistant", response))
        # Keep last 20 turns
        if len(_vina_chat_history) > 20:
            del _vina_chat_history[:2]

        # CoT UI action routing
        ui_action = _keyword_route_action(message)
        if ui_action is None:
            ui_action = _cot_route_action(message, response, _vina_chat_history[-6:])
        # Auto-learn: if still no match, use Qwen CoT to generate and persist new KB entry
        if ui_action is None:
            try:
                from kb_auto_learner import maybe_learn_and_route
                ui_action = maybe_learn_and_route(message, tool_hint="vina")
                if ui_action:
                    logger.info("[AutoLearn] vina learned: %s", ui_action.get("action"))
            except Exception as _ale:
                logger.warning("[AutoLearn] vina learner error: %s", _ale)
        logger.info("[UI] vina ui_action=%s", ui_action)

        return jsonify({"status": "success", "response": response, "ui_action": ui_action})

    except Exception as e:
        logger.error(f"Vina chat error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route('/chat/stream', methods=['POST'])
def vina_chat_stream_ep():
    """
    Streaming SSE chat endpoint.
    POST { "message": str, "context": { smiles, score, mode_energies, ... } }
    Streams: event: token  data: "chunk"
             event: done   data: {}
             event: error  data: "msg"
    """
    import json as _json

    # Read request data BEFORE entering generator (request context safety)
    _req = request.get_json(force=True) or {}
    _msg = (_req.get('message') or '').strip()
    _ctx = _req.get('context', {})

    def generate():
        try:
            if not _msg:
                yield "event: error\ndata: Empty message\n\n"
                return

            # ── Pre-detect emotion BEFORE prompt — relational rules fire THIS reply ──
            emotion = _detect_emotion(_msg)
            if emotion != "calm":
                _ctx["detected_emotion"] = emotion
            logger.info("[Relational] vina detected_emotion=%s msg=%r", emotion, _msg[:80])

            # Build context block
            ctx_lines = []
            if _ctx.get('smiles'):
                ctx_lines.append(f"Current ligand SMILES: {_ctx['smiles']}")
            if _ctx.get('score') is not None:
                ctx_lines.append(f"Best docking score: {_ctx['score']} kcal/mol")
            if _ctx.get('mode_energies'):
                energies = _ctx['mode_energies'][:5]
                ctx_lines.append(f"Top binding modes (kcal/mol): {', '.join(str(e) for e in energies)}")
            if _ctx.get('ligand_path'):
                ctx_lines.append(f"Ligand: {_ctx['ligand_path']}")
            else:
                ctx_lines.append("Ligand path: NOT SET")
            if _ctx.get('receptor_path'):
                ctx_lines.append(f"Receptor: {_ctx['receptor_path']}")
            else:
                ctx_lines.append("Receptor path: NOT SET")
            if _ctx.get('last_conversion'):
                lc = _ctx['last_conversion']
                ctx_lines.append(
                    f"[Just converted] {lc.get('filename','?')} → {lc.get('mol_type','?')} PDBQT "
                    f"saved at: {lc.get('output_path','?')} — path was auto-filled in the UI."
                )
            if _ctx.get('receptor_missing') and _ctx.get('last_conversion', {}).get('mol_type') == 'ligand':
                ctx_lines.append(
                    "[ACTION NEEDED] Ligand was just converted but Receptor path is still missing. "
                    "Ask the user if they also need to convert a receptor .pdb file, "
                    "and guide them to open the converter again via Ask Elion → PDB → PDBQT."
                )
            elif _ctx.get('ligand_missing') and _ctx.get('last_conversion', {}).get('mol_type') == 'receptor':
                ctx_lines.append(
                    "[ACTION NEEDED] Receptor was just converted but Ligand path is still missing. "
                    "Ask the user if they also need to convert a ligand .pdb file, "
                    "and guide them to open the converter again via Ask Elion → PDB → PDBQT."
                )
            if emotion != "calm":
                ctx_lines.append(
                    f"[User emotional state: {emotion} — "
                    + ("validate first, keep to ONE sentence." if emotion == "frustrated" else
                       "one idea only, plain language." if emotion == "overwhelmed" else
                       "clarify gently before next step." if emotion == "confused" else
                       "match energy briefly, then continue." if emotion == "excited" else "")
                    + "]"
                )

            ctx_block = ("\n[Current docking session]\n" + "\n".join(ctx_lines)) if ctx_lines else ""

            # Build ChatML prompt — use guided prompt in mini-chat mode
            sys_prompt = VINA_GUIDED_PROMPT if _ctx.get('guided_mode') else VINA_SYSTEM_PROMPT
            parts = [f"<|im_start|>system\n{sys_prompt}<|im_end|>"]
            for role, content in _vina_chat_history[-6:]:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append(f"<|im_start|>user\n{ctx_block}\n\n{_msg}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(parts)

            # Token budget scales with emotion and mode
            if _ctx.get('guided_mode'):
                _tok = 120
            elif emotion in ("frustrated", "overwhelmed"):
                _tok = 80
            else:
                _tok = 200
            stream_params = QwenParams(temperature=0.5, max_tokens=_tok, top_p=0.9)
            full_response = ""
            for chunk in qwen_stream(prompt, stream_params):
                full_response += chunk
                yield f"event: token\ndata: {_json.dumps(chunk)}\n\n"

            # Save to history
            _vina_chat_history.append(("user",      _msg))
            _vina_chat_history.append(("assistant", full_response.strip()))
            if len(_vina_chat_history) > 20:
                del _vina_chat_history[:2]

            # CoT UI action routing
            resp_lower = full_response.lower()
            if _ctx.get('guided_mode'):
                # First-time / PDB detection: user mentions .pdb (not .pdbqt) or is clearly new
                _u_lower = _msg.lower()
                _has_pdb_only = (
                    ('pdb' in _u_lower and 'pdbqt' not in _u_lower) or
                    any(w in _u_lower for w in ['i have pdb', 'i have a pdb', 'my pdb',
                                                 'only have pdb', 'got pdb', 'got a pdb',
                                                 'how do i start', 'where do i start',
                                                 'i\'m new', 'im new', 'first time',
                                                 'don\'t know', 'dont know', 'no idea'])
                )
                _resp_suggests_convert = any(w in resp_lower for w in [
                    'ask elion', 'convert it', 'pdb to pdbqt', 'pdb → pdbqt',
                    'convert your', 'converter', 'tool menu', 'glowing ask elion',
                    'convert a receptor', 'convert a ligand', 'convert the receptor',
                    'convert the ligand', 'open the converter', 'open converter'
                ])
                # Post-conversion: file was just saved, guide to fill remaining field + dock
                _is_post_conversion = bool(_ctx.get('last_conversion'))
                _resp_post_conv = any(w in resp_lower for w in [
                    'auto-filled', 'already saved', 'path was', 'now fill', 'fill in the',
                    'other field', 'fill the', 'fill your'
                ])
                if (_is_post_conversion and any(w in _u_lower for w in [
                    'now what', 'what now', 'next', 'what do i', 'proceed',
                    'ready', 'done', 'converted', 'what should'
                ])) or _resp_post_conv:
                    # Check if both paths are set → go straight to dock
                    _rec = _ctx.get('receptor_path', '').strip()
                    _lig = _ctx.get('ligand_path', '').strip()
                    if _rec and _lig:
                        ui_action = {"type": "ui_action", "action": "vina_dock_guided",
                                     "confidence": "high", "reason": "post-conversion: both paths set, dock now"}
                    else:
                        # Highlight whichever field is still empty
                        _missing = []
                        if not _rec: _missing += ['recPath', 'recLoadBtn']
                        if not _lig: _missing += ['ligPath', 'ligLoadBtn']
                        ui_action = {"type": "ui_action", "action": "load_receptor" if not _rec else "load_ligand",
                                     "confidence": "high", "reason": "post-conversion: fill remaining path"}
                elif _has_pdb_only or _resp_suggests_convert:
                    ui_action = {"type": "ui_action", "action": "guide_pdb_conversion",
                                 "confidence": "high", "reason": "guided: user has .pdb, needs conversion flow"}
                elif any(w in resp_lower for w in ['vina dock', 'click dock', 'click the vina', 'paths look correct',
                                                  'paths are correct', 'if those', 'if the paths', 'click vina dock']):
                    ui_action = {"type": "ui_action", "action": "vina_dock_guided", "confidence": "high", "reason": "guided: verify paths + dock"}
                elif any(w in resp_lower for w in ['click load', 'click the load', 'load button', 'ligand field', 'load the ligand']):
                    ui_action = {"type": "ui_action", "action": "load_ligand", "confidence": "high", "reason": "guided: load ligand"}
                elif any(w in resp_lower for w in ['click visualize', 'visualize button', 'click the visualize', 'click ⚡']):
                    ui_action = {"type": "ui_action", "action": "run_visualization", "confidence": "high", "reason": "guided: visualize"}
                elif any(w in resp_lower for w in ['receptor field', 'receptor path', 'load receptor']):
                    ui_action = {"type": "ui_action", "action": "load_receptor", "confidence": "high", "reason": "guided: load receptor"}
                else:
                    ui_action = _keyword_route_action(_msg)
            else:
                ui_action = _keyword_route_action(_msg)
                if ui_action is None:
                    ui_action = _cot_route_action(_msg, full_response, _vina_chat_history[-6:])
                # If CoT still returns none for "now what?" style questions,
                # and context has both paths → default to vina_dock_guided
                if ui_action is None:
                    _u = _msg.lower()
                    if any(w in _u for w in ["now what", "what now", "next step", "what next",
                                              "what do i", "what should", "proceed", "ready"]):
                        if _ctx.get("ligand_path") or _ctx.get("receptor_path"):
                            ui_action = {"type": "ui_action", "action": "vina_dock_guided",
                                         "confidence": "high", "reason": "paths set, next step is Vina Dock"}
            # Auto-learn: still no match → Qwen CoT generates + persists new KB entry
            if ui_action is None:
                try:
                    from kb_auto_learner import maybe_learn_and_route
                    ui_action = maybe_learn_and_route(_msg, tool_hint="vina")
                    if ui_action:
                        logger.info("[AutoLearn] vina stream learned: %s", ui_action.get("action"))
                except Exception as _ale:
                    logger.warning("[AutoLearn] vina stream error: %s", _ale)

            logger.info("[UI] vina stream ui_action=%s", ui_action)
            if ui_action:
                yield f"event: ui_action\ndata: {_json.dumps(ui_action)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            logger.error(f"Vina stream error: {e}")
            yield f"event: error\ndata: {_json.dumps(str(e))}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        }
    )


@bp.route('/chat/clear', methods=['POST'])
def vina_chat_clear_ep():
    """Clear in-memory chat history."""
    global _vina_chat_history
    _vina_chat_history = []
    return jsonify({"status": "success"})


# ==============================================================================
# PDB → PDBQT Converter
# ==============================================================================

def _pdb_to_pdbqt_rdkit(pdb_path: str, out_path: str, mol_type: str = "ligand") -> dict:
    """
    Convert a PDB file to PDBQT format.

    For ligands : uses RDKit — adds Hs, Gasteiger charges, AutoDock atom types.
    For receptors: uses direct PDB-line passthrough — preserves all original atom
                   names/residues/coordinates, computes Gasteiger charges via RDKit,
                   writes the exact PDBQT column layout Vina expects.
                   No ROOT/ENDROOT/TORSDOF — those are ligand-only.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdPartialCharges

    try:
        if mol_type == "receptor":
            return _pdb_to_pdbqt_receptor_passthrough(pdb_path, out_path)

        # ── Ligand path (RDKit full pipeline) ────────────────────────────────
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        if mol is None:
            return {"status": "error", "message": "RDKit could not parse the PDB file."}

        mol = Chem.AddHs(mol, addCoords=True)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)

        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            for atom in mol.GetAtoms():
                atom.SetDoubleProp("_GasteigerCharge", 0.0)

        _write_pdbqt_ligand(mol, out_path)
        return {"status": "success", "message": "Converted successfully (ligand).",
                "output_path": out_path}

    except Exception as exc:
        logger.exception("pdb_to_pdbqt_rdkit failed")
        return {"status": "error", "message": str(exc)}


def _pdb_to_pdbqt_receptor_passthrough(pdb_path: str, out_path: str) -> dict:
    """
    Receptor PDB → PDBQT by direct line passthrough.

    Strategy:
    1. Parse ATOM/HETATM lines from the PDB preserving original text (atom names,
       residue names, chain, sequence numbers, coordinates, B-factor).
    2. Load the same PDB via RDKit to compute per-atom Gasteiger charges.
       Map charges by atom serial number.
    3. Write output: same record/serial/name/resName/chain/resSeq/coords/occup/B
       but replace columns 66-76 with charge and 77-78 with AutoDock atom type.
    4. No ROOT/ENDROOT/TORSDOF — Vina treats those as errors in a receptor.
    """
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges
    import re

    # ── Step 1: read raw PDB lines ────────────────────────────────────────────
    with open(pdb_path) as fh:
        raw_lines = fh.readlines()

    atom_lines = [(i, l) for i, l in enumerate(raw_lines)
                  if l.startswith(('ATOM  ', 'HETATM'))]

    if not atom_lines:
        return {"status": "error", "message": "No ATOM/HETATM records found in PDB file."}

    # ── Step 2: compute Gasteiger charges via RDKit ───────────────────────────
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    charge_by_serial: dict[int, float] = {}
    if mol is not None:
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
            rdPartialCharges.ComputeGasteigerCharges(mol)
            for atom in mol.GetAtoms():
                ri = atom.GetPDBResidueInfo()
                if ri:
                    serial = ri.GetSerialNumber()
                    q = atom.GetDoubleProp("_GasteigerCharge") \
                        if atom.HasProp("_GasteigerCharge") else 0.0
                    charge_by_serial[serial] = 0.0 if (q != q) else q  # NaN → 0
        except Exception as e:
            logger.warning("Gasteiger charge computation failed: %s — using 0.0", e)

    # ── AutoDock atom type from element + residue context ────────────────────
    _AD_TYPES = {
        'C':  'C',  'N':  'NA', 'O':  'OA', 'S':  'SA',
        'H':  'HD', 'P':  'P',  'F':  'F',  'CL': 'Cl',
        'BR': 'Br', 'I':  'I',  'FE': 'Fe', 'ZN': 'Zn',
        'CA': 'Ca', 'MG': 'Mg', 'MN': 'Mn', 'CU': 'Cu',
    }
    def _ad_type(elem: str, atom_name: str) -> str:
        e = elem.upper().strip()
        # Distinguish polar H (HD) from non-polar H by whether the heavy atom name
        # suggests attachment to N/O/S — a rough but standard heuristic
        if e == 'H':
            return 'HD'
        return _AD_TYPES.get(e, e[:2] if len(e) > 1 else e)

    # ── Step 3: write PDBQT ───────────────────────────────────────────────────
    out_lines = []
    serial_re = re.compile(r'^\w{6}(.{5})')  # cols 6-10

    for _, line in atom_lines:
        # Ensure line is exactly padded to at least 80 cols
        line = line.rstrip('\n').rstrip('\r')
        line = line.ljust(80)

        # Parse serial (cols 6-10)
        try:
            serial = int(line[6:11].strip())
        except ValueError:
            serial = -1

        # Parse element from cols 76-77 (PDB standard) or infer from atom name
        elem = line[76:78].strip()
        if not elem:
            # Infer from atom name (cols 12-15): strip digits, take alpha chars
            aname = line[12:16].strip()
            elem  = re.sub(r'[^A-Za-z]', '', aname)
            elem  = elem[0].upper() + (elem[1:2].lower() if len(elem) > 1 else '')
            # Common 2-char elements
            if elem[:2].upper() in ('CL','BR','FE','ZN','CA','MG','MN','CU'):
                elem = elem[:2]
            else:
                elem = elem[0]

        atype  = _ad_type(elem, line[12:16].strip())
        charge = charge_by_serial.get(serial, 0.0)

        # Build output line: cols 0-65 verbatim, then charge (10 chars), space, atype (2 chars)
        base = line[:66]  # record..B-factor
        out_lines.append(f"{base}{charge:+9.3f} {atype:<2s}\n")

    out_lines.append("TER\n")
    Path(out_path).write_text("".join(out_lines))
    return {"status": "success", "message": "Converted successfully (receptor).",
            "output_path": out_path}


# AutoDock atom-type mapping from RDKit element symbol
_AUTODOCK_TYPES = {
    "C":  "C",  "N":  "NA", "O":  "OA", "S":  "SA",
    "H":  "H",  "P":  "P",  "F":  "F",  "Cl": "Cl",
    "Br": "Br", "I":  "I",  "Fe": "Fe", "Zn": "Zn",
    "Ca": "Ca", "Mg": "Mg", "Mn": "Mn",
}


def _autodock_type(atom) -> str:
    sym = atom.GetSymbol()
    # Carbons bonded only to C/H are non-polar (type A in some force fields; keep C for Vina)
    return _AUTODOCK_TYPES.get(sym, sym[:2] if len(sym) > 1 else sym)


def _write_pdbqt_ligand(mol, out_path: str):
    """Write a PDBQT file for a small-molecule ligand."""
    conf  = mol.GetConformer()
    lines = ["ROOT\n"]
    for i, atom in enumerate(mol.GetAtoms()):
        pos     = conf.GetAtomPosition(i)
        charge  = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if charge != charge:          # NaN guard
            charge = 0.0
        atype   = _autodock_type(atom)
        elem    = atom.GetSymbol()
        name    = f"{elem}{i+1}"
        line = (
            f"ATOM  {i+1:5d} {name:<4s} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {charge:+8.3f} {atype}\n"
        )
        lines.append(line)
    lines.append("ENDROOT\n")
    lines.append("TORSDOF 0\n")
    Path(out_path).write_text("".join(lines))


def _write_pdbqt_receptor(mol, out_path: str):
    """
    Write a PDBQT file for a rigid receptor.
    Receptors must use plain ATOM/HETATM records — no ROOT/ENDROOT/TORSDOF tags
    (those are ligand-only; Vina will reject the receptor with a parsing error otherwise).
    Hydrogens are stripped; partial charges set to 0.0.
    """
    conf  = mol.GetConformer()
    lines = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:   # skip explicit H (already removed by RemoveHs, but guard anyway)
            continue
        pos    = conf.GetAtomPosition(i)
        atype  = _autodock_type(atom)
        elem   = atom.GetSymbol()
        res    = atom.GetPDBResidueInfo()
        # Use PDB residue info when available (preserves original atom names / residue numbering)
        if res:
            aname  = res.GetName().strip() or elem
            resn   = res.GetResidueName().strip() or "UNK"
            resi   = res.GetResidueNumber()
            chain  = res.GetChainId().strip() or "A"
            ins    = res.GetInsertionCode().strip() or " "
        else:
            aname  = f"{elem}{i+1}"
            resn   = "UNK"
            resi   = i + 1
            chain  = "A"
            ins    = " "
        std_res = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
                   "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
        record = "ATOM  " if resn in std_res else "HETATM"
        line = (
            f"{record}{i+1:5d} {aname:<4s} {resn:<3s} {chain}{resi:4d}{ins}   "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {0.0:+8.3f} {atype}\n"
        )
        lines.append(line)
    lines.append("END\n")
    Path(out_path).write_text("".join(lines))