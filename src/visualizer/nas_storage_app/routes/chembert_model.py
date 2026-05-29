# =============================================================================
# routes/chembert_model.py
# ChemBERT model loader, AdjacencyWeightVisualizer (both attn + vina variants),
# and _get_viz() cache helpers.
# Imported by attn_routes.py and vina_chembert_routes.py.
# =============================================================================

import os, logging, torch, torch.nn as nn
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from flask import current_app

from nas_storage_app.routes.shared import (
    logger, DEFAULT_FINETUNED, DEFAULT_PRETRAINED, CHEMBERT_BASE,
    _VIZ, _ROOT, ATTN_ACTION_KB_PATH,
)

_attn_kb_cache: str | None = None
_attn_chat_history: list = []

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

ATTN_ACTION_KB_PATH = str(_ROOT / "Elion-AGI-Ecosystem" / "attention_visualization" /
                          "nas_storage_app" / ".qwen" / "attn_action_kb.md")
os.makedirs(os.path.dirname(ATTN_ACTION_KB_PATH), exist_ok=True)
_attn_kb_cache: str | None = None
_attn_chat_history: list = []

def _load_attn_kb() -> str:
    global _attn_kb_cache
    if _attn_kb_cache is not None:
        return _attn_kb_cache
    try:
        with open(ATTN_ACTION_KB_PATH, "r", encoding="utf-8") as f:
            _attn_kb_cache = f.read()
        logger.info("[CoT] attn_action_kb.md loaded (%d chars)", len(_attn_kb_cache))
    except Exception as e:
        logger.warning("[CoT] Could not load attn KB: %s", e)
        _attn_kb_cache = ""
    return _attn_kb_cache


def _detect_emotion(msg: str) -> str:
    """
    Fast rule-based emotional state detector — runs BEFORE the LLM call so the
    relational system prompt rules have concrete signal for THIS reply.
    Returns: calm | frustrated | confused | excited | overwhelmed
    """
    m = msg.lower()
    frustrated = ["too much","too long","you talk","stop","just tell me","again",
                  "still","why isn't","doesn't work","not working","wrong","no,",
                  "no!","ugh","wtf","terrible","useless","!!","???",
                  "you should","shouldn't"]
    overwhelmed = ["too many","so much","a lot","complex","complicated",
                   "step by step","slowly","simple","simpler","keep it short"]
    confused = ["what is","what's","i don't understand","confused","unclear",
                "what do i","how do i","what does","explain","huh",
                "i'm not sure","not sure","don't know","what now","now what"]
    excited = ["wow","great","amazing","nice","cool","works","worked",
               "got it","perfect","thanks","thank you","love it"]
    # Frustration beats others if co-present
    if any(s in m for s in frustrated):  return "frustrated"
    if any(s in m for s in overwhelmed): return "overwhelmed"
    if any(s in m for s in confused):    return "confused"
    # Excitement only if ends with ! and has positive word
    if m.endswith("!") and any(s in m for s in excited): return "excited"
    if any(s in m for s in excited):     return "excited"
    return "calm"


def _keyword_route_attn(user_message: str) -> dict | None:
    """
    RAG-powered UI action router for the ChemBERT / attention visualizer.
    Delegates to ElionUIRouter; falls back gracefully if not available.
    """
    try:
        from elion_ui_router import route_ui_action as _rag_route
        return _rag_route(user_message, tool_hint="attn")
    except Exception as _e:
        logger.warning("[UIRouter] RAG unavailable for attn, using legacy keywords: %s", _e)

    # ── Legacy keyword fallback (kept for safety) ─────────────────────────────
    u = user_message.lower()
    if any(w in u for w in ["what is vina", "vina docking", "autodock", "open vina"]):
        return {"action": "open_vina", "confidence": "high",
                "reason": "legacy: vina mention"}
    if any(w in u for w in ["chembert", "chem bert", "chem-bert", "attention weight"]):
        return {"action": "open_visualizer", "confidence": "high",
                "reason": "legacy: chembert mention"}
    if any(w in u for w in ["then", "now what", "next step", "show 3d", "visualize"]):
        return {"action": "show_3d", "confidence": "medium",
                "reason": "legacy: progression keyword"}
    if "compare" in u:
        return {"action": "compare_mode", "confidence": "medium",
                "reason": "legacy: compare keyword"}
    return None



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

VINA_ACTION_KB_PATH = str(_ROOT / "Elion-AGI-Ecosystem" / "vina_visualization" /
                          "nas_storage_app" / ".qwen" / "vina_action_kb.md")
os.makedirs(os.path.dirname(VINA_ACTION_KB_PATH), exist_ok=True)

DEFAULT_FINETUNED   = str(_ROOT / "Elion-AGI-Ecosystem" / "attention_visualization" /
                          "nas_storage_app" / "CHEMBERT" / "Finetuned_model_5.pt")
DEFAULT_PRETRAINED  = str(_ROOT / "Elion-AGI-Ecosystem" / "attention_visualization" /
                          "nas_storage_app" / "CHEMBERT" / "pretrained_model.pt")

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