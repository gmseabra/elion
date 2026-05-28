"""
shared.py — helpers imported by multiple route blueprints.
Keeps individual blueprint files focused on HTTP logic only.
"""

import os
import re
import logging
import threading

logger = logging.getLogger(__name__)

# ── Runtime state ─────────────────────────────────────────────────────────────
_attn_kb_cache:   str | None = None
_attn_chat_history: list = []
_vina_chat_history: list = []
_vina_progress_q = __import__('queue').Queue()
_finetune_jobs: dict = {}

# ── Qwen client (imported lazily to avoid import-time cost) ──────────────────
def _get_qwen():
    from nas_storage_app.qwen_client import (
        qwen_stream, qwen_compat, QwenParams, COACH_PARAMS, COT_ROUTING_PARAMS
    )
    return qwen_stream, qwen_compat, QwenParams, COACH_PARAMS, COT_ROUTING_PARAMS


# ── Knowledge base ────────────────────────────────────────────────────────────
def load_attn_kb() -> str:
    global _attn_kb_cache
    if _attn_kb_cache is not None:
        return _attn_kb_cache
    from elion_config import ATTN_ACTION_KB_PATH
    try:
        with open(ATTN_ACTION_KB_PATH, "r", encoding="utf-8") as f:
            _attn_kb_cache = f.read()
        logger.info("[CoT] attn_action_kb.md loaded (%d chars)", len(_attn_kb_cache))
    except Exception as e:
        logger.warning("[CoT] Could not load attn KB: %s", e)
        _attn_kb_cache = ""
    return _attn_kb_cache


# ── Emotion detector ─────────────────────────────────────────────────────────
def detect_emotion(msg: str) -> str:
    m = msg.lower()
    if any(s in m for s in ["too much","too long","stop","just tell me","doesn't work",
                              "not working","wrong","ugh","wtf","!!","???"]): return "frustrated"
    if any(s in m for s in ["too many","complex","complicated","step by step","simpler"]): return "overwhelmed"
    if any(s in m for s in ["what is","confused","unclear","how do i","explain","huh",
                              "don't know","what now","now what"]): return "confused"
    if m.endswith("!") and any(s in m for s in ["wow","great","works","perfect","thanks"]): return "excited"
    if any(s in m for s in ["wow","great","amazing","nice","cool","thanks","love it"]): return "excited"
    return "calm"


# ── Keyword router for attn ───────────────────────────────────────────────────
def keyword_route_attn(user_message: str) -> dict | None:
    try:
        from elion_ui_router import route_ui_action as _rag_route
        return _rag_route(user_message, tool_hint="attn")
    except Exception:
        pass
    msg = user_message.lower()
    if any(w in msg for w in ["visualize","show me","draw","plot","display","render"]):
        return {"type":"ui_action","action":"open_visualizer","confidence":"high","reason":"keyword: visualize"}
    if any(w in msg for w in ["compare","comparison","vs","versus","side by side"]):
        return {"type":"ui_action","action":"compare_molecules","confidence":"high","reason":"keyword: compare"}
    if any(w in msg for w in ["fine-tune","finetune","fine tune","train","retrain"]):
        return {"type":"ui_action","action":"fine_tune_model","confidence":"high","reason":"keyword: finetune"}
    if any(w in msg for w in ["load model","switch model","change model","pretrained","finetuned"]):
        return {"type":"ui_action","action":"load_model","confidence":"high","reason":"keyword: load model"}
    if any(w in msg for w in ["vina","docking","dock","binding","affinity"]):
        return {"type":"ui_action","action":"open_vina","confidence":"high","reason":"keyword: vina"}
    return None


# ── Keyword router for vina ───────────────────────────────────────────────────
def keyword_route_vina(msg: str) -> dict | None:
    m = msg.lower()
    if 'pdb' in m and 'pdbqt' not in m:
        return {"type":"ui_action","action":"guide_pdb_conversion","confidence":"high","reason":"user has .pdb"}
    if any(w in m for w in ['visualize','show me','display','render','plot']):
        return {"type":"ui_action","action":"run_visualization","confidence":"high","reason":"keyword: visualize"}
    if any(w in m for w in ['vina dock','start dock','run dock','click dock']):
        return {"type":"ui_action","action":"run_docking","confidence":"high","reason":"keyword: dock"}
    if any(w in m for w in ['grid','box','grid box']):
        return {"type":"ui_action","action":"toggle_grid_box","confidence":"high","reason":"keyword: grid"}
    if any(w in m for w in ['voxel','inspect']):
        return {"type":"ui_action","action":"open_voxel_inspector","confidence":"high","reason":"keyword: voxel"}
    return None


# ── Smart model loader (ChemBERT) ─────────────────────────────────────────────
def load_chembert_model(checkpoint_path: str):
    """Load a ChemBERT checkpoint, auto-detecting pretrained vs finetuned format."""
    import torch
    from elion_config import CHEMBERT_BASE
    from nas_storage_app.CHEMBERT.model   import Smiles_BERT, BERT_base
    from nas_storage_app.CHEMBERT.chembert import Vocab

    vocab   = Vocab()
    n_vocab = len(vocab)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    first_key  = next(iter(state_dict))

    if first_key.startswith("bert."):
        # Finetuned: BERT_base wrapper
        smiles_bert = Smiles_BERT(n_vocab, max_len=256)
        model = BERT_base(smiles_bert, nn_size=1)
        model.load_state_dict(state_dict)
    else:
        # Pretrained: bare Smiles_BERT — wrap in BERT_base
        smiles_bert = Smiles_BERT(n_vocab, max_len=256)
        smiles_bert.load_state_dict(state_dict)
        model = __import__('nas_storage_app.CHEMBERT.model', fromlist=['BERT_base']).BERT_base(smiles_bert, nn_size=1)

    model.eval()
    return model, vocab


# ── PDBQT helpers ─────────────────────────────────────────────────────────────
_AUTODOCK_TYPES = {
    'C':'C','N':'NA','O':'OA','S':'SA','H':'HD','P':'P','F':'F',
    'CL':'Cl','BR':'Br','I':'I','FE':'Fe','ZN':'Zn','CA':'Ca','MG':'Mg','MN':'Mn','CU':'Cu',
}

def autodock_type(elem: str) -> str:
    return _AUTODOCK_TYPES.get(elem.upper().strip(), elem[:2] if len(elem) > 1 else elem)


def write_pdbqt_ligand(mol, out_path: str):
    from rdkit.Chem import rdPartialCharges
    conf  = mol.GetConformer()
    lines = ["ROOT\n"]
    for i, atom in enumerate(mol.GetAtoms()):
        pos    = conf.GetAtomPosition(i)
        charge = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if charge != charge: charge = 0.0
        atype  = autodock_type(atom.GetSymbol())
        elem   = atom.GetSymbol()
        name   = f"{elem}{i+1}"
        lines.append(
            f"ATOM  {i+1:5d} {name:<4s} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {charge:+8.3f} {atype}\n"
        )
    lines += ["ENDROOT\n", "TORSDOF 0\n"]
    from pathlib import Path
    Path(out_path).write_text("".join(lines))


def pdb_to_pdbqt_receptor(pdb_path: str, out_path: str) -> dict:
    """Direct PDB line passthrough for receptor — preserves atom names, computes Gasteiger charges."""
    import re as _re
    from pathlib import Path
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges

    with open(pdb_path) as fh:
        raw_lines = fh.readlines()
    atom_lines = [(i, l) for i, l in enumerate(raw_lines) if l.startswith(('ATOM  ','HETATM'))]
    if not atom_lines:
        return {"status":"error","message":"No ATOM/HETATM records found in PDB file."}

    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    charge_by_serial: dict[int, float] = {}
    if mol is not None:
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
            rdPartialCharges.ComputeGasteigerCharges(mol)
            for atom in mol.GetAtoms():
                ri = atom.GetPDBResidueInfo()
                if ri:
                    q = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
                    charge_by_serial[ri.GetSerialNumber()] = 0.0 if (q!=q) else q
        except Exception as e:
            logger.warning("Gasteiger charge computation failed: %s — using 0.0", e)

    out_lines = []
    for _, line in atom_lines:
        line = line.rstrip('\n').rstrip('\r').ljust(80)
        try:    serial = int(line[6:11].strip())
        except: serial = -1
        elem = line[76:78].strip()
        if not elem:
            aname = line[12:16].strip()
            elem  = _re.sub(r'[^A-Za-z]', '', aname)
            elem  = elem[0].upper() + (elem[1:2].lower() if len(elem) > 1 else '')
            elem  = elem[:2] if elem[:2].upper() in ('CL','BR','FE','ZN','CA','MG','MN','CU') else elem[0]
        atype  = autodock_type(elem)
        charge = charge_by_serial.get(serial, 0.0)
        out_lines.append(f"{line[:66]}{charge:+9.3f} {atype:<2s}\n")
    out_lines.append("TER\n")
    Path(out_path).write_text("".join(out_lines))
    return {"status":"success","message":"Converted successfully (receptor).","output_path":out_path}


def pdb_to_pdbqt_ligand(pdb_path: str, out_path: str) -> dict:
    from pathlib import Path
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdPartialCharges

    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        return {"status":"error","message":"RDKit could not parse the PDB file."}
    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        for atom in mol.GetAtoms(): atom.SetDoubleProp("_GasteigerCharge", 0.0)
    write_pdbqt_ligand(mol, out_path)
    return {"status":"success","message":"Converted successfully (ligand).","output_path":out_path}