#!/usr/bin/env python3
# =============================================================================
# yupu_GIGN_pose.py  --  score ONE (ligand pose, receptor) pair with GIGN
#
# Drop this file in the GIGN repo dir (next to GIGN.py / HIL.py / utils.py /
# dataset_GIGN_ZccE.py), i.e.:
#     /home/huangzihang/repos/elion/src/elion/properties/Yupu_GIGN/yupu_GIGN_pose.py
#
# Invoked by the Flask endpoint  POST /pose/gign_score  as:
#     python yupu_GIGN_pose.py --stage <dir> --name <name>
# run with cwd = this dir (so ./model and the local imports resolve) inside the
# GIGN conda env.
#
# Reads a staging dir prepared by the endpoint:
#     <stage>/<name>_ligand.pdb     posed ligand (HETATM, from the Pose tool)
#     <stage>/<name>_protein.pdb    the uploaded receptor (.pdb)
#     <stage>/<name>_meta.json      { smiles, pocket_cutoff, drop_water, model }
#
# Behaviour matches how the GIGN model was TRAINED (see preprocessing.py:
# pymol `remove resn HOH` + `remove hydrogens` + `byres <ligand> around 5`):
#   - ligand : Chem.MolFromPDBFile(removeHs=True). If SMILES is supplied, bond
#              orders are fixed from it (the pose ligand has no CONECT records).
#   - pocket : DEFAULT pocket_cutoff = 5 cuts the 5 A binding pocket -- whole
#              protein residues with any heavy atom within 5 A of any ligand heavy
#              atom, waters dropped (drop_water) -- reproducing the training prep.
#              Set pocket_cutoff <= 0 to instead feed the FULL receptor, which
#              reproduces predict_ZccE.py (a train/inference MISMATCH -- the model
#              was trained on pockets, so full-protein scores are not meaningful).
#   - graph  : same featurisation as dataset_GIGN_ZccE (mol2graph + inter_graph,
#              dis_threshold=5), batched, run through GIGN(35,256,3).
#
# Prints a parseable marker the endpoint reads:
#     GIGN_PRED_PK <value>          (pK = -logKd/Ki ; dG = -1.36 * pK)
# All debug chatter goes to stderr so stdout stays clean.
# =============================================================================
import os
import sys
import json
import argparse

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np
import torch
import networkx as nx
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from torch_geometric.data import Data, Batch

# local GIGN repo modules (cwd must be the repo dir)
from GIGN import GIGN
from dataset_GIGN_ZccE import mol2graph          # identical atom featurisation

# undo the module-wide np.set_printoptions(threshold=inf) that dataset_GIGN_ZccE sets
np.set_printoptions(threshold=50)

# load_model_dict from utils (confirmed: model.load_state_dict(torch.load(ckpt, weights_only=False)))
try:
    from utils import load_model_dict as _load_model_dict
except Exception:
    _load_model_dict = None

# default checkpoint (matches predict_ZccE.py); override via meta['model'] / pose.gign_model
DEFAULT_MODEL = './model/epoch-165, train_loss-0.2755, train_rmse-0.5249, valid_rmse-1.1359, valid_pr-0.8530.pt'
WATER_RESNAMES = {'HOH', 'WAT', 'DOD', 'H2O'}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _safe_mol_from_pdb(pdb):
    """Robustly load a PDB into an RDKit mol. Full sanitise first; on failure parse
    unsanitised then do as much as possible so the atom-feature getters
    (GetHybridization / GetImplicitValence / GetTotalNumHs) don't raise."""
    mol = Chem.MolFromPDBFile(pdb, removeHs=True, sanitize=True, proximityBonding=True)
    if mol is not None:
        return mol
    mol = Chem.MolFromPDBFile(pdb, removeHs=True, sanitize=False, proximityBonding=True)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception as e:
        log('sanitize(catchErrors) warning:', e)
    return mol


def load_ligand(lig_pdb, smiles=''):
    mol = _safe_mol_from_pdb(lig_pdb)
    if mol is None:
        raise RuntimeError('RDKit could not parse ligand PDB: %s' % lig_pdb)
    if smiles:
        ref = Chem.MolFromSmiles(smiles)
        if ref is not None:
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
            except Exception as e:
                log('AssignBondOrdersFromTemplate failed (%s); using PDB/proximity bonds' % e)
    return mol


def _heavy_atom_xyz(pdb):
    """Heavy-atom coords (Nx3) from a PDB's ATOM/HETATM lines."""
    xs = []
    for raw in open(pdb):
        ln = raw.rstrip('\n').ljust(80)
        if ln[:6] not in ('ATOM  ', 'HETATM'):
            continue
        el = ln[76:78].strip()
        if el == 'H' or (not el and ln[12:16].strip()[:1] == 'H'):
            continue
        try:
            xs.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            pass
    return np.array(xs, dtype=float) if xs else np.zeros((0, 3))


def cut_pocket(prot_pdb, lig_xyz, out_pdb, cutoff=10.0, drop_water=True):
    """Keep whole protein residues with any heavy atom within <cutoff> A of any
    ligand heavy atom. Residue key = (chainID, resSeq, iCode). Writes out_pdb and
    returns (n_residues, n_atom_lines). Only used when pocket_cutoff > 0."""
    lines = [raw.rstrip('\n').ljust(80) for raw in open(prot_pdb)]
    atoms = []   # (line_idx, reskey, x, y, z, is_heavy)
    for i, ln in enumerate(lines):
        if ln[:6] not in ('ATOM  ', 'HETATM'):
            continue
        resname = ln[17:20].strip().upper()
        if drop_water and resname in WATER_RESNAMES:
            continue
        el = ln[76:78].strip()
        is_h = (el == 'H') or (not el and ln[12:16].strip()[:1] == 'H')
        try:
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError:
            continue
        reskey = (ln[21:22], ln[22:26], ln[26:27])
        atoms.append((i, reskey, x, y, z, not is_h))

    if lig_xyz.shape[0] == 0 or not atoms:
        keep = list(atoms)
    else:
        heavy = [a for a in atoms if a[5]]
        prot_xyz = np.array([[a[2], a[3], a[4]] for a in heavy], dtype=float)
        dm = distance_matrix(prot_xyz, lig_xyz)         # (n_prot_heavy, n_lig)
        near = np.where(dm.min(axis=1) <= cutoff)[0]
        keep_res = set(heavy[j][1] for j in near)
        keep = [a for a in atoms if a[1] in keep_res]

    keep_lines = [lines[a[0]] for a in keep]
    with open(out_pdb, 'w') as f:
        f.write('\n'.join(keep_lines))
        f.write('\nEND\n')
    return len(set(a[1] for a in keep)), len(keep_lines)


def inter_graph_quiet(ligand, pocket, dis_threshold=5.0):
    """Quiet copy of dataset_GIGN_ZccE.inter_graph (no array prints)."""
    atom_num_l = ligand.GetNumAtoms()
    g = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dm = distance_matrix(pos_l, pos_p)
    idx = np.where(dm < dis_threshold)
    for i, j in zip(idx[0], idx[1]):
        g.add_edge(i, j + atom_num_l)
    g = g.to_directed()
    edges = list(g.edges(data=False))
    if edges:
        return torch.stack([torch.LongTensor((u, v)) for u, v in edges]).T
    return torch.empty((2, 0), dtype=torch.long)


def build_data(ligand, pocket, dis_threshold=5.0):
    """Mirror dataset_GIGN_ZccE.mols2graphs, in-memory (no pickle round-trip)."""
    atom_num_l = ligand.GetNumAtoms()
    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p = mol2graph(pocket)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=-1)
    edge_index_inter = inter_graph_quiet(ligand, pocket, dis_threshold=dis_threshold)
    y = torch.FloatTensor([0.0])
    pos = torch.cat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros(atom_num_l), torch.ones(pocket.GetNumAtoms())], dim=0)
    return Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter,
                y=y, pos=pos, split=split)


def load_model(ckpt, device):
    model = GIGN(35, 256, 3).to(device)
    if _load_model_dict is not None:
        _load_model_dict(model, ckpt)
    else:
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        model.load_state_dict(sd)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, help='staging dir with the prepared inputs')
    ap.add_argument('--name', required=True, help='compound name / file stem')
    args = ap.parse_args()

    stage, name = args.stage, args.name
    meta_path = os.path.join(stage, name + '_meta.json')
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}

    lig_pdb = meta.get('ligand_pdb') or os.path.join(stage, name + '_ligand.pdb')
    prot_pdb = meta.get('protein_pdb') or os.path.join(stage, name + '_protein.pdb')
    smiles = (meta.get('smiles') or '').strip()
    cutoff = float(meta.get('pocket_cutoff') or 5.0)
    drop_water = bool(meta.get('drop_water', True))
    ckpt = (meta.get('model') or '').strip() or DEFAULT_MODEL

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log('device:', device, '| pocket_cutoff:', cutoff, '| drop_water:', drop_water)
    log('ckpt:', ckpt)

    # 1) ligand
    ligand = load_ligand(lig_pdb, smiles)

    # 2) pocket: full protein (default, matches predict_ZccE.py) OR cut if cutoff > 0
    if cutoff and cutoff > 0:
        lig_xyz = _heavy_atom_xyz(lig_pdb)
        pocket_pdb = os.path.join(stage, name + '_pocket.pdb')
        n_res, n_at = cut_pocket(prot_pdb, lig_xyz, pocket_pdb, cutoff=cutoff, drop_water=drop_water)
        log('pocket: CUT to %d residues / %d atoms within %.1f A (drop_water=%s)' % (n_res, n_at, cutoff, drop_water))
        pocket_src = pocket_pdb
    else:
        log('pocket: using FULL receptor (no cut) -- matches predict_ZccE.py')
        pocket_src = prot_pdb

    pocket = _safe_mol_from_pdb(pocket_src)
    if pocket is None:
        raise RuntimeError('RDKit could not parse pocket/receptor PDB: %s' % pocket_src)

    log('graph: ligand %d atoms, pocket %d atoms' % (ligand.GetNumAtoms(), pocket.GetNumAtoms()))

    # 3) graph + inference
    data = build_data(ligand, pocket, dis_threshold=5.0)
    batch = Batch.from_data_list([data]).to(device)
    model = load_model(ckpt, device)
    with torch.no_grad():
        out = model(batch)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    pk = float(pred.view(-1)[0].detach().cpu())

    # 4) parseable result
    print('GIGN_PRED_PK %.6f' % pk, flush=True)
    print('GIGN_DELTAG %.6f' % (-1.36 * pk), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('GIGN_ERROR %s' % e, file=sys.stderr, flush=True)
        sys.exit(1)