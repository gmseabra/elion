#!/usr/bin/env python3
# =============================================================================
# yupu_GIGN_backend.py  --  score ONE (ligand pose, receptor) pair with GIGN
#
# Same repo placement and algorithm as yupu_GIGN_pose.py:
#     /home/huangzihang/repos/elion/src/elion/properties/Yupu_GIGN/yupu_GIGN_backend.py
# (next to GIGN.py / HIL.py / utils.py / dataset_GIGN_ZccE.py / ./model/; cwd
# must be that dir when running, so the relative model path and local imports
# resolve.)
#
# DIFFERENCE FROM yupu_GIGN_pose.py:
#   yupu_GIGN_pose.py reads --stage/--name and expects pose_routes.py to have
#   already staged <stage>/<name>_{ligand,protein}.pdb + <name>_meta.json --
#   i.e. it can only be driven by the Flask endpoint (or a hand-built staging
#   dir that mimics it).
#
#   This script instead takes the ligand and receptor PDB paths directly on
#   the command line -- no staging dir, no meta.json -- so it can be run by
#   hand against any two PDB files:
#
#       python yupu_GIGN_backend.py --lig HJL-1_out.pdb --rec receptor.pdb \
#           --smiles 'Cc1ccc(...)cc1' --name HJL-1
#
#   NOTE: unlike yupu_GIGN_pose.py's meta.json (which is silently treated as
#   {} if the file is missing or the path is stale -- see the earlier
#   gign_20260707133629 vs gign_20260707133629_ rename incident), every
#   parameter here comes straight from argv. There is no meta.json this
#   script reads back and no silent fallback-to-empty-config path: if a flag
#   is wrong, argparse or an explicit error tells you, it does not quietly
#   substitute a default and keep going.
#
# Behaviour matches how the GIGN model was TRAINED (see preprocessing.py:
# pymol `remove resn HOH` + `remove hydrogens` + `byres <ligand> around 5`):
#   - ligand : Chem.MolFromPDBFile(removeHs=True), ATOM or HETATM records,
#              CONECT present or absent, CRLF or LF line endings -- all
#              accepted (see "ligand PDB handling" below). If --smiles is
#              given, bond orders are re-assigned from it via RDKit's
#              AssignBondOrdersFromTemplate.
#   - pocket : DEFAULT --pocket-cutoff 5 cuts the 5 A binding pocket -- whole
#              protein residues with any heavy atom within 5 A of any ligand
#              heavy atom, waters dropped unless --keep-water -- reproducing
#              the training prep. --pocket-cutoff <= 0 instead feeds the FULL
#              receptor (train/inference MISMATCH, see yupu_GIGN_pose.py).
#   - graph  : same featurisation as dataset_GIGN_ZccE (mol2graph + inter_graph,
#              dis_threshold=5 by default), batched, run through GIGN(35,256,3).
#
# Ligand PDB handling (why --lig accepts either format you throw at it):
#   RDKit's Chem.MolFromPDBFile does not care whether atoms are recorded as
#   ATOM or HETATM (the receptor's protein atoms are ATOM records and already
#   went through this same _safe_mol_from_pdb path in yupu_GIGN_pose.py), and
#   Python's default text-mode file reading already normalises CRLF/CR/LF to
#   '\n' before the manual fixed-column parsing in _heavy_atom_xyz/cut_pocket
#   ever sees a line -- so a Windows-style export (CRLF, HETATM+CONECT, no
#   explicit H, e.g. a prepared/reference ligand PDB) and a Vina/OpenBabel
#   pdbqt->pdb conversion (LF, ATOM only, no CONECT, a few explicit polar H)
#   both load correctly with no format-specific branching. This was verified
#   directly against both file shapes before writing this script.
#   The one thing worth knowing: a ligand PDB with CONECT records but no bond
#   ORDERS (which is all a PDB file can ever encode) does NOT give you
#   aromaticity for free -- RDKit will not perceive an aromatic ring from
#   CONECT connectivity alone, so a CONECT-only ligand and a CONECT-less
#   proximity-bonded ligand are equally non-aromatic without --smiles. Pass
#   --smiles whenever you have it; the warning below fires when you don't.
#
# Prints a parseable marker on stdout (same contract as yupu_GIGN_pose.py):
#     GIGN_PRED_PK <value>          (pK = -logKd/Ki ; dG = -1.36 * pK)
# All debug chatter goes to stderr so stdout stays clean. If --workdir is
# given (or left to default to a fresh temp dir), a <name>_gign.log with the
# same "# cmd / # cwd / # exit" header style as the endpoint's log is written
# there for side-by-side comparison with logs produced via pose_routes.py.
# =============================================================================
import os
import re
import sys
import json
import time
import argparse
import tempfile
import traceback

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

# default checkpoint (matches predict_ZccE.py / yupu_GIGN_pose.py); override via --model
DEFAULT_MODEL = './model/epoch-165, train_loss-0.2755, train_rmse-0.5249, valid_rmse-1.1359, valid_pr-0.8530.pt'
WATER_RESNAMES = {'HOH', 'WAT', 'DOD', 'H2O'}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# PDB loading -- identical to yupu_GIGN_pose.py, unchanged, so results are
# bit-for-bit reproducible between the two entry points given the same inputs.
# ---------------------------------------------------------------------------

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
        else:
            log('WARNING: --smiles could not be parsed by RDKit; using PDB/proximity bonds as-is')
    else:
        log('WARNING: no --smiles given -- bond orders come from CONECT records (if any) or '
            'proximity bonding only. Neither perceives aromaticity (a ligand PDB can only '
            'encode connectivity, never bond order), so aromatic rings will be featurised as '
            'plain single bonds. Pass --smiles for a featurisation that matches training.')
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


# ---------------------------------------------------------------------------
# New in this script: direct-path staging (replaces --stage/--name + meta.json)
# ---------------------------------------------------------------------------

def _sanitize_name(raw):
    """Same scrubbing pose_routes.py applies to the 'name' it puts in filenames."""
    return (re.sub(r'[^A-Za-z0-9_\-]', '_', raw)[:40]) or 'pose'


def _normalize_copy(src_path, dst_path):
    """Copy src_path -> dst_path, normalising line endings to '\n' and making
    sure the file ends in exactly one trailing newline. Defensive belt-and-
    braces: Python's own text-mode reads already do universal-newline
    translation for the manual column parsers in this file, and the RDKit
    version this was tested against loaded a CRLF ligand PDB fine as-is --
    but older/other RDKit builds have known issues with stray '\r' in PDB
    lines, and normalizing costs nothing, so every input is routed through
    this before RDKit or the column parsers ever see it."""
    with open(src_path, 'rb') as f:
        raw = f.read()
    text = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n').decode('utf-8', errors='replace')
    if not text.endswith('\n'):
        text += '\n'
    with open(dst_path, 'w') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(
        description='Score ONE (ligand pose, receptor) pair with GIGN, given direct file paths.')
    ap.add_argument('--lig', required=True, help='posed ligand .pdb (ATOM or HETATM, CONECT optional)')
    ap.add_argument('--rec', required=True, help='receptor .pdb (full, uncut)')
    ap.add_argument('--smiles', default='', help='ligand SMILES, used to fix bond orders/aromaticity')
    ap.add_argument('--name', default='', help='run name, used for staged filenames; default: --lig stem')
    ap.add_argument('--workdir', default='', help='dir to write normalized copies + pocket.pdb + log; '
                                                    'default: a fresh temp dir (kept, not auto-deleted)')
    ap.add_argument('--model', default='', help='absolute checkpoint .pt path; default: built-in checkpoint')
    ap.add_argument('--pocket-cutoff', type=float, default=5.0,
                     help='pocket radius in Angstrom; <=0 feeds the full receptor (train/inference mismatch)')
    ap.add_argument('--keep-water', action='store_true', help='do not drop HOH/WAT/DOD/H2O when cutting')
    ap.add_argument('--dis-threshold', type=float, default=5.0, help='ligand-pocket edge distance threshold')
    ap.add_argument('--device', default='', help='"cuda:0" / "cpu"; default: cuda:0 if available else cpu')
    args = ap.parse_args()

    t0 = time.time()
    argv_str = ' '.join(sys.argv)

    if not os.path.isfile(args.lig):
        raise RuntimeError('--lig not found: %s' % args.lig)
    if not os.path.isfile(args.rec):
        raise RuntimeError('--rec not found: %s' % args.rec)

    name = _sanitize_name(args.name or os.path.splitext(os.path.basename(args.lig))[0])
    workdir = args.workdir or tempfile.mkdtemp(prefix='yupu_gign_')
    os.makedirs(workdir, exist_ok=True)

    lig_pdb = os.path.join(workdir, name + '_ligand.pdb')
    prot_pdb = os.path.join(workdir, name + '_protein.pdb')
    _normalize_copy(args.lig, lig_pdb)
    _normalize_copy(args.rec, prot_pdb)

    # Informational only -- unlike yupu_GIGN_pose.py's meta.json, nothing in
    # this script ever reads this file back, so a stale/edited copy here
    # can't silently change behaviour on a later run the way a stale staging
    # dir could with --stage/--name.
    drop_water = not args.keep_water
    ckpt = args.model.strip() or DEFAULT_MODEL
    with open(os.path.join(workdir, name + '_meta.json'), 'w') as f:
        json.dump({
            'name': name, 'lig_src': os.path.abspath(args.lig), 'rec_src': os.path.abspath(args.rec),
            'ligand_pdb': lig_pdb, 'protein_pdb': prot_pdb, 'smiles': args.smiles,
            'pocket_cutoff': args.pocket_cutoff, 'drop_water': drop_water, 'model': ckpt,
            'dis_threshold': args.dis_threshold,
        }, f, indent=2)

    device = torch.device(args.device) if args.device else \
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log('device:', device, '| pocket_cutoff:', args.pocket_cutoff, '| drop_water:', drop_water)
    log('ckpt:', ckpt)
    log('workdir:', workdir)

    # 1) ligand
    ligand = load_ligand(lig_pdb, args.smiles.strip())

    # 2) pocket: full protein OR cut if cutoff > 0
    if args.pocket_cutoff and args.pocket_cutoff > 0:
        lig_xyz = _heavy_atom_xyz(lig_pdb)
        pocket_pdb = os.path.join(workdir, name + '_pocket.pdb')
        n_res, n_at = cut_pocket(prot_pdb, lig_xyz, pocket_pdb, cutoff=args.pocket_cutoff, drop_water=drop_water)
        log('pocket: CUT to %d residues / %d atoms within %.1f A (drop_water=%s)'
            % (n_res, n_at, args.pocket_cutoff, drop_water))
        pocket_src = pocket_pdb
    else:
        log('pocket: using FULL receptor (no cut) -- matches predict_ZccE.py / --pocket-cutoff<=0')
        pocket_src = prot_pdb

    pocket = _safe_mol_from_pdb(pocket_src)
    if pocket is None:
        raise RuntimeError('RDKit could not parse pocket/receptor PDB: %s' % pocket_src)

    log('graph: ligand %d atoms, pocket %d atoms' % (ligand.GetNumAtoms(), pocket.GetNumAtoms()))

    # 3) graph + inference
    data = build_data(ligand, pocket, dis_threshold=args.dis_threshold)
    batch = Batch.from_data_list([data]).to(device)
    model = load_model(ckpt, device)
    with torch.no_grad():
        out = model(batch)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    pk = float(pred.view(-1)[0].detach().cpu())
    dg = -1.36 * pk

    # 4) parseable result
    print('GIGN_PRED_PK %.6f' % pk, flush=True)
    print('GIGN_DELTAG %.6f' % dg, flush=True)

    log_path = os.path.join(workdir, name + '_gign.log')
    try:
        with open(log_path, 'w') as f:
            f.write('# pose GIGN score (direct-path backend)\n')
            f.write('# when : %s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
            f.write('# cmd  : %s\n' % argv_str)
            f.write('# cwd  : %s\n' % os.getcwd())
            f.write('# exit : 0\n')
            f.write('-' * 70 + '\n')
            f.write('GIGN_PRED_PK %.6f\n' % pk)
            f.write('GIGN_DELTAG %.6f\n' % dg)
            f.write('elapsed: %.1fs\n' % (time.time() - t0))
    except Exception as e:
        log('could not write log:', e)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print('GIGN_ERROR %s' % e, file=sys.stderr, flush=True)
        sys.exit(1)