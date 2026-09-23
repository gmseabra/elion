#!/usr/bin/env python
# -*- coding:utf-8 -*-
# =============================================================================
# gatoraffinity_pose.py — score ONE posed protein-ligand complex
# -----------------------------------------------------------------------------
# The single-pose entry point for the Elion UI's "GatorAffinity ΔG (server)"
# scorer. inference.py is a benchmark driver: it wants a pre-processed .pkl, a
# label column and a batch, and it prints RMSE/Pearson/CI. None of that exists
# when a browser has just finished a Monte-Carlo search and holds one pose. This
# script is the missing middle — PDB text in, one number out.
#
#     python gatoraffinity_pose.py --stage <dir> --name <id>
#
# THE CONTRACT WITH THE SERVER, because it is what makes this file replaceable.
# uiapp/routes/pose_routes.py::pose_gatoraffinity_score writes three files into
# <stage> BEFORE running this, and reads stdout after:
#
#     <stage>/<name>_ligand.pdb    the posed ligand, HETATM records, heavy atoms
#     <stage>/<name>_protein.pdb   the receptor as loaded in the browser
#     <stage>/<name>_meta.json     everything else (below)
#
# and this script must print, on stdout, a line containing
#
#     GATOR_PRED_PK <float>
#
# That marker is the whole interface. The route greps for it and computes
# ΔG = -1.36 * pK itself. Everything else goes to stderr — the same split
# yupu_GIGN_pose.py uses next door, so a diagnostic line can never be mistaken
# for the score. Exit code is not the signal: a script that exits 0 without the
# marker is still a failure, which is deliberate, because a model that silently
# printed nothing used to look like a success with a missing number.
#
# ON THE SIGN, and it is the trap the playbook flags for GIGN (§16, NEW_FEATURES
# §1): pK is HIGHER-is-better, and it is the opposite of every reward elsewhere
# in this tree. If you ever wire this into Estimators, parse GATOR_DELTAG, not
# GATOR_PRED_PK. Like GIGN, there is deliberately no Property subclass here.
#
# meta.json keys, all written by the route from config/input_routes.yml:
#     name, ligand_pdb, protein_pdb   absolute paths (the three files above)
#     smiles                          ligand SMILES; "" is survivable, see below
#     pocket_cutoff                   Å; 0 => use the whole receptor
#     drop_water                      drop HOH/WAT/DOD when cutting
#     model                           checkpoint path; "" => the default below
#     device                          "cuda" | "cpu"
#     fragmentation_method            "PS_300"
#
# WHY THIS CUTS A POCKET AND process_pdbs.py DOES NOT.
# data/process_pdbs.py takes `protein_pdb` and hands it straight to
# pdb_to_list_blocks — no cutting anywhere in the file. That is not an omission:
# its input CSV already names pockets (data/example/example.csv points at
# 1a4h_pocket_5A.pdb), because the LP-PDBbind fine-tuning set is pockets. The UI
# has no pocket, it has whatever receptor the user loaded — often a whole
# prepared protein of 5000+ residues. Feeding that in is not "slower and
# equivalent"; the block set the model sees is a different distribution from the
# one it was trained on, and the prediction is not interpretable. So the cut
# happens here, at 5 Å byres, matching the *_pocket_5A.pdb convention.
#
# Set pocket_cutoff: 0 in the config to skip it and pass the full receptor. That
# is a supported knob and a bad idea; it exists so the mismatch can be measured
# rather than argued about.
#
# ON THE LABEL. The model is trained on the LP-PDBbind pKd/pKi column, so its
# output IS a pK (example.csv: 5.92, 4.85). It is emitted here unchanged and the
# route converts. If you retrain on kcal/mol, this marker's meaning changes and
# the route's conversion has to come out with it.
# =============================================================================

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from typing import List, Optional, Tuple

# Run from anywhere: the repo root is this file's own directory, and `data.*`,
# `models.*` and `utils.*` are imported as top-level packages throughout the
# codebase. The route already sets cwd to this directory; this makes a manual
# run from elsewhere work too, which is what you will want when debugging.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np


DEFAULT_CKPT = os.path.join(
    _HERE, 'model_checkpoints', 'Kd+Ki+IC50_experimental_fine_tuning.ckpt')

# Residue names dropped when cutting. pdb_to_list_blocks already skips anything
# Biopython flags 'W', so this is belt-and-braces for files where the flag does
# not survive — and it is what makes the pocket .pdb written beside the log
# match what the model actually saw.
WATERS = {'HOH', 'WAT', 'DOD', 'H2O', 'SOL'}


def log(msg: str) -> None:
    """Chatter to STDERR, so stdout carries only the markers.

    Matching yupu_GIGN_pose.py beside it, and for its reason: the marker is
    parsed by grepping stdout for a token, and a diagnostic line that happened
    to contain the token would be picked up as the score. Keeping the two
    streams separate makes that impossible rather than unlikely.

    The route merges them (stderr=subprocess.STDOUT) before writing
    <name>_gatoraffinity.log and echoing into the browser's mini-console, so
    nothing is lost by this — ordering in the log is preserved and every line
    still reaches the UI on failure.
    """
    print('[gatoraffinity_pose] %s' % msg, file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# PDB text handling
# -----------------------------------------------------------------------------
# Deliberately plain text rather than Biopython here, for two reasons. The cut
# has to preserve the ORIGINAL lines byte for byte — pdb_to_list_blocks reads
# the element column (77-78) and Biopython's writer is happy to re-derive it
# from the atom name, which turns a CA calcium into a C alpha carbon. And the
# pocket file is a debugging artefact people open in PyMOL; a re-serialised file
# with renumbered atoms is much harder to compare against the input.
# ─────────────────────────────────────────────────────────────────────────────
def _element_of(line: str) -> str:
    el = line[76:78].strip()
    if el:
        return el.upper()
    # No element column: fall back to the atom name, which is where the
    # ambiguity above lives. Only used for the H filter, never for the model.
    name = line[12:16].strip()
    return (name[0] if name else '').upper()


def _res_key(line: str) -> Tuple[str, str, str]:
    """chain + residue number + insertion code — one whole residue."""
    return (line[21:22], line[22:26].strip(), line[26:27].strip())


def _xyz(line: str) -> Tuple[float, float, float]:
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def read_atom_lines(path: str) -> List[str]:
    out = []
    with open(path, 'r', errors='replace') as fh:
        for ln in fh:
            if ln.startswith(('ATOM  ', 'HETATM')) and len(ln) >= 54:
                out.append(ln.rstrip('\n'))
    return out


def cut_pocket(receptor_lines: List[str],
               lig_coords: np.ndarray,
               cutoff: float,
               drop_water: bool) -> Tuple[List[str], int, int]:
    """`byres (receptor within <cutoff> of ligand)`, hydrogens and waters out.

    Whole residues, not atoms: a half-residue is not something the block
    representation can express — pdb_to_list_blocks builds one block per
    residue and averages its atoms, so clipping atoms off a residue silently
    moves that block's coordinate.

    Returns (lines, n_residues_kept, n_residues_seen).
    """
    keep_lines, seen = [], []
    for ln in receptor_lines:
        if _element_of(ln) == 'H':
            continue
        if drop_water and ln[17:20].strip().upper() in WATERS:
            continue
        seen.append(ln)

    if cutoff <= 0:
        n_res = len({_res_key(ln) for ln in seen})
        return seen, n_res, n_res

    # Group by residue once, then one vectorised distance pass per residue.
    order: List[Tuple[str, str, str]] = []
    by_res = {}
    for ln in seen:
        k = _res_key(ln)
        if k not in by_res:
            by_res[k] = []
            order.append(k)
        by_res[k].append(ln)

    cut2 = float(cutoff) ** 2
    for k in order:
        lines = by_res[k]
        try:
            coords = np.asarray([_xyz(ln) for ln in lines], dtype=np.float64)
        except ValueError:
            continue                      # malformed coordinate columns: skip the residue
        # (n_res_atoms, 1, 3) - (1, n_lig, 3) -> squared distances
        d2 = ((coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1)
        if d2.min() <= cut2:
            keep_lines.extend(lines)

    return keep_lines, len({_res_key(l) for l in keep_lines}), len(order)


def write_pdb(path: str, lines: List[str]) -> None:
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
        fh.write('\nEND\n')


def ligand_lines_heavy(lines: List[str]) -> List[str]:
    """Heavy atoms only.

    extract_pdb_ligand does NOT filter hydrogens (pdb_to_list_blocks does, for
    the protein side), and struct_to_bonds matches the 3-D structure against the
    SMILES — which is heavy-atom-only as RDKit parses it. An explicit-H ligand
    therefore fails the match rather than degrading, so filter here. The UI's own
    exporter already writes heavy atoms only (pose_routes.py builds its atom list
    from the pre-AddHs molecule), so this is for hand-run and imported poses.
    """
    return [ln for ln in lines if _element_of(ln) != 'H']


def unique_atom_names(lines: List[str]) -> List[str]:
    """Give every atom in the ligand a name unique within its residue.

    THIS IS NOT COSMETIC AND IT IS NOT OPTIONAL. Biopython's PDBParser keys a
    residue's atoms by name and DISCARDS later duplicates — quietly, on the
    QUIET=True parser that extract_pdb_ligand constructs. The UI's pose exporter
    (uiapp/routes/pose_routes.py, and PG.mc._ligandPdb in web/static/js/pose.js)
    writes each atom's name as its ELEMENT and nothing else:

        HETATM    1  C   LIG A   1      ...
        HETATM    2  C   LIG A   1      ...

    so a 40-atom ligand of C/N/O arrives as THREE atoms. Not an error, not an
    exception — three. The fragmentation then fails its graph match ("g1 node 40,
    g2 node 3"), extract_pdb_ligand catches that, falls back to per-atom blocks,
    and the model returns a perfectly plausible pK for a three-atom ligand.

    That is the worst failure mode available in this file: a number, in the right
    range, in the right units, on the wrong molecule. It cannot be caught
    downstream because nothing downstream knows how many atoms there should be.

    Renaming here rather than in the UI is deliberate — the exporter's output is
    also what /pose/gign_score and /pose/deepatom_score consume, and changing it
    would change their inputs to fix a problem neither of them has.

    Names are element + a base-36 counter per element, kept inside the 4-column
    name field with the PDB's own alignment (1-character elements start in
    column 14, 2-character elements in column 13).

    The counter is constrained to START WITH A DIGIT, which costs nothing and
    prevents a name that reads as the wrong element: plain base-36 makes carbon
    number 21 " CL ", and viewers that infer the element from the atom name
    rather than columns 77-78 — PyMOL among them, depending on how the file is
    loaded — will draw and select it as chlorine. The element column stays
    correct either way, so this never affected the model; it affected the person
    opening the staged file to find out why a score looked wrong.
    """
    counters, out, seen = {}, [], set()
    for ln in lines:
        el = _element_of(ln) or 'X'
        width = 4 - len(el)                       # room left for the counter
        n = counters.get(el, 0)
        name = None
        while n < 36 ** max(width, 1):
            suffix = np.base_repr(n, 36) if n else '0'
            if len(suffix) <= width and suffix[0].isdigit():
                cand = el + suffix
                pad = cand.ljust(3).rjust(4) if len(el) == 1 else cand.ljust(4)
                if pad not in seen:
                    name = pad
                    break
            n += 1
        counters[el] = n + 1
        if name is None:
            raise ValueError(
                'ran out of unique 4-character atom names for element %r — this '
                'ligand has more than %d %s atoms, which no PDB name field can '
                'express. Give the file real atom names before scoring it.'
                % (el, 36 ** max(width, 1), el))
        seen.add(name)
        out.append(ln[:12] + name + ln[16:])
    return out


def check_parsed_atoms(pdb_path: str, expected: int, what: str) -> None:
    """Guard the rename above, and the same trap on the protein side.

    Counts what Biopython actually yields and compares it with the heavy-atom
    lines in the file. They must agree; if they do not, something was dropped
    silently and every number after this point is about a different molecule.
    """
    from Bio.PDB import PDBParser
    st = PDBParser(QUIET=True).get_structure('chk', pdb_path)
    got = sum(1 for a in st[0].get_atoms() if (a.element or '').strip().upper() != 'H')
    if got != expected:
        raise ValueError(
            'Biopython parsed %d %s atoms from %s but the file holds %d heavy-atom '
            'records. Atoms were dropped, almost always because two atoms in one '
            'residue share a name (columns 13-16). Scoring this would produce a '
            'plausible number for the wrong molecule, so it is refused here.'
            % (got, what, pdb_path, expected))
    log('%s: %d atoms in, %d parsed ✓' % (what, expected, got))


def lig_code_and_resi(lines: List[str]) -> Tuple[str, Optional[int]]:
    """The residue name and number extract_pdb_ligand will look for.

    Read from the file rather than assumed. The UI writes `LIG ... 1`, but this
    script is also the thing you run by hand against a ligand pulled out of a
    crystal structure, where it is UNL, MOL, or the actual HET code.
    """
    if not lines:
        return 'UNL', None
    code = lines[0][17:20].strip() or 'UNL'
    try:
        resi = int(lines[0][22:26])
    except ValueError:
        resi = None
    return code, resi


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str):
    """inference.py::load_model, with the failure paths spelled out.

    Two checkpoint shapes are in circulation and they fail differently:

      * a pickled model object      -> torch.load returns it, done.
      * {'model_state_dict': ...}   -> the architecture is NOT in the file. It
        is rebuilt from args.json in the checkpoint's OWN directory. Move a
        .ckpt without its args.json and you get FileNotFoundError on a path you
        never typed, which reads as a corrupt checkpoint and is not one.
    """
    import torch
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError('checkpoint not found: %s' % ckpt_path)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        args_json = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'args.json')
        if not os.path.isfile(args_json):
            raise FileNotFoundError(
                'this checkpoint is a state dict, so the architecture comes from '
                'args.json beside it, and there is no args.json at %s. Copy it '
                'from wherever the checkpoint came from.' % args_json)
        from argparse import Namespace
        import models
        with open(args_json) as fh:
            train_args = Namespace(**json.load(fh))
        model = models.create_model(train_args)
        model.load_state_dict(ckpt['model_state_dict'])
        log('rebuilt the model from %s' % args_json)
        return model
    log('loaded a pickled model object')
    return ckpt


def to_device(batch: dict, device):
    import torch
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def resolve_device(want: str) -> str:
    """Never fail because a GPU is busy or absent — say so and use the CPU.

    A trajectory run queues one call per new Vina minimum. Turning "no CUDA
    device" into a hard error there loses the whole curve; a slow curve is worth
    more than no curve, and the log line says which one you got.
    """
    import torch
    want = (want or 'cuda').strip().lower()
    if want.startswith('cuda') and not torch.cuda.is_available():
        log('WARNING: device "%s" was requested but torch.cuda.is_available() is '
            'False — falling back to CPU. Expect this to take much longer per '
            'pose. (Common cause: the conda env has a CPU-only torch build.)' % want)
        return 'cpu'
    return want


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description='Score one posed protein-ligand complex with GatorAffinity.')
    ap.add_argument('--stage', required=True,
                    help='staging directory written by /pose/gatoraffinity_score')
    ap.add_argument('--name', required=True,
                    help='basename of the staged files inside --stage')
    # Overrides for a hand-run. meta.json wins when it carries the key, because
    # the server is the thing that has the configuration.
    ap.add_argument('--model', default=None, help='checkpoint (default: meta.json, else the repo default)')
    ap.add_argument('--device', default=None, help='cuda | cpu')
    ap.add_argument('--pocket-cutoff', type=float, default=None, help='Å; 0 = whole receptor')
    ap.add_argument('--fragmentation-method', default=None, help='PS_300')
    args = ap.parse_args()

    t0 = time.time()
    stage, name = args.stage, args.name

    meta_path = os.path.join(stage, name + '_meta.json')
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    else:
        log('no %s — falling back to the default file names in --stage' % meta_path)

    lig_pdb = meta.get('ligand_pdb') or os.path.join(stage, name + '_ligand.pdb')
    prot_pdb = meta.get('protein_pdb') or os.path.join(stage, name + '_protein.pdb')
    smiles = (meta.get('smiles') or '').strip()
    cutoff = args.pocket_cutoff if args.pocket_cutoff is not None else float(meta.get('pocket_cutoff', 5) or 0)
    drop_water = bool(meta.get('drop_water', True))
    ckpt = (args.model or meta.get('model') or '').strip() or DEFAULT_CKPT
    device_want = (args.device or meta.get('device') or 'cuda')
    frag = (args.fragmentation_method or meta.get('fragmentation_method') or 'PS_300')

    for p in (lig_pdb, prot_pdb):
        if not os.path.isfile(p):
            log('ERROR: missing input %s' % p)
            return 2

    # ── 1. read the two structures ──────────────────────────────────────────
    lig_lines = ligand_lines_heavy(read_atom_lines(lig_pdb))
    rec_lines = read_atom_lines(prot_pdb)
    if not lig_lines:
        log('ERROR: no heavy atoms in %s' % lig_pdb)
        return 2
    if not rec_lines:
        log('ERROR: no ATOM/HETATM records in %s' % prot_pdb)
        return 2
    lig_coords = np.asarray([_xyz(ln) for ln in lig_lines], dtype=np.float64)
    lig_code, lig_resi = lig_code_and_resi(lig_lines)
    log('ligand: %d heavy atoms, resname %s, resi %s' % (len(lig_lines), lig_code, lig_resi))

    # Rewrite with unique atom names before anything Biopython touches it. See
    # unique_atom_names — the UI names every atom after its element, and the
    # parser silently keeps one atom per name. The original file is left alone;
    # this copy sits beside it so the two can be diffed when a score looks wrong.
    lig_named = os.path.join(stage, name + '_ligand_named.pdb')
    write_pdb(lig_named, unique_atom_names(lig_lines))
    check_parsed_atoms(lig_named, len(lig_lines), 'ligand')

    if not smiles:
        log('WARNING: no SMILES in meta.json. Fragmentation needs one — '
            'extract_pdb_ligand will fall back to one block PER ATOM, which is a '
            'different input distribution from the PS_300 fragments the model was '
            'trained on. The number below will be a number; treat it as untrusted.')

    # ── 2. cut the pocket ───────────────────────────────────────────────────
    pocket_lines, n_keep, n_seen = cut_pocket(rec_lines, lig_coords, cutoff, drop_water)
    if not pocket_lines:
        log('ERROR: the %.1f Å pocket is empty — %d receptor residues were '
            'considered and none came within range of the ligand. Almost always '
            'the pose and the receptor are in different coordinate frames.'
            % (cutoff, n_seen))
        return 3
    pocket_pdb = os.path.join(stage, name + '_pocket.pdb')
    write_pdb(pocket_pdb, pocket_lines)
    # Real receptors carry proper per-residue atom names, so this normally just
    # passes — but a receptor that came out of a converter may not, and the
    # failure looks identical to the ligand one.
    check_parsed_atoms(pocket_pdb, len(pocket_lines), 'pocket')
    if cutoff > 0:
        log('pocket: %d of %d residues within %.1f Å -> %s' % (n_keep, n_seen, cutoff, pocket_pdb))
    else:
        log('pocket_cutoff = 0: passing the FULL receptor (%d residues). This is a '
            'train/inference mismatch — the checkpoint was fine-tuned on 5 Å '
            'pockets. -> %s' % (n_keep, pocket_pdb))

    # ── 3. blocks ───────────────────────────────────────────────────────────
    # Imported here, not at module scope: torch + rdkit + biotite is several
    # seconds of import, and every failure above this line is one a user can fix
    # from the message alone. Failing fast is worth more than tidy imports.
    #
    # UNDER redirect_stdout, and this is not tidiness. The upstream converters
    # print to stdout as a side effect of being imported and called:
    #
    #     data/converter/pdb_to_list_blocks.py:17   print(f'Project directory: …')
    #     data/converter/pdb_lig_to_blocks.py       "Could not fragment ligand …"
    #
    # stdout is the marker channel. Today none of those lines contains
    # GATOR_PRED_PK so nothing breaks, but that is luck, not design — the route
    # takes the FIRST line carrying the token and reads a float out of it, so one
    # upstream diagnostic that happened to mention it would be parsed as the
    # score. Redirecting is better than patching the vendored files: it survives
    # a re-sync from the GatorAffinity repo and covers prints nobody has added
    # yet. The lines are not lost — they land on stderr with the rest of the
    # chatter, in order.
    with contextlib.redirect_stdout(sys.stderr):
        from data.pdb_utils import VOCAB
        from data.dataset import blocks_to_data
        from data.converter.pdb_lig_to_blocks import extract_pdb_ligand
        from data.converter.pdb_to_list_blocks import pdb_to_list_blocks

        VOCAB.load_tokenizer(frag)

        list_lig_blocks, _ = extract_pdb_ligand(
            lig_named, lig_code, smiles or None, lig_idx=lig_resi,
            use_model=0, fragmentation_method=frag)
    lig_blocks = list_lig_blocks[0]
    if len(list_lig_blocks) > 1:
        log('NOTE: %d copies of %s in the ligand file; scoring the first.'
            % (len(list_lig_blocks), lig_code))

    # Fragmentation failure is CAUGHT inside extract_pdb_ligand — it prints
    # "Could not fragment ligand ..." and silently returns one block per atom.
    # That is a different input distribution from the PS_300 fragments the
    # checkpoint was trained on, so it must not pass as a normal result. The
    # tell is blocks == heavy atoms.
    if smiles and len(lig_blocks) == len(lig_lines):
        log('WARNING: the ligand came back as %d blocks for %d heavy atoms, i.e. '
            'one block per atom — fragmentation FAILED and extract_pdb_ligand fell '
            'back. Look for "Could not fragment ligand" above. Usual cause: the '
            'SMILES does not match the posed atoms (wrong molecule, or a pose '
            'exported from a different build). The score below is off-distribution.'
            % (len(lig_blocks), len(lig_lines)))

    # selected_chains=None => every chain in the pocket. Hard-coding 'A' (the
    # example CSV's value) would silently drop half of an interface pocket that
    # spans two chains, and the score would still look plausible.
    with contextlib.redirect_stdout(sys.stderr):
        prot_chains = pdb_to_list_blocks(pocket_pdb, selected_chains=None,
                                         return_indexes=False, is_rna=False)
    prot_blocks = sum(prot_chains, [])
    if not prot_blocks:
        log('ERROR: the pocket parsed to zero blocks. Check %s.' % pocket_pdb)
        return 3
    log('blocks: %d protein, %d ligand (%s)' % (len(prot_blocks), len(lig_blocks), frag))

    p_data = blocks_to_data(prot_blocks)
    l_data = blocks_to_data(lig_blocks)
    item = {
        'id': name,
        'X': np.concatenate([p_data['X'], l_data['X']], axis=0),
        'B': np.concatenate([p_data['B'], l_data['B']], axis=0),
        'A': np.concatenate([p_data['A'], l_data['A']], axis=0),
        'block_lengths': p_data['block_lengths'] + l_data['block_lengths'],
        'segment_ids': [0] * len(p_data['block_lengths']) + [1] * len(l_data['block_lengths']),
        # The forward pass takes a label and returns (loss, pred). There is no
        # ground truth here — 0.0 feeds the loss term, which is discarded.
        'label': 0.0,
    }

    # ── 4. run the model ────────────────────────────────────────────────────
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from data.dataset import PDBBindBenchmark
        from utils.random_seed import setup_seed, SEED

    setup_seed(SEED)
    device = resolve_device(device_want)
    log('checkpoint: %s' % ckpt)
    model = load_model(ckpt).to(device)
    model.eval()

    batch = to_device(PDBBindBenchmark.collate_fn([item]), device)
    with torch.no_grad():
        _loss, pred = model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            block_embeddings=batch.get('block_embeddings', None),
            block_embeddings0=batch.get('block_embeddings0', None),
            block_embeddings1=batch.get('block_embeddings1', None))
    pk = float(np.asarray(pred.detach().cpu().numpy()).reshape(-1)[0])

    # ── 5. answer ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    result = {
        'name': name, 'pred_pk': pk, 'deltaG': -1.36 * pk,
        'pocket_cutoff': cutoff, 'pocket_residues': n_keep,
        'protein_blocks': len(prot_blocks), 'ligand_blocks': len(lig_blocks),
        'fragmentation_method': frag, 'model': ckpt, 'device': device,
        'smiles': smiles, 'elapsed_s': round(elapsed, 2),
    }
    try:
        with open(os.path.join(stage, name + '_result.json'), 'w') as fh:
            json.dump(result, fh, indent=2)
    except OSError as e:
        log('could not write the result json: %s' % e)   # not fatal; stdout is the contract

    log('done in %.1fs' % elapsed)
    # THE line the route parses. Keep the token, the space and the plain decimal.
    print('GATOR_PRED_PK %.4f' % pk, flush=True)
    print('GATOR_DELTAG %.4f' % (-1.36 * pk), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        # Full traceback, because a bare exception message ("KeyError: 'B'") is
        # not something anyone can act on from a browser console. To stderr with
        # the rest of the chatter; the route merges the streams, so it still
        # reaches the log and the UI.
        traceback.print_exc(file=sys.stderr)
        print('GATOR_ERROR %s' % e, file=sys.stderr, flush=True)
        sys.exit(1)
