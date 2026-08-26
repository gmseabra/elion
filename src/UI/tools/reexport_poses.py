#!/usr/bin/env python3
"""Re-export already-docked poses with their chemistry restored.

For batches that were downloaded before the fix: point this at the folder of
Vina outputs (or at a zip's `files/` folder of already-converted .pdb) and it
rewrites every pose as

    <stem>.sdf   bond orders + formal charges stated outright — open THIS in
                 Maestro; nothing is left for it to perceive
    <stem>.pdb   the same pose as HETATM records, with symmetric
                 order-duplicated CONECT and the formal charge in columns 79-80

plus `_manifest.csv`, `_best_poses.sdf` and `_best_poses.pdb`.

Nothing is re-docked and no coordinate moves: the atoms, their order and their
positions are copied through untouched. Only the chemistry annotation — which a
.pdbqt cannot carry and a .pdb carries badly — is rebuilt, preferring the
ligand's own SMILES over anything perceived.

    # simplest: everything under a results folder, SMILES found automatically
    python tools/reexport_poses.py /data/vina/LGBM_suzuki/4/pdbqt

    # be explicit about the SMILES table and where to write
    python tools/reexport_poses.py /data/vina/LGBM_suzuki/4/pdbqt \\
        --smiles-csv /data/results_TS/LGBM_suzuki_TS_4.csv \\
        --out /data/vina/LGBM_suzuki/4/export

    # check what would happen without writing anything
    python tools/reexport_poses.py /data/vina/... --dry-run

Exit status is 0 when every file was exported, 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # …/UI, so `uiapp` imports

try:
    from uiapp.core import pose_chem as pc
except Exception:                                   # running from a copied tree
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'pose_chem', os.path.join(os.path.dirname(_HERE), 'uiapp', 'core', 'pose_chem.py'))
    pc = importlib.util.module_from_spec(_spec)
    sys.modules['pose_chem'] = pc
    _spec.loader.exec_module(pc)

SUFFIXES = ('.pdbqt', '.pdb', '.ent')

# AutoDock atom type → element. A PDBQT puts its type where a PDB puts nothing,
# so the element has to be mapped back before anything can read the file as
# chemistry. (Same table as uiapp/routes/pose_routes.py; kept local so this tool
# runs standalone, with no Flask import.)
AD_TO_ELEM = {
    'C': 'C', 'A': 'C', 'N': 'N', 'NA': 'N', 'NS': 'N',
    'OA': 'O', 'OS': 'O', 'O': 'O', 'SA': 'S', 'S': 'S',
    'H': 'H', 'HD': 'H', 'HS': 'H', 'HB': 'H',
    'P': 'P', 'F': 'F', 'I': 'I', 'CL': 'Cl', 'BR': 'Br',
    'MG': 'Mg', 'CA': 'Ca', 'MN': 'Mn', 'FE': 'Fe', 'ZN': 'Zn', 'CU': 'Cu',
}


def pdbqt_atom_line(ln):
    """One PDBQT ATOM/HETATM line → a standard PDB line with a real element."""
    tail = ln[54:].split()
    adt = tail[-1].strip().upper() if tail else ''
    elem = AD_TO_ELEM.get(adt, adt[:1] if adt else '')
    if not elem:
        nm = ln[12:16].strip().upper()
        elem = nm[0] if nm else 'C'
    if len(elem) == 2:
        elem = elem[0] + elem[1].lower()
    return ln[:66].rstrip().ljust(66) + '          ' + elem.rjust(2)


def split_models(text, already_pdb):
    """(scores, [atom-line blocks]) — one block per docked pose, best first."""
    scores, models = [], []
    cur, cur_score = [], None
    for ln in text.splitlines():
        head = ln[:6].rstrip()
        if head == 'MODEL':
            if cur:
                models.append(cur), scores.append(cur_score)
            cur, cur_score = [], None
            continue
        if head == 'ENDMDL':
            if cur:
                models.append(cur), scores.append(cur_score)
            cur, cur_score = [], None
            continue
        if cur_score is None and ln.startswith('REMARK') and 'VINA RESULT' in ln:
            try:
                cur_score = float(ln.split(':', 1)[1].split()[0])
            except (IndexError, ValueError):
                pass
            continue
        if head in ('ATOM', 'HETATM'):
            cur.append(ln.rstrip('\n') if already_pdb else pdbqt_atom_line(ln))
    if cur:
        models.append(cur), scores.append(cur_score)
    return scores, models


def collect(inputs, recursive=True):
    """Every docked file under the given paths, de-duplicated and sorted."""
    out = []
    for raw in inputs:
        p = os.path.abspath(os.path.expanduser(raw))
        if os.path.isfile(p):
            if os.path.splitext(p)[1].lower() in SUFFIXES:
                out.append(p)
            continue
        if not os.path.isdir(p):
            print('  ! not found: %s' % raw, file=sys.stderr)
            continue
        if recursive:
            for root, _dirs, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in files
                           if os.path.splitext(f)[1].lower() in SUFFIXES)
        else:
            out.extend(os.path.join(p, f) for f in os.listdir(p)
                       if os.path.splitext(f)[1].lower() in SUFFIXES)
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Re-export docked poses as .sdf/.pdb with correct bond '
                    'orders and formal charges. Never re-docks, never moves an atom.')
    ap.add_argument('inputs', nargs='+',
                    help='files or folders holding *_out.pdbqt / .pdb / .ent')
    ap.add_argument('--out', default=None,
                    help='output folder (default: <first input>/reexport)')
    ap.add_argument('--smiles-csv', action='append', default=[],
                    help='CSV/SMI table with Name + SMILES columns; repeatable. '
                         'Extra tables to the ones found automatically.')
    ap.add_argument('--all-poses', action='store_true',
                    help='export every pose, not just Vina\'s best (MODEL 1)')
    ap.add_argument('--no-pdb', action='store_true', help='write only .sdf')
    ap.add_argument('--no-sdf', action='store_true', help='write only .pdb')
    ap.add_argument('--flat', action='store_true',
                    help='write into --out directly instead of a files/ subfolder')
    ap.add_argument('--no-recursive', action='store_true',
                    help='do not descend into subfolders')
    ap.add_argument('--dry-run', action='store_true', help='report, write nothing')
    ap.add_argument('--quiet', action='store_true', help='only the summary')
    args = ap.parse_args(argv)

    if not pc.RDKIT_OK:
        print('RDKit is required for this tool (pip install rdkit).', file=sys.stderr)
        return 1

    files = collect(args.inputs, recursive=not args.no_recursive)
    if not files:
        print('nothing to do: no %s files found' % '/'.join(SUFFIXES), file=sys.stderr)
        return 1

    out_root = args.out or os.path.join(
        os.path.abspath(os.path.expanduser(args.inputs[0]))
        if os.path.isdir(args.inputs[0])
        else os.path.dirname(os.path.abspath(args.inputs[0])), 'reexport')
    out_dir = out_root if args.flat else os.path.join(out_root, 'files')

    print('re-exporting %d file(s)' % len(files))
    index = pc.build_smiles_index(files, args.smiles_csv)
    print('SMILES index: %d entr%s%s' % (len(index), 'y' if len(index) == 1 else 'ies',
                                         '' if index else ' — falling back to perception'))
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    rows, best_sdf, best_pdb = [], [], []
    counts = {'template': 0, 'perceived': 0, 'geometry': 0, 'none': 0}
    failed = 0

    for fp in files:
        try:
            text = open(fp, errors='ignore').read()
        except OSError as e:
            print('  ! unreadable %s (%s)' % (fp, e), file=sys.stderr)
            failed += 1
            continue

        already_pdb = os.path.splitext(fp)[1].lower() != '.pdbqt'
        scores, models = split_models(text, already_pdb)
        if not models:
            print('  ! no poses in %s' % fp, file=sys.stderr)
            failed += 1
            continue

        smiles, where = pc.resolve_smiles(fp, text, index)
        stem = re.sub(r'\.(pdbqt|pdb|ent)$', '', os.path.basename(fp), flags=re.I)
        keep = range(len(models)) if args.all_poses else [0]

        for n in keep:
            score = scores[n] if n < len(scores) else None
            name = stem if len(list(keep)) == 1 else '%s_pose%d' % (stem, n + 1)
            pose = pc.build_pose(models[n], smiles=smiles)
            counts[pose.source] = counts.get(pose.source, 0) + 1

            remarks = []
            if score is not None:
                remarks.append('REMARK 999 VINA RESULT: %.2f kcal/mol' % score)
            if pose.smiles:
                remarks.append('REMARK   1 SMILES %s' % pose.smiles)

            sdf = '' if args.no_sdf else pc.pose_sdf_block(
                pose, name=name,
                props={'VINA_SCORE_KCAL_MOL': ('' if score is None else score),
                       'SMILES': pose.smiles, 'CHEMISTRY_SOURCE': pose.source,
                       'SOURCE_FILE': fp})
            pdb = '' if args.no_pdb else '\n'.join(
                pc.pose_pdb_lines(pose, models[n], remarks=remarks) + ['END']) + '\n'

            if not args.dry_run:
                if sdf:
                    open(os.path.join(out_dir, name + '.sdf'), 'w').write(sdf)
                if pdb:
                    open(os.path.join(out_dir, name + '.pdb'), 'w').write(pdb)
            if sdf:
                best_sdf.append((score, sdf))
            if pdb and n == 0:
                best_pdb.append((score, name, pc.pose_pdb_lines(
                    pose, models[n], remarks=['REMARK 999 SOURCE %s' % name])))

            rows.append({'file': name, 'score_kcal_mol': ('' if score is None else score),
                         'atoms': len(pose.atoms), 'chemistry': pose.source,
                         'net_charge': sum(pose.charges), 'smiles': pose.smiles,
                         'smiles_from': where, 'note': pose.note, 'source': fp})
            if not args.quiet:
                print('  %-42s %-9s %s' % (name, pose.source,
                                           ('%+.2f' % score) if score is not None else ''))

    if not args.dry_run and rows:
        with open(os.path.join(out_root, '_manifest.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['file', 'score_kcal_mol', 'atoms',
                                               'chemistry', 'net_charge', 'smiles',
                                               'smiles_from', 'note', 'source'])
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r['score_kcal_mol'] == '',
                                                 r['score_kcal_mol'])):
                w.writerow(r)
        if best_sdf:
            best_sdf.sort(key=lambda t: (t[0] is None, t[0]))
            open(os.path.join(out_root, '_best_poses.sdf'), 'w').write(
                ''.join(s for _s, s in best_sdf))
        if best_pdb:
            best_pdb.sort(key=lambda t: (t[0] is None, t[0]))
            lines = []
            for i, (_s, _n, blk) in enumerate(best_pdb, 1):
                lines.append('MODEL     %4d' % i)
                lines.extend(blk)
                lines.append('ENDMDL')
            lines.append('END')
            open(os.path.join(out_root, '_best_poses.pdb'), 'w').write('\n'.join(lines) + '\n')

    print('\n--- summary ---')
    print('  exported          : %d pose(s) from %d file(s)' % (len(rows), len(files)))
    print('  exact (SMILES)    : %d' % counts.get('template', 0))
    print('  perceived (RDKit) : %d' % counts.get('perceived', 0))
    print('  geometry only     : %d' % (counts.get('geometry', 0) + counts.get('none', 0)))
    if failed:
        print('  unreadable        : %d' % failed)
    if args.dry_run:
        print('  (dry run — nothing written)')
    else:
        print('  written to        : %s' % out_root)
        print('\nOpen the .sdf files in Maestro: bond orders and formal charges are '
              'explicit there,\nso nothing has to be perceived from the coordinates.')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
