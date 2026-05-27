# -*- coding: utf-8 -*-
"""
augment_complexes.py
====================
Chimera-free replacement for 02c_augment_in_Chimera_VS.py

Generates augmented protein-ligand complex PDB files by applying random
rotations (about the ligand centre) and random translations -- replicating
exactly what the original Chimera script did, but using only NumPy.

Usage (single complex, drop-in for original Chimera call)
---------------------------------------------------------
    python augment_complexes.py <ligand_pdb> <complex_pdb>

    e.g.
    python augment_complexes.py 1abc_ligand.pdb 1abc_complex.pdb

Usage (batch -- mirrors 02b_augment_in_Chimera_VS_opt.sh)
---------------------------------------------------------
    python augment_complexes.py batch \
        --dataset-dir /path/to/Dataset_VS \
        --workers 8

Output
------
Augmented PDB files are written to:
    <dataset_dir>/Dataset_VS_augmented/   (batch mode)
    ../../Dataset_VS_augmented/           (single mode, matching original)

Each file is named:  <pdb_code>_augmented_<sample_id>.pdb
Produces NUMBER_OF_SCANNED_ROT_ANGLES x NUMBER_OF_SCANNED_TURN_AXES
augmented structures per complex (default 6 x 6 = 36).

Dependencies
------------
    pip install numpy
"""

import os
import sys
import argparse
import math
import numpy as np
from random import randint, uniform, random as rnd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# tuneable constants (mirror original script)
NUMBER_OF_SCANNED_ROT_ANGLES = 6
NUMBER_OF_SCANNED_TURN_AXES  = 6
START_ANGLE = -180
END_ANGLE   =  180


# ===========================================================================
# PDB I/O  (pure text parsing -- preserves exact PDB formatting)
# ===========================================================================

def parse_pdb(pdb_path):
    """
    Parse a PDB file into a list of raw line strings.
    Returns (lines, coord_indices, coords_array).

    coord_indices : list of int   -- indices into lines that are ATOM/HETATM
    coords_array  : (N, 3) float64 array of those coordinates
    """
    with open(pdb_path, 'r') as fh:
        lines = fh.readlines()

    coord_indices = []
    coords = []
    for i, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coord_indices.append(i)
                coords.append([x, y, z])
            except ValueError:
                pass  # skip malformed coordinate lines

    if coords:
        coords_array = np.array(coords, dtype=np.float64)
    else:
        coords_array = np.empty((0, 3))

    return lines, coord_indices, coords_array


def write_pdb(lines, coord_indices, new_coords, output_path):
    """
    Write a PDB file identical to the source but with coordinates replaced
    by new_coords. Column positions follow PDB fixed-width standard:
        cols 31-38: X, cols 39-46: Y, cols 47-54: Z  (1-based)
        i.e. line[30:38], line[38:46], line[46:54]    (0-based)
    """
    out_lines = list(lines)
    for j, idx in enumerate(coord_indices):
        line = out_lines[idx]
        x, y, z = new_coords[j]
        out_lines[idx] = (
            line[:30]
            + "{:8.3f}".format(x)
            + "{:8.3f}".format(y)
            + "{:8.3f}".format(z)
            + line[54:]
        )
    with open(output_path, 'w') as fh:
        fh.writelines(out_lines)


# ===========================================================================
# Geometry helpers
# ===========================================================================

def get_ligand_center(lig_pdb):
    """
    Compute the geometric centre of the ligand as the midpoint of its
    bounding box -- identical to the original Chimera script logic.
    """
    _, _, coords = parse_pdb(lig_pdb)
    if coords.size == 0:
        raise ValueError("No ATOM/HETATM coordinates found in {}".format(lig_pdb))
    maxc = coords.max(axis=0)
    minc = coords.min(axis=0)
    return (maxc + minc) / 2.0


def rodrigues_rotation_matrix(axis, angle_deg):
    """
    Build a 3x3 rotation matrix using Rodrigues' rotation formula.

    Parameters
    ----------
    axis      : array-like (3,)  -- arbitrary rotation axis (need not be unit)
    angle_deg : float            -- rotation angle in degrees

    Returns
    -------
    R : (3, 3) float64 ndarray
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm

    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c

    x, y, z = axis
    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])
    return R


def apply_rotation(coords, center, axis, angle_deg):
    """
    Rotate coords by angle_deg degrees around axis passing through center.
    Mirrors Chimera's turn command with a specified centre.
    """
    R   = rodrigues_rotation_matrix(axis, angle_deg)
    pts = coords - center        # translate to origin
    pts = (R.dot(pts.T)).T       # rotate
    pts = pts + center           # translate back
    return pts


def apply_translation(coords, axis, length):
    """
    Translate coords along axis by length Angstroms.
    Mirrors Chimera's move command (axis normalised internally).
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return coords
    unit = axis / norm
    return coords + unit * length


# ===========================================================================
# Core augmentation logic
# ===========================================================================

def augment_complex_safe(lig_pdb, cmplx_pdb, augmented_dir):
    """
    Generate all augmented variants for a single protein-ligand complex.

    Replicates the nested loop in 02c_augment_in_Chimera_VS.py:
        for rot_angle in range(START, END, step):   # 6 base angles
            for turn_axis in range(6):              # 6 random axes
                rotate around ligand centre
                translate by random vector and length
                write PDB
    """
    # create output dir if it does not exist yet
    try:
        os.makedirs(augmented_dir)
    except OSError:
        pass  # already exists -- fine

    pdb_code = os.path.basename(cmplx_pdb).split('_')[0]
    center = get_ligand_center(lig_pdb)
    lines, coord_indices, coords_orig = parse_pdb(cmplx_pdb)

    if coords_orig.size == 0:
        print("  [WARN] No coordinates in {} -- skipping.".format(cmplx_pdb))
        return

    step = (END_ANGLE - START_ANGLE) // NUMBER_OF_SCANNED_ROT_ANGLES  # 60 degrees

    sample_id = -1  # first index 0, same as original

    for base_angle in range(START_ANGLE, END_ANGLE, step):
        # jitter within +/- step (matches original randint(-step, step))
        rot_angle = base_angle + randint(-step, step)

        for _ in range(NUMBER_OF_SCANNED_TURN_AXES):
            sample_id += 1

            # rotation: random integer axis components 0-999 (matches original)
            rot_axis = np.array([
                int(1000 * rnd()),
                int(1000 * rnd()),
                int(1000 * rnd()),
            ], dtype=np.float64)

            coords = apply_rotation(coords_orig.copy(), center, rot_axis, rot_angle)

            # translation: random integer axis, random length in [-1, 1] Angstroms
            move_axis = np.array([
                int(1000 * rnd()),
                int(1000 * rnd()),
                int(1000 * rnd()),
            ], dtype=np.float64)
            move_length = uniform(-1.0, 1.0)

            coords = apply_translation(coords, move_axis, move_length)

            out_name = "{0}_augmented_{1}.pdb".format(pdb_code, sample_id)
            out_path = os.path.join(augmented_dir, out_name)
            write_pdb(lines, coord_indices, coords, out_path)

    print("  [{}] wrote {} augmented structures -> {}".format(
        pdb_code, sample_id + 1, augmented_dir))


# ===========================================================================
# Batch mode (mirrors the shell script)
# ===========================================================================

def _worker(args):
    """Top-level function required for multiprocessing pickling."""
    lig_pdb, cmplx_pdb, augmented_dir = args
    try:
        augment_complex_safe(lig_pdb, cmplx_pdb, augmented_dir)
        return True, os.path.basename(cmplx_pdb)
    except Exception as exc:
        return False, "{}: {}".format(os.path.basename(cmplx_pdb), exc)


def batch_augment(dataset_dir, n_workers=None):
    """
    Walk Dataset_VS/<pdbid>/ directories and augment every complex in
    parallel -- mirroring 02b_augment_in_Chimera_VS_opt.sh.
    """
    dataset_name  = "Dataset_VS"
    augmented_dir = os.path.join(dataset_dir, "{}_augmented".format(dataset_name))
    src_dir       = os.path.join(dataset_dir, dataset_name)

    if not os.path.isdir(src_dir):
        raise IOError("Source directory not found: {}".format(src_dir))

    pdbids = sorted(
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    )

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 4)

    print("Batch augmentation: {} complexes, {} workers".format(len(pdbids), n_workers))
    print("Output -> {}\n".format(augmented_dir))

    tasks = []
    for pdbid in pdbids:
        cdir      = os.path.join(src_dir, pdbid)
        lig_pdb   = os.path.join(cdir, "{}_ligand.pdb".format(pdbid))
        cmplx_pdb = os.path.join(cdir, "{}_complex.pdb".format(pdbid))
        if not os.path.exists(lig_pdb) or not os.path.exists(cmplx_pdb):
            print("  [SKIP] {}: missing ligand or complex PDB".format(pdbid))
            continue
        tasks.append((lig_pdb, cmplx_pdb, augmented_dir))

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = dict((pool.submit(_worker, t), t) for t in tasks)
        total   = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            success, msg = fut.result()
            if success:
                ok += 1
                print("  [{}/{}] OK  -- {}".format(i, total, msg))
            else:
                fail += 1
                print("  [{}/{}] ERR -- {}".format(i, total, msg))

    print("\nDone. {} succeeded, {} failed.".format(ok, fail))


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    # backward-compat: bare positional args (original Chimera call style)
    # e.g.  python augment_complexes.py 1abc_ligand.pdb 1abc_complex.pdb
    non_flag_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(non_flag_args) >= 2 and non_flag_args[0] not in ("batch", "single"):
        lig_pdb, cmplx_pdb = non_flag_args[0], non_flag_args[1]
        cmplx_dir     = os.path.dirname(os.path.abspath(cmplx_pdb))
        augmented_dir = os.path.normpath(
            os.path.join(cmplx_dir, "..", "..", "Dataset_VS_augmented")
        )
        augment_complex_safe(lig_pdb, cmplx_pdb, augmented_dir)
        return

    parser = argparse.ArgumentParser(
        description="Augment protein-ligand complexes via random rotation + "
                    "translation (Chimera-free)."
    )
    subparsers = parser.add_subparsers(dest="mode")

    # single-complex sub-command
    single = subparsers.add_parser("single", help="Augment one complex.")
    single.add_argument("lig_pdb",   help="Path to ligand PDB  (e.g. 1abc_ligand.pdb)")
    single.add_argument("cmplx_pdb", help="Path to complex PDB (e.g. 1abc_complex.pdb)")
    single.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: ../../Dataset_VS_augmented)."
    )

    # batch sub-command
    batch = subparsers.add_parser("batch", help="Augment all complexes in Dataset_VS/.")
    batch.add_argument("--dataset-dir", required=True,
                       help="Parent directory containing Dataset_VS/")
    batch.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: nCPU-4).")

    args = parser.parse_args()

    if args.mode == "single":
        if args.out_dir:
            augmented_dir = args.out_dir
        else:
            cmplx_dir     = os.path.dirname(os.path.abspath(args.cmplx_pdb))
            augmented_dir = os.path.normpath(
                os.path.join(cmplx_dir, "..", "..", "Dataset_VS_augmented")
            )
        augment_complex_safe(args.lig_pdb, args.cmplx_pdb, augmented_dir)

    elif args.mode == "batch":
        batch_augment(args.dataset_dir, n_workers=args.workers)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()