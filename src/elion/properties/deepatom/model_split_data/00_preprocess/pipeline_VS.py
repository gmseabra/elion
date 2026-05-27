#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_VS.py
==============
Unified Python replacement for the four bash pipeline scripts:

    02a_VS__Non_augmented_data_make_directory.sh
    02b_augment_no_Chimera_VS_opt.sh
    03a_generate_atomtypes_VS_Non_augmented.sh
    03b_generate_atomtypes_VS_augmented.sh

Equivalent to running them in order via:

    /bin/bash PRE_DIR/02a_VS__Non_augmented_data_make_directory.sh  -d BATCH_DIR
    /bin/bash PRE_DIR/02b_augment_no_Chimera_VS_opt.sh              -d BATCH_DIR  -p PRE_DIR
    /bin/bash PRE_DIR/03a_generate_atomtypes_VS_Non_augmented.sh    -d BATCH_DIR  -s SCRIPTS_DIR
    /bin/bash PRE_DIR/03b_generate_atomtypes_VS_augmented.sh        -d BATCH_DIR  -s SCRIPTS_DIR

Usage
-----
    python pipeline_VS.py \
        --batch-dir  /path/to/batch_dir \
        --pre-dir    /path/to/pre_dir \
        --scripts-dir /path/to/scripts_dir

    # Override CPU count (default: nCPU - 4):
    python pipeline_VS.py ... --workers 8

    # Run only specific stages (1-based, comma-separated):
    python pipeline_VS.py ... --stages 1,2

Stages
------
    1  Copy non-augmented complex PDBs to Dataset_VS_Non_augmented/
    2  Generate augmented structures (rotation + translation)
    3  Generate atom-type files for non-augmented complexes
    4  Generate atom-type files for augmented complexes

Dependencies
------------
    pip install biopython        # needed by 01_preprocess_complexes_VS.py
    pip install numpy            # needed by 02c_augment_no_Chimera_VS.py

Directory layout expected under Dataset_VS/
-------------------------------------------
The code supports two layouts automatically:

  Flat (one level):
      Dataset_VS/<pdbid>/<pdbid>_ligand.pdb
      Dataset_VS/<pdbid>/<pdbid>_complex.pdb

  Nested (two levels, e.g. virtual-screening batches):
      Dataset_VS/<compound>/<index>/<pdbid>_ligand.pdb
      Dataset_VS/<compound>/<index>/<pdbid>_complex.pdb

In both cases every leaf directory that contains a matching
*_ligand.pdb + *_complex.pdb pair is processed.
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def n_workers_default() -> int:
    """nCPU - 4, minimum 1 (mirrors the shell scripts)."""
    return max(1, multiprocessing.cpu_count() - 4)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _find_complex_pairs(dataset_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Walk dataset_dir recursively and return every leaf directory that contains
    both a *_ligand.pdb and a *_complex.pdb file.

    Returns a sorted list of (lig_pdb, cmplx_pdb, pdbid) triples, where pdbid
    is derived from the ligand filename stem (e.g. '1abc' from '1abc_ligand.pdb').

    Handles both flat and nested layouts:
        Flat:   Dataset_VS/<pdbid>/<pdbid>_ligand.pdb
        Nested: Dataset_VS/<compound>/<index>/<pdbid>_ligand.pdb
    """
    pairs: List[Tuple[Path, Path, str]] = []

    for leaf_dir in sorted(dataset_dir.rglob("*")):
        if not leaf_dir.is_dir():
            continue

        lig_files   = sorted(leaf_dir.glob("*_ligand.pdb"))
        cmplx_files = sorted(leaf_dir.glob("*_complex.pdb"))

        if not lig_files or not cmplx_files:
            continue

        # Match ligand <-> complex by shared pdbid prefix
        lig_map = {f.name.replace("_ligand.pdb", ""): f for f in lig_files}
        cmp_map = {f.name.replace("_complex.pdb", ""): f for f in cmplx_files}

        for pdbid in sorted(set(lig_map) & set(cmp_map)):
            pairs.append((lig_map[pdbid], cmp_map[pdbid], pdbid))

    return pairs


# ---------------------------------------------------------------------------
# Stage 1 - 02a_VS__Non_augmented_data_make_directory.sh
#
# Copies <pdbid>_complex.pdb  ->  Dataset_VS_Non_augmented/<pdbid>.pdb
# for every complex pair found anywhere under Dataset_VS/.
# ---------------------------------------------------------------------------

def stage1_copy_non_augmented(batch_dir: Path) -> None:
    """Copy each complex PDB into the Non_augmented flat directory."""
    _log("\n" + "=" * 70)
    _log("STAGE 1 - Build Dataset_VS_Non_augmented/")
    _log("=" * 70)

    dataset_dir = batch_dir / "Dataset_VS"
    non_aug_dir = batch_dir / "Dataset_VS_Non_augmented"
    non_aug_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset_VS directory not found: {dataset_dir}")

    pairs = _find_complex_pairs(dataset_dir)
    total = len(pairs)

    if total == 0:
        _log(f"  [WARN] No ligand/complex PDB pairs found under {dataset_dir}")
        return

    for i, (lig_pdb, cmplx_pdb, pdbid) in enumerate(pairs, 1):
        dst = non_aug_dir / f"{pdbid}.pdb"
        _log(f"COMPLEX  {pdbid} :   {i}  out of  {total}")
        shutil.copy2(cmplx_pdb, dst)
        _log("-" * 54)

    _log(f"Stage 1 complete - {total} complexes processed.")


# ---------------------------------------------------------------------------
# Stage 2 - 02b_augment_no_Chimera_VS_opt.sh
#
# Calls 02c_augment_no_Chimera_VS.py for each complex in parallel.
# ---------------------------------------------------------------------------

def _augment_worker(args):
    """Top-level picklable worker for ProcessPoolExecutor."""
    python_exe, script, lig_pdb, cmplx_pdb = args
    try:
        result = subprocess.run(
            [python_exe, script, str(lig_pdb), str(cmplx_pdb)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False, str(cmplx_pdb), result.stderr.strip()
        return True, str(cmplx_pdb), result.stdout.strip()
    except Exception as exc:
        return False, str(cmplx_pdb), str(exc)


def stage2_augment(batch_dir: Path, pre_dir: Path, n_workers: int) -> None:
    """Generate augmented structures for every complex."""
    _log("\n" + "=" * 70)
    _log("STAGE 2 - Generate augmented structures")
    _log("=" * 70)

    dataset_dir   = batch_dir / "Dataset_VS"
    augmented_dir = batch_dir / "Dataset_VS_augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)

    script = pre_dir / "02c_augment_no_Chimera_VS.py"
    if not script.exists():
        raise FileNotFoundError(f"Augmentation script not found: {script}")

    python_exe = sys.executable

    pairs = _find_complex_pairs(dataset_dir)
    total = len(pairs)

    if total == 0:
        _log(f"  [WARN] No ligand/complex PDB pairs found under {dataset_dir}")
        return

    tasks = [
        (python_exe, str(script), str(lig_pdb), str(cmplx_pdb))
        for lig_pdb, cmplx_pdb, _ in pairs
    ]

    _log(f"Augmenting {len(tasks)} of {total} complexes with {n_workers} workers.")
    _log(f"Output -> {augmented_dir}\n")

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_augment_worker, t): t for t in tasks}
        for idx, fut in enumerate(as_completed(futures), 1):
            success, name, msg = fut.result()
            if success:
                ok += 1
                _log(f"  [{idx}/{len(tasks)}] OK  -- {Path(name).name}")
            else:
                fail += 1
                _log(f"  [{idx}/{len(tasks)}] ERR -- {Path(name).name}: {msg}")

    _log(f"\nStage 2 complete - {ok} succeeded, {fail} failed.")


# ---------------------------------------------------------------------------
# Stage 3 & 4 - 03a / 03b  generate_atomtypes
#
# Calls arpeggio.py for every *.pdb in the given input directory, writing
# results to the output directory.  Parallelised the same way as the shells.
# ---------------------------------------------------------------------------

def _arpeggio_worker(args):
    """Top-level picklable worker for arpeggio calls.

    arpeggio.py ignores its outdir argument and always writes
    <stem>.atomtypes next to the input PDB file.  After running it
    we move the produced file into out_dir ourselves.
    """
    python_exe, arpeggio_script, pdb_file, out_dir = args
    pdb_file = Path(pdb_file)
    out_dir  = Path(out_dir)
    atomtypes_src = pdb_file.with_suffix(".atomtypes")  # written by arpeggio
    atomtypes_dst = out_dir / atomtypes_src.name         # where we want it
    try:
        result = subprocess.run(
            [python_exe, str(arpeggio_script), str(pdb_file), str(out_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False, str(pdb_file), result.stderr.strip()

        # Move .atomtypes from beside the PDB into the designated out_dir
        if atomtypes_src.exists():
            shutil.move(str(atomtypes_src), str(atomtypes_dst))
        elif not atomtypes_dst.exists():
            # arpeggio may have used a slightly different name; best-effort glob
            candidates = list(pdb_file.parent.glob(pdb_file.stem + "*.atomtypes"))
            if candidates:
                shutil.move(str(candidates[0]), str(atomtypes_dst))
            else:
                return False, str(pdb_file), ".atomtypes file not produced"

        return True, str(pdb_file), ""
    except Exception as exc:
        return False, str(pdb_file), str(exc)


def _run_arpeggio_stage(
    stage_num: int,
    label: str,
    input_dir: Path,
    out_dir: Path,
    scripts_dir: Path,
    n_workers: int,
) -> None:
    """Generic runner for stages 3 and 4."""
    _log("\n" + "=" * 70)
    _log(f"STAGE {stage_num} - Generate atom-type files ({label})")
    _log("=" * 70)

    arpeggio = scripts_dir / "arpeggio_mod2" / "arpeggio.py"
    if not arpeggio.exists():
        raise FileNotFoundError(f"arpeggio.py not found: {arpeggio}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # rglob finds PDBs in flat or nested subdirectories
    pdb_files = sorted(input_dir.rglob("*.pdb"))
    total     = len(pdb_files)

    if total == 0:
        _log(f"  [WARN] No PDB files found in {input_dir}")
        return

    _log(f"Processing {total} PDB files with {n_workers} workers.")
    _log(f"Input  -> {input_dir}")
    _log(f"Output -> {out_dir}\n")

    python_exe = sys.executable
    tasks = [(python_exe, arpeggio, pdb, out_dir) for pdb in pdb_files]

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_arpeggio_worker, t): t for t in tasks}
        for idx, fut in enumerate(as_completed(futures), 1):
            success, name, msg = fut.result()
            short = Path(name).name
            if success:
                ok += 1
                _log(f"  [{idx}/{total}] OK  -- {short}")
            else:
                fail += 1
                _log(f"  [{idx}/{total}] ERR -- {short}: {msg}")

    moved = len(list(out_dir.glob("*.atomtypes")))
    _log(f"\nStage {stage_num} complete - {ok} succeeded, {fail} failed "
         f"({moved} .atomtypes files in {out_dir.name}/).")


def stage3_atomtypes_non_augmented(
    batch_dir: Path, scripts_dir: Path, n_workers: int
) -> None:
    input_dir = batch_dir / "Dataset_VS_Non_augmented"
    out_dir   = batch_dir / "atomtypes"
    _run_arpeggio_stage(3, "non-augmented", input_dir, out_dir, scripts_dir, n_workers)


def stage4_atomtypes_augmented(
    batch_dir: Path, scripts_dir: Path, n_workers: int
) -> None:
    input_dir = batch_dir / "Dataset_VS_augmented"
    out_dir   = batch_dir / "atomtypes_aug"
    _run_arpeggio_stage(4, "augmented", input_dir, out_dir, scripts_dir, n_workers)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Dataset_VS preprocessing pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--batch-dir", required=True,
        help="Root directory containing Dataset_VS/  (the -d argument)."
    )
    parser.add_argument(
        "--pre-dir", required=True,
        help="Directory holding the preprocessing scripts  (the -p argument)."
    )
    parser.add_argument(
        "--scripts-dir", required=True,
        help="Directory holding arpeggio_mod2/arpeggio.py  (the -s argument)."
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"Parallel workers (default: nCPU - 4 = {n_workers_default()})."
    )
    parser.add_argument(
        "--stages", type=str, default="1,2,3,4",
        help="Comma-separated list of stages to run (default: 1,2,3,4)."
    )

    args   = parser.parse_args()
    stages = {int(s.strip()) for s in args.stages.split(",")}

    batch_dir   = Path(args.batch_dir).resolve()
    pre_dir     = Path(args.pre_dir).resolve()
    scripts_dir = Path(args.scripts_dir).resolve()
    n_workers   = args.workers if args.workers is not None else n_workers_default()

    _log(f"batch_dir   : {batch_dir}")
    _log(f"pre_dir     : {pre_dir}")
    _log(f"scripts_dir : {scripts_dir}")
    _log(f"workers     : {n_workers}")
    _log(f"stages      : {sorted(stages)}")

    if 1 in stages:
        stage1_copy_non_augmented(batch_dir)

    if 2 in stages:
        stage2_augment(batch_dir, pre_dir, n_workers)

    if 3 in stages:
        stage3_atomtypes_non_augmented(batch_dir, scripts_dir, n_workers)

    if 4 in stages:
        stage4_atomtypes_augmented(batch_dir, scripts_dir, n_workers)

    _log("\n" + "=" * 70)
    _log("Pipeline finished.")
    _log("=" * 70)


if __name__ == "__main__":
    main()