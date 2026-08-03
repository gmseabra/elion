#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_atomtypes.py
=====================
Single-file replacement for the first half of the DeepAtom virtual-screening
preprocessing chain.  It merges THREE previously separate scripts:

    00_put_protein_and_ligands_into_directories.sh   (stage 1)
    01_preprocess_complexes_VS.py                    (stage 2)
    pipeline_VS.py                                   (stages 3-6)

which themselves replaced the original four bash scripts
(02a / 02b / 03a / 03b).

It is therefore equivalent to running, in order:

    /bin/bash  00_put_protein_and_ligands_into_directories.sh  -d BATCH_DIR
    python     01_preprocess_complexes_VS.py                       BATCH_DIR
    python     pipeline_VS.py  --batch-dir BATCH_DIR --pre-dir PRE_DIR --scripts-dir SCRIPTS_DIR

Usage
-----
    python generate_atomtypes.py \
        --batch-dir   /path/to/batch_dir \
        --pre-dir     /path/to/00_preprocess \
        --scripts-dir /path/to/model_split_data

    # Override CPU count (default: nCPU - 4):
    python generate_atomtypes.py ... --workers 8

    # Run only specific stages (1-based, comma-separated):
    python generate_atomtypes.py ... --stages 1,2

Stages
------
    1  Build Dataset_VS/<pdbid>/ dirs; copy each ligand + the shared protein
       into them.                                   (was 00_...directories.sh)
    2  Remove altlocs/waters, build <pdbid>_complex.pdb for every complex.
                                                     (was 01_preprocess_complexes_VS.py)
    3  Copy non-augmented complex PDBs -> Dataset_VS_Non_augmented/
    4  Generate augmented structures (rotation + translation)
    5  Generate atom-type files for non-augmented complexes
    6  Generate atom-type files for augmented complexes

(Stages 3-6 are exactly the four stages of the former pipeline_VS.py.)

External scripts still invoked as subprocesses
----------------------------------------------
    PRE_DIR/02c_augment_no_Chimera_VS.py          (stage 4)
    SCRIPTS_DIR/arpeggio_mod2/arpeggio.py         (stages 5 & 6)

Dependencies
------------
    pip install biopython     # stage 2
    pip install numpy         # used by 02c_augment_no_Chimera_VS.py
    # openbabel is OPTIONAL - only needed if a ligand exists solely as .mol2
    # (in the VS pipeline ligands already arrive as .pdb, so it is never used).

Directory layout expected under batch_dir
-----------------------------------------
    batch_dir/
        protein/<something>.pdb                 (one shared receptor)
        ligands_in_bound_pose/<pdbid>.pdb       (one file per pose/ligand)

After stage 1 this becomes, and the remaining stages consume:

    batch_dir/Dataset_VS/<pdbid>/<pdbid>_ligand.pdb
    batch_dir/Dataset_VS/<pdbid>/<pdbid>_protein.pdb
    batch_dir/Dataset_VS/<pdbid>/<pdbid>_complex.pdb   (built in stage 2)

Stages 3-6 additionally support a nested layout
(Dataset_VS/<compound>/<index>/<pdbid>_*.pdb); every leaf directory holding a
matching *_ligand.pdb + *_complex.pdb pair is processed.
"""

import argparse
import itertools
import multiprocessing
import os
import shutil
import subprocess
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

# --- Biopython (stage 2 only) -------------------------------------------------
# Imported up front but guarded, so that running only later stages does not
# require biopython to be installed.
try:
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB import PDBIO
    from Bio.PDB.PDBIO import Select
    _HAVE_BIO = True
except ImportError:                                    # pragma: no cover
    _HAVE_BIO = False

# --- OpenBabel (optional, stage 2) -------------------------------------------
# Only used if a ligand is present as .mol2 but not .pdb.  In the standard VS
# pipeline ligands already arrive as .pdb, so this branch is never taken.
try:                                                   # openbabel 3.x
    from openbabel import openbabel as ob
except ImportError:                                    # pragma: no cover
    try:                                               # openbabel 2.x
        import openbabel as ob
    except ImportError:
        ob = None


# ---------------------------------------------------------------------------
# Generic helpers
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


# ===========================================================================
# STAGE 1 - was 00_put_protein_and_ligands_into_directories.sh
#
# For every ligand <pdbid>.pdb in batch_dir/ligands_in_bound_pose/, create
# batch_dir/Dataset_VS/<pdbid>/ and copy:
#     ligand   ->  <pdbid>_ligand.pdb
#     protein  ->  <pdbid>_protein.pdb   (the single shared receptor)
# ===========================================================================

def stage1_put_into_directories(batch_dir: Path) -> None:
    _log("\n" + "=" * 70)
    _log("STAGE 1 - Build Dataset_VS/<pdbid>/ directories")
    _log("=" * 70)

    ligands_dir = batch_dir / "ligands_in_bound_pose"
    protein_dir = batch_dir / "protein"
    dataset_dir = batch_dir / "Dataset_VS"

    if not ligands_dir.is_dir():
        raise FileNotFoundError(f"ligands_in_bound_pose not found: {ligands_dir}")
    if not protein_dir.is_dir():
        raise FileNotFoundError(f"protein directory not found: {protein_dir}")

    # The original `cp ../protein/*.pdb dest_file` assumes exactly one receptor.
    protein_pdbs = sorted(protein_dir.glob("*.pdb"))
    if not protein_pdbs:
        raise FileNotFoundError(f"No protein .pdb found in {protein_dir}")
    if len(protein_pdbs) > 1:
        _log(f"  [WARN] {len(protein_pdbs)} protein PDBs found in {protein_dir}; "
             f"using the first: {protein_pdbs[0].name}")
    protein_pdb = protein_pdbs[0]

    lig_pdbs = sorted(ligands_dir.glob("*.pdb"))
    total = len(lig_pdbs)
    if total == 0:
        _log(f"  [WARN] No ligand .pdb files found in {ligands_dir}")
        return

    for i, lig in enumerate(lig_pdbs, 1):
        bar = lig.stem                                  # filename without .pdb
        out_dir = dataset_dir / bar
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lig, out_dir / f"{bar}_ligand.pdb")
        shutil.copy2(protein_pdb, out_dir / f"{bar}_protein.pdb")
        _log(f"  [{i}/{total}] {bar}")

    _log(f"Stage 1 complete - {total} complex directories prepared.")


# ===========================================================================
# STAGE 2 - was 01_preprocess_complexes_VS.py
#
# For every Dataset_VS/<pdbid>/ directory:
#   * strip altlocs (keep 'A') and water with Biopython  -> *_protein_wt_altloc.pdb
#   * blank the altloc column (col 17) via sed
#   * (optional) convert a .mol2-only ligand to .pdb with OpenBabel
#   * concatenate cleaned protein + renumbered ligand -> <pdbid>_complex.pdb
#
# The protein/ligand concatenation and TER-line handling are preserved
# byte-for-byte from the original script - this logic is fiddly and
# column-sensitive, so it is intentionally left unchanged.
# ===========================================================================

def stage2_preprocess_complexes(batch_dir: Path) -> None:
    _log("\n" + "=" * 70)
    _log("STAGE 2 - Build <pdbid>_complex.pdb for every complex")
    _log("=" * 70)

    if not _HAVE_BIO:
        raise ImportError(
            "Stage 2 requires Biopython.  Install it with:  pip install biopython"
        )

    vs_dataset_dir = str(batch_dir / "Dataset_VS")
    if not os.path.isdir(vs_dataset_dir):
        raise FileNotFoundError(f"Dataset_VS not found: {vs_dataset_dir}")

    # Selector that drops water and any altloc other than 'A'.
    class NotDisordered(Select):
        def accept_atom(self, atom):
            residue_id = atom.get_parent().get_id()
            hetfield = residue_id[0]

            isNotWater = hetfield[0] != "W"             # i.e. water
            isOrderedOrFirstAltloc = (
                not atom.is_disordered() or atom.get_altloc() == 'A'
            )
            return isNotWater and isOrderedOrFirstAltloc

    original_cwd = os.getcwd()
    os.chdir(vs_dataset_dir)
    try:
        pdb_codes = [pdb_code for pdb_code in os.listdir(".")]
        counter = 0
        num_complx = len(pdb_codes)

        for pdb_code in pdb_codes:
            prot_path = os.path.join(vs_dataset_dir, pdb_code)
            # Skip stray non-directory entries (originally this would crash).
            if not os.path.isdir(prot_path):
                continue
            os.chdir(prot_path)

            counter += 1
            print("===================================================================")
            print("{}:   complex {} (out of {})".format(pdb_code, counter, num_complx))

            lig_mol2 = "{0}_ligand.mol2".format(pdb_code)
            lig_pdb = "{0}_ligand.pdb".format(pdb_code)
            prot_pdb = "{0}_protein.pdb".format(pdb_code)
            cmplx_pdb = "{0}_complex.pdb".format(pdb_code)

            # intermediate file
            prot_wt_altloc = "{0}_protein_wt_altloc.pdb".format(pdb_code)

            # ----------------------------------------------------------------
            # remove altloc records (except altloc "A") and water molecules
            pdb_parser = PDBParser()
            s = pdb_parser.get_structure('my_pdb', prot_pdb)

            io = PDBIO()
            io.set_structure(s)
            io.save(prot_wt_altloc, select=NotDisordered())

            # ----------------------------------------------------------------
            # remove the altloc identifier "A" from column 17
            subprocess.call(["sed", "-i", "-e", 's/./ /17', prot_wt_altloc])

            # ----------------------------------------------------------------
            if os.path.exists(lig_mol2) and not os.path.exists(lig_pdb):
                # convert ligand from MOL2 to PDB format (needs OpenBabel)
                try:
                    obConversion = ob.OBConversion()
                    obConversion.SetInAndOutFormats("mol2", "pdb")
                    mol = ob.OBMol()
                    obConversion.ReadFile(mol, lig_mol2)
                    obConversion.WriteFile(mol, lig_pdb)
                except Exception:
                    print("*" * 20 + pdb_code + "*" * 20)
                    os.chdir("..")
                    continue

            # ----------------------------------------------------------------
            # concatenate the protein and ligand files
            print('-----------------debug6 os.getcwd(): %s' % os.getcwd())
            print('-----------------debug6 cmplx_pdb: %s' % cmplx_pdb)

            with open(cmplx_pdb, 'w') as outfile:  # 'w' not 'a': avoid corrupt output on retry
                # use a deque to keep track of the last two lines added;
                # it keeps both last line containing an atom, and the TER line
                last_line = deque(['_', '_'], maxlen=2)

                with open(prot_wt_altloc, 'r') as infile:
                    for line in infile:

                        isEND = line.startswith("END")
                        isTER = line.startswith("TER")
                        if line == "END\n":
                            isHydrogen = False
                        else:
                            isHydrogen = not isTER and len(line) > 77 and line[77] == "H"

                        if not isEND and not isHydrogen and not isTER:
                            last_line.append(line)
                            outfile.write(line)

                        elif isTER:
                            if len(last_line) < 2 or last_line[0] == '_':
                                last_line.append(line)

                            if len(last_line) < 2:
                                # Not enough context to build a proper TER line; write a bare one
                                outfile.write("TER\n")
                                continue

                            if last_line[1] == "TER\n":
                                ter_serial = deque(itertools.islice(last_line[0], 6, 11))
                            else:
                                ter_serial = deque(itertools.islice(last_line[1], 6, 11))

                            ter_serial_str = "%5s" % str(int(''.join(ter_serial).strip()) + 1)

                            if last_line[1] == "TER\n":
                                ter_resName = deque(itertools.islice(last_line[0], 17, 20))
                            else:
                                ter_resName = deque(itertools.islice(last_line[1], 17, 20))

                            ter_resName_str = ''.join(ter_resName)

                            if last_line[1] == "TER\n":
                                chain_id = last_line[0][21]
                            else:
                                chain_id = last_line[1][21]

                            chain_id_str = ''.join(chain_id)

                            # column 27 is the insertion code, e.g. in 1BCU.pdb
                            if last_line[1] == "TER\n":
                                ter_resSeq = deque(itertools.islice(last_line[0], 22, 27))
                            else:
                                ter_resSeq = deque(itertools.islice(last_line[1], 22, 27))

                            ter_resSeq_str = ''.join(ter_resSeq)

                            TER_line = "TER   " + ter_serial_str + " " * 6 + \
                                       ter_resName_str + " " + chain_id + \
                                       ter_resSeq_str + "\n"

                            outfile.write(TER_line)

                        elif isHydrogen:
                            pass

                        elif isEND:
                            break

                # Determine last protein atom serial number.
                # last_line is a maxlen=2 deque; [0] is older, [1] is newer.
                # After the loop, [1] is the TER line (or last ATOM if no TER),
                # and [0] is the last ATOM line before it.
                last_atom_line = None
                for candidate in reversed(last_line):
                    if candidate not in ('_', 'TER\n') and not candidate.startswith('TER'):
                        last_atom_line = candidate
                        break
                if last_atom_line is None:
                    raise ValueError("Could not find last ATOM line in {}".format(prot_wt_altloc))
                last_prot_serial = int(last_atom_line[6:11].strip()) + 1

                # also add chain identifier "y" to the ligand
                with open(lig_pdb, 'r') as infile:
                    atom_counter = 0
                    for line in infile:
                        if line.startswith(("ATOM", "HETATM")) and line[77] != "H":
                            # columns 7-11 are serial number of atom
                            old_lig_serial = int(line[6:11].strip())
                            new_lig_serial = last_prot_serial + old_lig_serial
                            new_lig_serial_str = "%5s" % (str(new_lig_serial))

                            last_col = line[76:78].strip()
                            atom_symbol = None

                            if len(last_col) > 0:
                                atom_symbol = last_col
                            else:
                                atom_symbol = line[12:16].strip()[0]

                            atom_counter += 1
                            atom_name_number = atom_symbol + str(atom_counter)
                            atom_name_number_str = None

                            if len(atom_name_number) < 4:
                                atom_name_number_str = " " + "%-3s" % atom_name_number
                            else:
                                atom_name_number_str = "%4s" % atom_name_number

                            lig_ResName = "LIG"

                            outfile.write("HETATM" + new_lig_serial_str + " " +
                                          atom_name_number_str + " " +
                                          lig_ResName + " " + 'y' + line[22:])

                outfile.write("END")

            os.chdir("..")
    finally:
        os.chdir(original_cwd)

    _log("Stage 2 complete - complex PDBs built.")


# ===========================================================================
# STAGE 3 - was pipeline_VS.py stage 1 (02a_..._make_directory.sh)
#
# Copies <pdbid>_complex.pdb -> Dataset_VS_Non_augmented/<pdbid>.pdb
# for every complex pair found anywhere under Dataset_VS/.
# ===========================================================================

def stage3_copy_non_augmented(batch_dir: Path) -> None:
    _log("\n" + "=" * 70)
    _log("STAGE 3 - Build Dataset_VS_Non_augmented/")
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

    _log(f"Stage 3 complete - {total} complexes processed.")


# ===========================================================================
# STAGE 4 - was pipeline_VS.py stage 2 (02b_augment_no_Chimera_VS_opt.sh)
#
# Calls 02c_augment_no_Chimera_VS.py for each complex in parallel.
# ===========================================================================

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


def stage4_augment(batch_dir: Path, pre_dir: Path, n_workers: int) -> None:
    _log("\n" + "=" * 70)
    _log("STAGE 4 - Generate augmented structures")
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

    _log(f"\nStage 4 complete - {ok} succeeded, {fail} failed.")


# ===========================================================================
# STAGES 5 & 6 - were pipeline_VS.py stages 3 & 4 (03a / 03b)
#
# Calls arpeggio.py for every *.pdb in the given input directory, writing
# .atomtypes results to the output directory.
# ===========================================================================

def _arpeggio_worker(args):
    """Top-level picklable worker for arpeggio calls.

    arpeggio.py ignores its outdir argument and always writes
    <stem>.atomtypes next to the input PDB file.  After running it
    we move the produced file into out_dir ourselves.
    """
    python_exe, arpeggio_script, pdb_file, out_dir = args
    pdb_file = Path(pdb_file)
    out_dir  = Path(out_dir)
    atomtypes_src = pdb_file.with_suffix(".atomtypes")   # written by arpeggio
    atomtypes_dst = out_dir / atomtypes_src.name          # where we want it
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
    """Generic runner for stages 5 and 6."""
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


def stage5_atomtypes_non_augmented(
    batch_dir: Path, scripts_dir: Path, n_workers: int
) -> None:
    input_dir = batch_dir / "Dataset_VS_Non_augmented"
    out_dir   = batch_dir / "atomtypes"
    _run_arpeggio_stage(5, "non-augmented", input_dir, out_dir, scripts_dir, n_workers)


def stage6_atomtypes_augmented(
    batch_dir: Path, scripts_dir: Path, n_workers: int
) -> None:
    input_dir = batch_dir / "Dataset_VS_augmented"
    out_dir   = batch_dir / "atomtypes_aug"
    _run_arpeggio_stage(6, "augmented", input_dir, out_dir, scripts_dir, n_workers)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build atom-type files for the Dataset_VS preprocessing pipeline "
                    "(merges 00_...directories.sh + 01_preprocess_complexes_VS.py + pipeline_VS.py).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--batch-dir", required=True,
        help="Root directory containing protein/, ligands_in_bound_pose/ (and, "
             "after stage 1, Dataset_VS/).  (the -d argument)."
    )
    parser.add_argument(
        "--pre-dir", required=True,
        help="Directory holding 02c_augment_no_Chimera_VS.py  (the -p argument)."
    )
    parser.add_argument(
        "--scripts-dir", required=True,
        help="Directory holding arpeggio_mod2/arpeggio.py  (the -s argument)."
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"Parallel workers for stages 4-6 (default: nCPU - 4 = {n_workers_default()})."
    )
    parser.add_argument(
        "--stages", type=str, default="1,2,3,4,5,6",
        help="Comma-separated list of stages to run (default: 1,2,3,4,5,6)."
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
        stage1_put_into_directories(batch_dir)

    if 2 in stages:
        stage2_preprocess_complexes(batch_dir)

    if 3 in stages:
        stage3_copy_non_augmented(batch_dir)

    if 4 in stages:
        stage4_augment(batch_dir, pre_dir, n_workers)

    if 5 in stages:
        stage5_atomtypes_non_augmented(batch_dir, scripts_dir, n_workers)

    if 6 in stages:
        stage6_atomtypes_augmented(batch_dir, scripts_dir, n_workers)

    _log("\n" + "=" * 70)
    _log("generate_atomtypes.py finished.")
    _log("=" * 70)


if __name__ == "__main__":
    main()