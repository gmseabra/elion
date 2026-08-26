"""Ligand → PDBQT for the Vina batch, in parallel.

WHAT CHANGED AND WHY
--------------------
The old chain was

    <name>.pdb  ──obabel──▶  <name>.mol2  ──obabel──▶  <name>.pdbqt

and the first arrow is where the chemistry was lost. A PDB carries no formal
charges, and obabel does not honour RDKit's repeated-CONECT convention for
higher bond orders, so it re-perceived everything from the coordinates. On
γ-pyranones (chromen-4-one, coumarin, flavone) and aromatic N-oxides it gets
that wrong — the resulting mol2 is not even kekulizable — and every consumer
downstream inherited the mistake.

Now, when the sibling <name>.sdf exists (get_pdbs_from_results_TS_smiles.py
writes one per ligand), it is used directly:

    <name>.sdf  ──obabel──▶  <name>.pdbqt

An SDF states bond orders and formal charges outright, so obabel has nothing to
perceive and cannot perceive it wrongly. The .pdb path is kept as a fallback for
older batches that predate the .sdf files.

Docking results are NOT expected to change: Vina reads AutoDock atom types and
coordinates, and comparing the two chains atom-by-atom on the affected molecules
gives identical types and identical hydrogen counts. This is about the chemistry
that every OTHER tool reads out of the files.
"""
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

# Configuration derived from your original script
INPUT_DIR = "LGBM_suzuki/3/pdb"
OUTPUT_BASE = "LGBM_suzuki/3"

# obabel -xh keeps every hydrogen in the PDBQT (Vina only needs the polar ones,
# but keeping them all makes the file re-readable as a real molecule later).
OB_PDBQT_FLAGS = ["-opdbqt", "-xh"]


def convert_molecule(stem):
    """Worker: one ligand → <OUTPUT_BASE>/pdbqt/<stem>.pdbqt."""
    pdbqt_path = os.path.join(OUTPUT_BASE, "pdbqt", f"{stem}.pdbqt")
    if os.path.exists(pdbqt_path):
        return "skipped"

    sdf_path = os.path.join(INPUT_DIR, f"{stem}.sdf")
    pdb_path = os.path.join(INPUT_DIR, f"{stem}.pdb")

    try:
        if os.path.exists(sdf_path):
            # Preferred: bond orders and charges come straight from the SDF.
            subprocess.run(["obabel", "-isdf", sdf_path, "-O", pdbqt_path]
                           + OB_PDBQT_FLAGS, check=True, capture_output=True)
            return "success"

        # Fallback for batches written before the .sdf existed. Note this is the
        # lossy route: obabel has to re-perceive bond orders from geometry.
        mol2_path = os.path.join(OUTPUT_BASE, "mol2", f"{stem}.mol2")
        subprocess.run(["obabel", "-ipdb", pdb_path, "-omol2", "-O", mol2_path],
                       check=True, capture_output=True)
        subprocess.run(["obabel", "-imol2", mol2_path, "-O", pdbqt_path]
                       + OB_PDBQT_FLAGS, check=True, capture_output=True)
        return "legacy"
    except Exception:
        return "error"


def main():
    os.makedirs(os.path.join(OUTPUT_BASE, "mol2"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE, "pdbqt"), exist_ok=True)

    # One entry per ligand, preferring the .sdf and falling back to the .pdb.
    stems = sorted({os.path.splitext(f)[0] for f in os.listdir(INPUT_DIR)
                    if f.endswith(".sdf")} |
                   {os.path.splitext(f)[0] for f in os.listdir(INPUT_DIR)
                    if f.endswith(".pdb")})
    total_files = len(stems)
    n_sdf = sum(1 for s in stems if os.path.exists(os.path.join(INPUT_DIR, s + ".sdf")))
    print(f"Starting parallel conversion of {total_files} ligand(s) "
          f"— {n_sdf} from .sdf (lossless), {total_files - n_sdf} from .pdb (legacy)...")

    summary = {"success": 0, "legacy": 0, "error": 0, "skipped": 0}
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(convert_molecule, stems),
                            total=total_files, desc="Converting"))
    for res in results:
        summary[res] += 1

    print("\n--- Conversion Summary ---")
    print(f"Converted from .sdf (exact chemistry) : {summary['success']}")
    print(f"Converted from .pdb (legacy, perceived): {summary['legacy']}")
    print(f"Skipped (already exist)               : {summary['skipped']}")
    print(f"Errors                                : {summary['error']}")
    if summary["legacy"]:
        print("\nNote: the legacy .pdb route cannot preserve formal charges or "
              "bond orders.\nRe-run get_pdbs_from_results_TS_smiles.py to write "
              ".sdf files for those ligands.")


if __name__ == "__main__":
    main()
