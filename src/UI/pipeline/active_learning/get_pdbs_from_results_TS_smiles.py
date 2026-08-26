"""SMILES → 3D ligand structures for the Vina batch.

WHAT CHANGED AND WHY
--------------------
This script used to write ONE file per ligand, a .pdb, and that single choice is
where the chemistry started leaking away:

  * a PDB file has no place to record a formal charge (columns 79-80 exist but
    RDKit's writer leaves them blank and most tools ignore them), so a quinoline
    N-oxide's N⁺/O⁻ pair was gone the moment the file hit the disk;
  * RDKit writes higher bond orders as repeated CONECT entries, a convention the
    next tool in the chain (obabel) does not have to honour — and does not.

So `obabel -ipdb x.pdb -omol2` re-perceived the chemistry from coordinates, got
γ-pyranones and N-oxides wrong (an un-kekulizable aromatic ring in both cases),
and every consumer downstream inherited the mistake. Opened in Maestro, a
4H-chromen-4-one showed up as a 4-hydroxychromene and an N-oxide as N-hydroxy:
H-bond donors and acceptors swapped, net charge wrong.

The fix is to stop using PDB as the interchange format. Each ligand is now
written as:

    <name>.sdf   ← the one the pdbqt conversion reads. Bond orders and formal
                   charges are explicit fields in an MDL molfile, so nothing
                   downstream has to guess.
    <name>.smi   ← the ligand's SMILES, so any later step (the UI's pose
                   download, a re-export, a sanity check) can recover the exact
                   intended structure from the file name alone.
    <name>.pdb   ← still written, for tools that insist on PDB, and now with a
                   REMARK SMILES line carrying the ground truth.

Docking itself is unaffected by any of this — Vina reads AutoDock atom types and
coordinates, and those were always right. This is about every other tool that
reads the files.
"""
import os
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Define paths
input_csv = "results_TS/LGBM_suzuki_TS_4.csv"
output_dir = "../vina/docking_score/LGBM_suzuki/4/pdb"

# Embedding is deterministic so a rerun reproduces the same conformers.
RANDOM_SEED = 0xC0FFEE


def write_ligand(mol, smiles, name, out_dir):
    """Write <name>.sdf (+ .smi sidecar, + .pdb) for one embedded molecule."""
    sdf_path = os.path.join(out_dir, f"{name}.sdf")
    mol.SetProp("_Name", str(name))
    mol.SetProp("SMILES", smiles)
    with Chem.SDWriter(sdf_path) as w:          # bond orders + charges, losslessly
        w.write(mol)

    with open(os.path.join(out_dir, f"{name}.smi"), "w") as fh:
        fh.write(f"{smiles}\t{name}\n")

    pdb_path = os.path.join(out_dir, f"{name}.pdb")
    block = Chem.MolToPDBBlock(mol)
    with open(pdb_path, "w") as fh:             # SMILES first: a PDB cannot hold
        fh.write(f"REMARK   1 SMILES {smiles}\n")   # charges, so carry the truth
        fh.write(block)                             # alongside it
    return sdf_path


def main():
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    ok = failed_parse = failed_embed = 0

    for index, row in df.iterrows():
        smiles = row["SMILES"]
        name = str(row["Name"])

        # 1. Molecule from SMILES — this object is the ground truth for the whole
        #    downstream chain. Everything after here only moves it between files.
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES at row {index}: {smiles}")
            failed_parse += 1
            continue

        # 2. Explicit hydrogens (needed for sensible 3D geometry and for docking)
        mol = Chem.AddHs(mol)

        # 3. 3D coordinates. ETKDGv3 is the current recommended embedding.
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:                          # crowded/macrocyclic input
            status = AllChem.EmbedMolecule(mol, useRandomCoords=True,
                                           randomSeed=RANDOM_SEED)
        if status != 0:
            print(f"Could not generate 3D coordinates for: {name}")
            failed_embed += 1
            continue

        # 4. Quick relax for believable bond lengths and angles
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:                        # no MMFF params for this atom type
            AllChem.UFFOptimizeMolecule(mol)

        # 5. Write .sdf (+ .smi, + .pdb)
        write_ligand(mol, smiles, name, output_dir)
        ok += 1
        print(f"Successfully converted: {name}")

    print("\nProcessing complete.")
    print(f"  written        : {ok}")
    print(f"  bad SMILES     : {failed_parse}")
    print(f"  embed failures : {failed_embed}")
    print(f"\nNext step reads the .sdf files, not the .pdb — see "
          f"engines/vina/pdb_to_pdbqt_cpu_thread.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
