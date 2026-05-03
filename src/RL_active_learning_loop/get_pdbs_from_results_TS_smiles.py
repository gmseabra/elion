import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem

# Define paths
input_csv = "/blue/lic/huangzihang/repos/elion/src/RL_active_learning_loop/results_TS/LGBM_suzuki_TS_4.csv"
output_dir = "/blue/lic/huangzihang/repos/elion/src/vina/docking_score/LGBM_suzuki/4/pdb"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data
df = pd.read_csv(input_csv)

for index, row in df.iterrows():
    smiles = row['SMILES']
    name = str(row['Name'])
    
    # 1. Create Molecule object from SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    if mol:
        # 2. Add Hydrogens (important for 3D geometry and docking)
        mol = Chem.AddHs(mol)
        
        # 3. Generate 3D Coordinates
        # AllChem.ETKDG() is the standard algorithm for high-quality conformers
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        if status == 0:
            # 4. Optional: Quick energy minimization for better bond lengths
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 5. Save to PDB
            output_path = os.path.join(output_dir, f"{name}.pdb")
            Chem.MolToPDBFile(mol, output_path)
            print(f"Successfully converted: {name}")
        else:
            print(f"Could not generate 3D coordinates for: {name}")
    else:
        print(f"Invalid SMILES at row {index}: {smiles}")

print("\nProcessing complete.")