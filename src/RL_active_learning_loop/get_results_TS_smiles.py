import pandas as pd
import os

# 1. Path to the subset (contains the Name/ligand_id and the scores)
# Based on your previous snippet, this is the subset file
subset_path = '/blue/lic/huangzihang/repos/elion/src/vina/LGBM_suzuki_vina_3_no_smile.csv'

# 2. Path to the source file containing the SMILES strings
reference_path = "/blue/lic/huangzihang/repos/elion/src/RL_active_learning_loop/results_TS/LGBM_suzuki_TS_3.csv"

# 3. Output path for the new training file
output_path = "/blue/lic/huangzihang/repos/elion/src/RL_active_learning_loop/results_vina/LGBM_suzuki_vina_3.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load dataframes
# subset_df should have 'ligand_id' and 'score'
subset_df = pd.read_csv(subset_path)
# reference_df should have 'Name' and 'SMILES'
reference_df = pd.read_csv(reference_path)

# 4. Merge: Take only rows that exist in the subset_df
# Matching 'ligand_id' from subset to 'Name' in the reference
merged_df = subset_df.merge(
    reference_df[['Name', 'SMILES']], 
    left_on='Ligand_ID', 
    right_on='Name', 
    how='inner'
)

# 5. Format to SMILES, LABELS
# We extract 'SMILES' and the 'score' (renamed to LABELS)
final_df = merged_df[['SMILES', 'Affinity']].rename(columns={'score': 'LABELS'})

# 6. Save as train.smi
final_df.to_csv(output_path, index=False)

print(f"Successfully created {output_path}")
print(f"Total molecules in subset: {len(final_df)}")
print("\nFirst 5 rows of train.smi:")
print(final_df.head())