import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Configuration derived from your original script
INPUT_DIR = "/blue/lic/huangzihang/repos/elion/src/vina/LGBM_suzuki/3/pdb"
OUTPUT_BASE = "/blue/lic/huangzihang/repos/elion/src/vina/LGBM_suzuki/3"

def convert_molecule(pdb_file):
    """Worker function to process a single molecule."""
    basename = os.path.splitext(pdb_file)[0]
    mol2_path = os.path.join(OUTPUT_BASE, "mol2", f"{basename}.mol2")
    pdbqt_path = os.path.join(OUTPUT_BASE, "pdbqt", f"{basename}.pdbqt")
    pdb_path = os.path.join(INPUT_DIR, pdb_file)
    
    # --- SKIP LOGIC START ---
    # Check if the final output file already exists
    if os.path.exists(pdbqt_path):
        return "skipped"
    # --- SKIP LOGIC END ---
    
    try:
        # Step 1: PDB -> MOL2
        subprocess.run(
            ["obabel", "-ipdb", pdb_path, "-omol2", "-O", mol2_path],
            check=True, capture_output=True
        )
        
        # Step 2: MOL2 -> PDBQT
        subprocess.run(
            ["obabel", "-imol2", mol2_path, "-opdbqt", "-O", pdbqt_path, "-xh"],
            check=True, capture_output=True
        )
        return "success"
    except Exception:
        return "error"

def main():
    # Ensure output directories exist
    os.makedirs(os.path.join(OUTPUT_BASE, "mol2"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE, "pdbqt"), exist_ok=True)
    
    # List all PDB files to process
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdb")]
    total_files = len(files)
    
    print(f"Starting parallel conversion of {total_files} files...")
    
    # Updated summary to track skipped files
    summary = {"success": 0, "error": 0, "skipped": 0}
    
    # Use all available CPU cores for parallel processing[cite: 2]
    with ProcessPoolExecutor() as executor:
        # Map the function over the list of files with a progress bar[cite: 2]
        results = list(tqdm(executor.map(convert_molecule, files), total=total_files, desc="Converting"))

    # Aggregate results
    for res in results:
        summary[res] += 1

    print("\n--- Conversion Summary ---")
    print(f"Successfully converted: {summary['success']}")
    print(f"Skipped (already exist): {summary['skipped']}")
    print(f"Errors: {summary['error']}")

if __name__ == "__main__":
    main()