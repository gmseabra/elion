from . import preprocess, mol_properties, enumerate_mols
#import preprocess, mol_properties, enumerate_mols

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem
from tqdm import tqdm, tqdm_pandas
from tqdm.auto import tqdm

## Suppress RDKit output
from rdkit import RDLogger
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)

def prepare_2D_db(mol_db, min_WT=50, max_WT=500, max_stereoisomers=4):
    """
    Perform all steps to prepare a database of 2D structures

    Args:
        df (Pandas DataFrame): A dataframe with the molecules. Must contain one column
                               ("SMILES") with the SMILES codes for the molecules.
        
        Filters: (set to -1 for no filtering)
            min_WT (int, optional): Minimum molecular weight for the molecules. Defaults to 50.
            max_WT (int, optional): Maximum molecular weight. Defaults to 500.
            max_stereoisomers (int, optional): Maximum number of stereoisomers. Defaults to 4.

    Returns:
        Tuple(mol_db, dup)[Pandas DataFrames]: 
                mol_db: The DataFrame with the structures in 3D format, ready to be saved to file
                dup: A DataFrame with the molecules from the original df considered to be duplicates.
    """
    
    # Preprocess the database to remove salts, invalid molecules, etc.
    mol_db, dup = preprocess.preprocess_db(mol_db)

    # Filter the molecules
    if min_WT != -1 and max_WT != -1:
        mol_db = mol_properties.filter_by_MolWT(mol_db, min_WT, max_WT)

    # Expand stereoisomers (only if stereo info is not present in SMILES)
    if max_stereoisomers != -1:
        mol_db = mol_properties.filter_by_max_stereoisomers(mol_db,max_stereoisomers)
        mol_db = enumerate_mols.enumerate_stereoisomers(mol_db)

    return mol_db, dup


def prepare_3D_db(mol_db, min_WT=50, max_WT=500, max_stereoisomers=4):
    """
    Perform all steps to prepare a database of 3D structures

    Args:
        df (Pandas DataFrame): A dataframe with the molecules. Must contain one column
                               ("SMILES") with the SMILES codes for the molecules.
        min_WT (int, optional): Minimum molecular weight for the molecules. Defaults to 50.
        max_WT (int, optional): Maximum molecular weight. Defaults to 500.
        max_stereoisomers (int, optional): Maximum number of stereoisomers. Defaults to 4.

    Returns:
        Tuple(mol_db, dup)[Pandas DataFrames]: 
                mol_db: The DataFrame with the structures in 3D format, ready to be saved to file
                dup: A DataFrame with the molecules from the original df considered to be duplicates.
    """
    
    # First, generate a database of 2D structures
    mol_db, dup = prepare_2D_db(mol_db, min_WT, max_WT, max_stereoisomers)

    # Now, we need to generate 3D structures for all molecules in the Database
    # First, add Hydrogens
    tqdm.pandas(desc="Adding Hydrogens")
    mol_db.ROMol = mol_db.ROMol.progress_apply(Chem.AddHs)

    # Now, generate the 3D structures
    tqdm.pandas(desc="Generating 3D structures")
    scratch = mol_db.ROMol.progress_apply(AllChem.EmbedMolecule)

    return mol_db, dup


