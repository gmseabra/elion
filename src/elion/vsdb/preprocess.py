"""
Preprocesses the molecules to remove invalid entries
"""

import pandas as pd
import molvs
from rdkit import Chem
from rdkit.Chem import PandasTools, Draw, SaltRemover, MolStandardize
from tqdm import trange, tnrange, tqdm_notebook
from tqdm.auto import tqdm
import time

def only_valid_atoms(mol):
    """
    Checks that a molecule only has valid atoms.
    """
    valid_atoms=['C','N','O','H','S','P','As','Se','F','Cl','Br','I']


    if mol is None or mol.GetNumAtoms() < 1:
        return False
    else:
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in valid_atoms:return False
    return True    

def drop_invalid_mols(df):
    """
    Delete DataFrame rows with invalid atoms
    """
    df['valid'] = df['ROMol'].progress_apply(only_valid_atoms)
    return df[(df['valid'])].copy().drop(['valid'], axis=1)

def standardize_molecule(mol, canonicalize_tautomer=False):
    """
    Completely standardizes a molecule to be used in this pipeline. The full process includes:,
    1. Standardize using MolVS standardizer
       The standardization process consists of the following stages:
       (See [here](https://molvs.readthedocs.io/en/latest/api.html#molvs-standardize).)
       1.1. RDKit RemoveHs(), 
       1.2. RDKit SanitizeMol(): Kekulize, check valencies, 
                                 set aromaticity, conjugation and hybridization
       1.3. MetalDisconnector:  breaking covalent bonds between metals and organic 
                                atoms under certain conditions.
       1.4. Normalizer: Apply a series of Normalization transforms to correct functional
                        groups and recombine charges. Each transform is repeatedly applied 
                        until no further changes occur.
       1.5. Reionizer: Fix charges and reionize a molecule such that the strongest acids ionize first.
       1.6. RDKit AssignStereochemistry().
    2. Remove salts
    3. Remove mixture fragments (keep the largest organic fragment)
    4. Remove charges (adding / removing Hs)
    5. Canonicalize tautomer: Returns just one 'canonical' tautomer (Only if canonicalize_tautomer=True)
    
    INPUT : 'mol': a RDkit ```mol``` object
    OUTPUT: a new RDKit ```mol``` object that has been standaridized.
    """

    # Think about this: 
    # rdkit.Chem.PandasTools.RemoveSaltsFromFrame(frame, molCol='ROMol')
    # http://rdkit.org/docs/source/rdkit.Chem.PandasTools.html#rdkit.Chem.PandasTools.RemoveSaltsFromFrame
    
    std = molvs.Standardizer(prefer_organic=True)
    taut = molvs.tautomer.TautomerCanonicalizer()
    remover = Chem.SaltRemover.SaltRemover()
    unchg = molvs.charge.Uncharger()
    unmix = molvs.fragment.LargestFragmentChooser(prefer_organic=True)

    # Standardize the molecule
    this_mol = remover.StripMol(mol, dontRemoveEverything=True)      # Remove salts

    # Sometimes, an entry may have just a salt, such as LiCl. In this case,
    # removing salts retunds an empty molecule. So, we only keeo going
    # if the returned molecule is valid.

    #if this_mol.GetNumAtoms() > 1:
    this_mol = unmix.choose(this_mol)          # Remove fragments / mixtures
    this_mol = std.standardize(this_mol)       # Standardize the mol
    this_mol = unchg.uncharge(this_mol)        # Remove charges

    if canonicalize_tautomer:
        this_mol = taut.canonicalize(this_mol)     # Canonicalize the tautomers
    return this_mol


# -----------------------------------------------------------------------------
#
#                               DATA PIPELINE
#
# -----------------------------------------------------------------------------
def preprocess_db(df,smiles_col="SMILES", mol_col="ROMol", standardize_ROMol=True, verbose=False):
    """ 
    Receives a Pandas Dataframe containing molecules, and processes the 
    whole pipeline in the dataframe.
    
    Arguments:
      - smiles_col: The name of the SMILES column. Default: SMILES. 
                    ==> This column *MUST* be present in the dataframe.
      - mol_col: THe name of the RDKit ROMol column. Default:"ROMol" 

    The pipeline consists in:
    1. Add a 'ROMol' colummn to the DataFrame
    1. Standadize the molecules applying `standardize_molecule` to each molecule.
        1.1. Standardize format
        1.2. Removes salts
        1.3. Removes fragments / mixtures
        1.4. Removes charges bu adding / removing H
        1.5. Canonicalize tautomers
    2. Generate InChI Keys
    3. Remove molecules with invalid atoms
    4. Remove duplicates based on InChI keys
    5. Reset DataFrame index
    
    Returns:
        df: The new processed DataFrame, with no duplicates, and the extra columns:
            - mol
            - InChI Key
        dup: A DataFrame with only the duplicate entries (all copies). This is included
             in case the user needs to look at it later. In *most* cases, the duplication
             is just the same compounds with different counter ions.
    """

    assert(isinstance(df, pd.DataFrame)), "This function must receive a Pandas DataFrame as argument."
    assert(smiles_col in df.columns), "The DataFrame must have a SMILES column."
    
    _start = time.time()

    # Remove possible invalid (nan) SMILES
    df.dropna(subset=[smiles_col], inplace=True)

    # There is a bug int he current RDKit in that the ROMol object created
    # lacks some stereo information, which created problems later.
    # For now, the solution is removing this column and recriating it from 
    # the SMILES column.
    if standardize_ROMol:
        if mol_col in df.columns:
            df = df.drop(columns=mol_col)

        PandasTools.AddMoleculeColumnToFrame(df,smilesCol=smiles_col,molCol=mol_col)

    # Remove molecules with invalid atoms
    tqdm.pandas(desc="Removing molecules with invalid atoms")
    df = drop_invalid_mols(df)

    # Standardize
    tqdm.pandas(desc="Standardizing")
    df[mol_col] = df[mol_col].progress_apply(standardize_molecule)

    # Reset the SMILES column to conform to the new ROMol contents,
    # which may have changed after standardization
    tqdm.pandas(desc="Resetting SMILES")
    df = df.drop(columns=smiles_col)
    df[smiles_col] = df[mol_col].progress_apply(Chem.MolToSmiles)

    # Create InChI Keys
    tqdm.pandas(desc="Generating InChI Keys")
    df['InChI Key'] = df[mol_col].progress_apply(Chem.MolToInchiKey, options="/FixedH")

    # Remove duplicates
    if verbose: print("Looking for duplicates...")
    n_duplicates = df.shape[0] - len(df['InChI Key'].unique())
    dup=pd.DataFrame()
    if n_duplicates > 1:
        if verbose: print(f"{n_duplicates} duplicates found. Removing...")
        dup = df[ df.duplicated(subset=['InChI Key'], keep=False) ]
        df.drop_duplicates(subset=['InChI Key'], keep='first', inplace=True)
    if verbose: print("Done")
    
    # Reset DataFrame Indexes
    df.reset_index(drop=True, inplace=True)
    
    if verbose: print(f"Total time: {(time.time() - _start):0.3} seconds")
    # Done!
    return df, dup
