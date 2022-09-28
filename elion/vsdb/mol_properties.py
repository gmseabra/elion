"""
Utilities to 
(a) Add properties to database
(b) filter databases according to some property
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors
from tqdm.auto import tqdm



#########
# Filters
#########

def filter_by_MolWT(df, min_WT=0, max_WT=500):
    """
    Filters the dataframe removing molecules with moleular weight outside the interval [min_WT,maxWT].
    """

    if 'MolWT' not in df.columns:
        df = add_MolWT(df)

    below = df.index[( df['MolWT'] < min_WT )]
    above = df.index[( df['MolWT'] > max_WT )]
    to_drop = below.append(above)
    valid = df.shape[0] - len(to_drop)
    
    print(f"Received DataFrame with {df.shape[0]} entries, ", end="")
    print(f"from which {valid} have {min_WT} < WT < {max_WT} ({valid/df.shape[0]:0.2%}).")
    return df.drop(to_drop).reset_index(drop=True)

def filter_by_max_stereoisomers(df, max_stereoisomers=10):
    """
    Filters the dataframe removing molecules with more than 'max_stereoisomers' stereoisomers.
    """
    print(f"Received DataFrame with {df.shape[0]} entries, ", end="")

    opts = Chem.EnumerateStereoisomers.StereoEnumerationOptions(tryEmbedding=True, unique=True, maxIsomers=max_stereoisomers, onlyUnassigned=True)
    if 'n_stereo' not in df.columns:
        df['n_stereo'] = df['ROMol'].apply(Chem.EnumerateStereoisomers.GetStereoisomerCount, opts)
    to_drop = df.index[(df['n_stereo'] > max_stereoisomers)]
    valid = df.shape[0] - len(to_drop)
    print(f"from which {valid} have up to {max_stereoisomers} stereoisomers ({valid/df.shape[0]:0.2%}).")
    return df.drop(to_drop).drop(columns=['n_stereo']).reset_index(drop=True)


############
# Properties
############

def add_InChI(df):
    """
    Adds InChI hashes column. 
    
    Takes a Pandas DataFrame containing a 'ROMol' column and creates
    a column containing 'InChI'from the 'ROMol' values.
    If 'InChI' are already present, they are overwritten by 
    the new 'InChI'.
    """
    tqdm.pandas(desc="Generating InChI")
    df['InChI']   = df['ROMol'].progress_apply(Chem.inchi.MolToInchi)
    tqdm.pandas(desc="")
    return df

def add_InChI_keys(df):
    """
    Adds InChI Keys column. 
    
    Takes a Pandas DataFrame containing a 'ROMol' column and creates
    a column containing 'InChI_keys' from the 'ROMol' values.
    If 'InChI_keys' are already present, they are overwritten by 
    the new 'InChI_keys'.
    """
    tqdm.pandas(desc="Generating InChI_keys")
    df['InChIKey']   = df['ROMol'].progress_apply(Chem.inchi.MolToInchiKey)
    tqdm.pandas(desc="")
    return df

def add_MolWT(df):
    """
    Adds a MolWT (Molecular Weight) column. 
    
    Takes a Pandas DataFrame containing a 'ROMol' column and creates
    a column containing the molecular weight.
    If 'MolWT' are already present, they are overwritten by 
    the new 'MolWT'.
    """
    tqdm.pandas(desc="Calculating Molecular Weights")
    df['MolWT']   = df['ROMol'].progress_apply(Chem.Descriptors.MolWt)
    tqdm.pandas(desc="")
    return df

def add_LogP(df):
    """
    Adds a LogP column. 
    
    Takes a Pandas DataFrame containing a 'ROMol' column and creates
    a column containing 'LogP' values from the 'ROMol' values.
    If 'LogP' are already present, they are overwritten by 
    the new calculated 'LogP'.
    """
    tqdm.pandas(desc="Calculating LogP")
    df['LogP']     = df['ROMol'].progress_apply(Chem.Descriptors.MolLogP)
    tqdm.pandas(desc="")
    return df

def add_mol_from_smiles(df, smiles_col="SMILES", mol_col="ROMol"):
    """
    Takes a dataframe with a <smiles_col> 'SMILES' column, and: 
    1. Adds a <,ol_col> 'ROMol' column (RDKit molecule), based on the SMILES column.
    2. If the SMILES could not be converted to ROMol, remove that
       molecule from the dataframe. Then,
    3. Reset the index from the DataFrame.
    """
    if mol_col not in df.columns:        
        print(f"Received DataFrame with {df.shape[0]} entries, ", end="")
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smiles_col, molCol=mol_col)
        df.dropna(subset=[mol_col], inplace=True)
        print(f"from which {df.shape[0]} could be converted to RDKit.mol objects.")
        df = df.reset_index(drop=True)
    else:
        print("DataFram already contains the 'ROMol' column. Nothing to be done.")
    return df

def add_canonical_SMILES(df, smiles_col="SMILES", mol_col="ROMol"):
    """
    Takes a Pandas DataFrame containing a 'ROMol' column and creates
    a column containing canonical 'SMILES' from the 'ROMol' values.
    If 'SMILES' are already present, they are overwritten by 
    the new canonical 'SMILES'.
    """
    tqdm.pandas(desc="Generating canonical SMILES")
    df[smiles_col]   = df[mol_col].progress_apply(Chem.MolToSmiles)
    return df

