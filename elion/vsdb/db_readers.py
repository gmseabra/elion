"""
Collection of methods to read chemical databases into a Pandas Dataframe
"""
import pandas as pd
from rdkit import Chem #, DataStructs
from rdkit.Chem import PandasTools #, Draw, SaltRemover, Descriptors, AllChem
#from tqdm import trange, tnrange, tqdm_notebook, tqdm_pandas, tqdm
from tqdm.auto import tqdm
from vsdb import mol_properties

def read_drugbank_links_file(file):
    """
    Reads molecules from the 'links' file from DrugBank, keeping only molecules with valid SMILES. 
    In the process, it also adds a 'ROMol' column, and caonicalize the SMILES. 
    
    """
    with open(file, 'r') as f:
        df = pd.read_csv(file)
        print(f"The file has a total of {df.shape[0]} entries, ", end="")
        df.dropna(subset=['SMILES'], inplace=True)
        print(f"from which SMILES strings were present in {df.shape[0]}.")
        
        # Adds the 'ROMol' column, and canonicalize the SMILES
        df = mol_properties.add_mol_from_smiles(df)
        df = mol_properties.add_canonical_SMILES(df)
        
        return df
    
    
