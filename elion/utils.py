"""
Just some utilities used by various files
"""

__author__ = 'Gustavo Seabra'
__copyright__ = 'Copyright 2020, Me'
__credits__ = ['Gustavo Seabra']
__license__ = 'None Yet'
__version__ = '0.1'
__maintainer__ = 'Gustavo Seabra'
__email__ = 'seabra _at_ cop.ufl.edu'
__status__ = 'Prototype'

# --Python
from typing import List, Dict
import pandas as pd
import numpy as np

# Fingerprinting
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# Add Fingerprints
def get_morganfingerprints(mol):
    """
    Given a mol object, returns its Morgan Fingerprint (ECFP4)

    Args:
        mol (RDKit Mol): The molecule to be fingerprinted

    Returns:
        np.array: Fingerprint
    """
    
    fp = GetMorganFingerprintAsBitVect(mol,2)
    return np.array(list(map(int,fp.ToBitString())))

def get_fingerprint_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        print("INVALID SMILES RECEIVED:", smi)
        fp = np.zeros(2048,dtype=int)
    else:
        fp = RDKFingerprint(mol)
        fp = np.array(list(map(int,fp.ToBitString())))
    return fp

def get_rdkfingerprint(mol):
    """
    Given a mol object, returns its RDKit Fingerprint

    Args:
        mol (RDKit Mol): The molecule to be fingerprinted

    Returns:
        np.array: Fingerprint
    """
    fp = RDKFingerprint(mol)
    return np.array(list(map(int,fp.ToBitString())))

def add_fingerprint(frame: pd.DataFrame, fp: str='rdkit') -> pd.DataFrame:
    """
    Given a Pandas DataFrame with a SMILES column and molecules in rows,
    returns the same DataFrame, now with the 'FINGERPRINT' column, containing
    the fingerprint for the molecule.

    Args:
        frame (pd.DataFrame): DataFrame containing a SMILES column
        fp (str, optional): ['rdkit'] The fingerprint type to use. Defaults to 'rdkit'.

    Returns:
        pd.DataFrame: The same DataFrame, now with the added 'FINGERPRINT' column.
    """

    fp = fp.lower()
    assert (fp in ['rdkit']), "Fingerprint type not available"

    if fp == 'rdkit':
        if 'ROMol' not in frame.columns:
            PandasTools.AddMoleculeColumnToFrame(frame,smilesCol="SMILES")

        print("Generating RDKit Fingerprints... ", end='')
        frame['FINGERPRINT'] = frame['ROMol'].apply(get_rdkfingerprint)
        print("Done")
    return frame

def print_dict(dict_to_print:dict):
    """Pretty-prints a dictionary containing properties for 
       molecules. The dictionary must have the property names 
       for keys, and lists of values for values:
           {prop1:[value1,vlue2,...],
            prop2:[value1,vlue2,...],
            ...}
    Args:
        dict_to_print (dict): dict_to_print
    """
    n_samples = 0
    print("Molecule  ", end="")
    for prop in dict_to_print.keys():
        print(f"{prop[:15]:>20s}", end="")
        n_samples = len(dict_to_print[prop])
    print("\n",end="")
    print(f"{'-'*10}{'-'*20*len(dict_to_print)}")
    for sample in range(n_samples):
        print(f"{sample+1:8d}  ",end="")
        for prop,val in dict_to_print.items():
            print(f"{val[sample]:>20f}", end="")
        print("\n", end="")
    return

# Progress Table
def print_progress_table(reward_properties:Dict, predictions:Dict, rewards:Dict):
    print("-"*55)
    print(f"|  {'CURRENT PROPERTY AVERAGES':^50} |")
    print("-"*55)
    print(f"|  {'PROPERTY':<25s} |  {'VALUE':>8s}  |  {'REWARD':>8s} |")
    print("-"*55)
    for prop in reward_properties.keys():
        line= (f"|  {prop:<25s} "
            f"|  {np.average(predictions[prop]):8.3f}  "
            f"|  {np.average(rewards[prop])    :8.3f} |")
        print(line)
    print("-"*55)
    return

# Save a SMILES file
def save_smi_file(filename, smiles, predictions):
    """ Saves the molecules and predictions in a .smi file
    """
    with open(filename,'w') as output:
       
        # ----- Print predictions -------------------------------------------------------
        line = f"SMILES,Name,{','.join([str(x) for x in predictions.keys()])}\n"
        output.write(line)
        i=1
        for smi, *pred in zip(smiles, *[predictions[x] for x in predictions]):
            line = f"{smi}, Gen-{i:05d}," + ','.join([f"{x:.2f}" for x in pred]) +  "\n"
            output.write(line)
            i += 1


