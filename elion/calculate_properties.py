from pathlib import Path
from rdkit import Chem
import input_reader



# NOTE: At the moment, the code is passing molecule-by-molecule to the 
#       predictor. This need to be changed to pass a list of molecules
#       so that the predictor calculates properties for the whole list 
#       at once.

def predict_properties(mols,properties):
    """Calculates the properties for a list of molecules

    Args:
        mols ([RDKit ROMol]): List of RDKit ROMol objects
        properies (dict):  Dictionary with properties as keys and details
                           of predictor functions as values.
    Returns:
        Dict: Dictionary of properties as keys and predictions (floats) as values
    """
    pred = {}
    for _prop in properties.keys():
        pred[_prop] = []

    _mols = [].extend(mols)
    for _prop, _cls in properties.items():
        predictions = _cls.predict(mols)
        pred[_prop] = predictions
    return pred

def predict_rewards(n_mols, predictions, properties):
    """Calculates the rewards, given a dict of pre-calculated properties.

    Args:
        n_mols (int): number of molecules
        predictions (dict): Dictionary with properties as keys and lists of
                            predicted values as values.
        properties (_type_): Dictionary with properties as keys and details
                             of the property pbjects as values

    Returns:
        dict: Dict with property names as keys and rewards as values.
    """

    rew = {}

    for _prop, cls in properties.items():
        _values = predictions[_prop]
        rew[_prop] = cls.reward(_values)
    return rew

if __name__ == "__main__":
    #"""
    #    Reads a SMILES file, and calculates properties and rewards
    #    for each molecule in the file.
    #"""

    root_dir = Path().cwd()
    input_file = Path(root_dir,"elion/input_example.yml")
    config = input_reader.read_input_file(input_file)

    # Debug: Prints the confic dict
    if config['Control']['verbosity'] > 0:
        import pprint
        pprint.pprint(config)

    # Read SMILES file
    # Filters out invalid SMILES
    mols, smis = [], []
    smiles_file = Path(config['Control']['smiles_file'])

    if smiles_file.is_file():
        with open(smiles_file,'r') as smif:
            for line in smif.readlines():
                smi = line.split()[0]
                mol = Chem.MolFromSmiles(smi)
                
                if mol is not None:
                    mols.append(mol)
                    smis.append(smi)
    else:
        msg = ( "ERROR when reading config file:\n"
               f"Could not find the smiles_file <{smiles_file.absolute()}>.")
        quit(msg)
    
    # Calculates Properties and Rewards
    predictions = predict_properties(mols, config['Properties'])
    rewards = predict_rewards(len(mols), predictions,config['Properties'])  

    # Pretty-print results
    # ====================
    # For pretty-printing purposes, find the longest
    # SMILES string to print
    LENGTH_LIM = 0
    for smi in smis:
        if len(smi) > LENGTH_LIM:
            LENGTH_LIM = len(smi)
    if LENGTH_LIM > 30:
        LENGTH_LIM = 30

    # Prints Properties
    print( "\nProperties")
    print(f"{'#':>6s}  {'Molecule':{LENGTH_LIM+3}s}", end="")
    for prop, cls in config['Properties'].items():
        print(f"  {prop}", end="")
    print("")
    for ind, smi in enumerate(smis):
        CONT = "..." if len(smi) > LENGTH_LIM else "   "
        print(f"{ind:>6d}  {smi:{LENGTH_LIM}.{LENGTH_LIM}s}{CONT}", end="")

        for prop, cls in config['Properties'].items():
            title_len = len(prop)
            value = predictions[prop][ind]
            print(f"  {value:>{title_len}.2f}", end="")
        print("")

    # Prints Rewards
    print( "\nRewards")
    print(f"{'#':>6s}  {'Molecule':{LENGTH_LIM+3}s}", end="")
    for prop, cls in config['Properties'].items():
        print(f"  {prop}", end="")
    print("")
    
    for ind, smi in enumerate(smis):
        CONT = "..." if len(smi) > LENGTH_LIM else "   "
        print(f"{ind:>6d}  {smi:{LENGTH_LIM}.{LENGTH_LIM}s}{CONT}", end="")

        for prop, cls in config['Properties'].items():
            title_len = len(prop)
            reward = rewards[prop][ind]
            print(f"  {reward:>{title_len}.2f}", end="")
        print("")
