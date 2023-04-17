from pathlib import Path
from rdkit import Chem
import input_reader



# NOTE: At the moment, the code is passing molecule-by-molecule to the 
#       predictor. This need to be changed to pass a list of molecules
#       so that the predictor calculates properties for the whole list 
#       at once.

def predict_properties(molecules,properties):
    """Calculates the properties for a list of molecules

    Args:
        molecules ([str]): List of SMILES strings
        properies (dict):  Dictionary with properties as keys and details
                           of predictor functions as values.
    Returns:
        Dict: Dictionary of properties as keys and predictions (floats) as values
    """
    pred = {}
    for _prop in properties.keys():
        pred[_prop] = []

    for _mol in molecules:
        for _prop, _cls in properties.items():
            _value = _cls.predict(Chem.MolFromSmiles(_mol))
            pred[_prop].append(_value)
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
    for _prop in properties.keys():
        rew[_prop] = []

    # Predicts rewards molecule-wise
    for _mol in range(n_mols):
        for _prop, cls in properties.items():
            _value = predictions[_prop][_mol]
            rew[_prop].append(cls.reward(_value))
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
    example_mols = []
    smiles_file = Path(config['Control']['smiles_file'])

    if smiles_file.is_file():
        with open(smiles_file,'r') as smif:
            for line in smif.readlines():
                example_mols.append(line.split()[0])
    else:
        msg = ( "ERROR when reading config file:\n"
               f"Could not find the smiles_file <{smiles_file.absolute()}>.")
        quit(msg)
    
    # Calculates Properties and Rewards
    predictions = predict_properties(example_mols, config['Properties'])
    rewards = predict_rewards(len(example_mols), predictions,config['Properties'])  

    # Pretty-print results 

    # For pretty-printing purposes, find the longest
    # SMILES string to print
    length_lim = 0
    for mol in example_mols:
        if len(mol) > length_lim:
            length_lim = len(mol)
    if length_lim > 30:
        length_lim = 30

    # Prints Properties
    print( "\nProperties")
    print(f"{'#':>6s}  {'Molecule':{length_lim+3}s}", end="")
    for prop, cls in config['Properties'].items():
        print(f"  {prop}", end="")
    print("")
    for ind, mol in enumerate(example_mols):
        print(f"{ind:>6d}  {mol:{length_lim}.{length_lim}s}...", end="")

        for prop, cls in config['Properties'].items():
            title_len = len(prop)
            value = predictions[prop][ind]
            print(f"  {value:>{title_len}.2f}", end="")
        print("")

    # Prints Rewards
    print( "\nRewards")
    print(f"{'#':>6s}  {'Molecule':{length_lim+3}s}", end="")
    for prop, cls in config['Properties'].items():
        print(f"  {prop}", end="")
    print("")
    for ind, mol in enumerate(example_mols):
        print(f"{ind:>6d}  {mol:{length_lim}.{length_lim}s}...", end="")

        for prop, cls in config['Properties'].items():
            title_len = len(prop)
            reward = rewards[prop][ind]
            print(f"  {reward:>{title_len}.2f}", end="")
        print("")
