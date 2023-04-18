from pathlib import Path
from rdkit import Chem
import input_reader
from properties.Estimators import Estimators

if __name__ == "__main__":
    #"""
    #    Reads a SMILES file, and calculates properties and rewards
    #    for each molecule in the file.
    #"""

    root_dir = Path().cwd()
    input_file = Path(root_dir,"elion/input_example.yml")
    config = input_reader.read_input_file(input_file)

    # Debug: Prints the config dict
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

    estimator = Estimators(config['Properties'])
    predictions = estimator.estimate_properties(mols)
    rewards = estimator.estimate_rewards(predictions)

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
    for prop, cls in predictions.items():
        print(f"  {prop}", end="")
    print("")
    for ind, smi in enumerate(smis):
        CONT = "..." if len(smi) > LENGTH_LIM else "   "
        print(f"{ind:>6d}  {smi:{LENGTH_LIM}.{LENGTH_LIM}s}{CONT}", end="")

        for prop, cls in predictions.items():
            title_len = len(prop)
            value = predictions[prop][ind]
            print(f"  {value:>{title_len}.2f}", end="")
        print("")

    # Prints Rewards
    print( "\nRewards")
    print(f"{'#':>6s}  {'Molecule':{LENGTH_LIM+3}s}", end="")
    for prop, cls in rewards.items():
        print(f"  {prop}", end="")
    print("")
    
    for ind, smi in enumerate(smis):
        CONT = "..." if len(smi) > LENGTH_LIM else "   "
        print(f"{ind:>6d}  {smi:{LENGTH_LIM}.{LENGTH_LIM}s}{CONT}", end="")

        for prop, cls in rewards.items():
            title_len = len(prop)
            reward = rewards[prop][ind]
            print(f"  {reward:>{title_len}.2f}", end="")
        print("")
