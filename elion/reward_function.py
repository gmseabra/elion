from pathlib import Path
from rdkit import Chem
import input_reader
from utils import print_results
from properties.Estimators import Estimators


def calculate(config):
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

    estimator = Estimators(config['Reward_function'])
    predictions = estimator.estimate_properties(mols)
    rewards = estimator.estimate_rewards(predictions)

    # Prints results
    print_results(smis, predictions, header="PROPERTIES")
    print_results(smis, rewards, header="REWARDS")
    
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
    calculate(config)
