from pathlib import Path
from rdkit import Chem
import input_reader
from utils import print_results
from properties.Estimators import Estimators


def calculate_properties(config):
    # Read SMILES file
    # Filters out invalid SMILES

    
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
