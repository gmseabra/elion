from pathlib import Path
import input_reader
from generators.Generator import Generator
import reward_function
from utils import print_results
from properties.Estimators import Estimators
from rdkit import Chem


# Entry point for all Elion calculations
# --------------------------------------
# All details should be in the input file (YAML format). 
# Here we will read this input file, and direct the calculations
# accordingly.

if __name__ == "__main__":
    """
        main
    """

    #import argparse

    #-- Command line arguments
    #parser = argparse.ArgumentParser(
    #    description=''' Entry point for all elion calculations.
    #                ''')

    #parser.add_argument('-i','--input_file',
    #                    help='Path to the input file',
    #                    default='./input.yml')

    #args = parser.parse_args()
    #input_file = args.input_file
    input_file = "/home/seabra/work/li/elion_dev/tests/modular/input.yml"
    #--
    config = input_reader.read_input_file(input_file)

    # Debug: Prints the config dict
    if config['Control']['verbosity'] > 0:
        import pprint
        pprint.pprint(config)

    # Calculation Type
    run_type = config['Control']['run_type']
    if run_type == 'calculate_properties':
        reward_function.calculate(config)
    elif run_type == 'generate':
        generator = Generator(config['Generator'])
        smis = generator.generate_mols()
        mols = [ Chem.MolFromSmiles(x) for x in smis ] 

        # Calculates Properties and Rewards
        estimator = Estimators(config['Reward_function'])
        predictions = estimator.estimate_properties(mols)
        rewards = estimator.estimate_rewards(predictions)

        # Prints results
        print_results(smis, predictions, header="PROPERTIES")
        print_results(smis, rewards, header="REWARDS")

    elif run_type == 'bias_generator':
        pass
    elif run_type == 'post_process':
        pass
    
   