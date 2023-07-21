"""
# Entry point for all Elion calculations
# --------------------------------------
# All details should be in the input file (YAML format). 
# Here we will read this input file, and direct the calculations
# accordingly.
"""
import argparse

from rdkit import Chem
import input_reader
import reward_function
from utils import print_results, print_stats, save_smi_file
from generators.Generator import Generator
from properties.Estimators import Estimators

def main():
    """Elion: A Workflow for the Design of Small Molecules with Desired Properties
    """

    #-- Command line arguments
    parser = argparse.ArgumentParser(
        description=''' Entry point for all elion calculations.
                    ''')

    parser.add_argument('-i','--input_file',
                        help='Path to the input file',
                        default='./input.yml')

    args = parser.parse_args()
    input_file = args.input_file
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
        # Initialize
        generator = Generator(config['Generator']).generator
        estimator = Estimators(config['Reward_function'])
        
        # Generate molecules
        smis = generator.generate_mols()
        mols = [ Chem.MolFromSmiles(x) for x in smis ] 

        # Calculates & prints Properties
        predictions = estimator.estimate_properties(mols)
        if config['Control']['verbosity'] > 0:
            print_results(smis, predictions, header="PROPERTIES")
        else:
            print_stats(predictions, header="STATISTICS", print_header=True)
        save_smi_file(config['Control']['smiles_file'], smis, predictions)
        
        # Calculates & prints Rewards
        #rewards = estimator.estimate_rewards(predictions)
        #print_results(smis, rewards, header="REWARDS")

    elif run_type == 'bias_generator':
        generator = Generator(config['Generator']).generator
        estimator = Estimators(config['Reward_function'])
        generator.bias_generator(config['Control'], estimator)
        
    elif run_type == 'post_process':
        pass
    
if __name__ == '__main__':
    main()
    