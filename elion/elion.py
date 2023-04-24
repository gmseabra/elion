from pathlib import Path
from rdkit import Chem
import input_reader
from utils import print_results
from properties.Estimators import Estimators


# This will be the entry point for all Elion calculations
# All details should be in the input file (YAML format). 
# Here we will read this input file, and direct the calculations
# accordingly.

if __name__ == "__main__":
    #"""
    #"""

    import argparse

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
    root_dir = Path().cwd()
    config = input_reader.read_input_file(input_file)

    # Debug: Prints the config dict
    if config['Control']['verbosity'] > 0:
        import pprint
        pprint.pprint(config)

    # Calculation Type
    run_type = config['Control']['run_type']
    if run_type == 'calculate_properties':
        import reward_function
        reward_function.calculate(config)
    elif run_type == 'generate':
        pass
    elif run_type == 'bias_generator':
        pass
    elif run_type == 'post_process':
        pass
    
   