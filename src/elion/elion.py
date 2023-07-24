"""
# Entry point for all Elion calculations
# --------------------------------------
# All details should be in the input file (YAML format). 
# Here we will read this input file, and direct the calculations
# accordingly.
"""
import argparse
from pathlib import Path

from rdkit import Chem
import input_reader
import reward_function
import utils
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

    # Given a SMILES file, just calculate the properties
    if run_type == 'calculate_properties':
        smiles_file = Path(config['Control']['smiles_file'])
        output_file = Path(config['Control']['output_smi_file'])
        estimator = Estimators(config['Reward_function'])

        mols, smis = utils.read_smi_file(smiles_file)
        predictions = estimator.estimate_properties(mols)

        if config['Control']['verbosity'] > 0:
            utils.print_results(smis, predictions, header="PROPERTIES")
        else:
            utils.print_stats(predictions, header="STATISTICS", print_header=True)
            
        utils.save_smi_file(output_file, smis, predictions)

    # Generate new molecules
    elif run_type == 'generate':
        # Initialize
        output_file = Path(config['Control']['output_smi_file'])
        generator = Generator(config['Generator']).generator
        estimator = Estimators(config['Reward_function'])
        
        # Generate molecules
        smis = generator.generate_mols()
        mols = [ Chem.MolFromSmiles(x) for x in smis ] 

        # Calculates & prints Properties
        predictions = estimator.estimate_properties(mols)
        if config['Control']['verbosity'] > 0:
            utils.print_results(smis, predictions, header="PROPERTIES")
        else:
            utils.print_stats(predictions, header="STATISTICS", print_header=True)
        utils.save_smi_file(output_file, smis, predictions)
        
    elif run_type == 'bias_generator':
        generator = Generator(config['Generator']).generator
        estimator = Estimators(config['Reward_function'])
        generator.bias_generator(config['Control'], estimator)
        
    elif run_type == 'post_process':
        pass

    else:
        raise ValueError((f"Invalid run_type: {run_type}\n"
                          f"Valid options are: calculate_properties, generate, bias_generator, post_process")
                         )
    
if __name__ == '__main__':
    main()
    