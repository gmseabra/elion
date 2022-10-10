#!/usr/bin/env python
__description__="""
    Simply generate new molecule SMILES using
    some generator.

    This code uses the ReLeaSE program from:
    https://github.com/isayev/ReLeaSE
"""

import time
import pandas as pd

from generators.release.release_generator import release_smiles_generator, print_torch_info

# Scikit-learn Model
#from sklearn.ensemble import RandomForestRegressor
from joblib import load

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

# Local 
from utils import get_fingerprint_from_smiles
from properties import properties_and_rewards
import input_reader

#-------------------------------------------
def generate_new(generator, n_to_generate):
    """
    Generates a sample of n_to_generate molecules using the provided generator.

    Args:
        generator (generator): Generator object to use. Must have a 'generate' function that
                               gets as argument a number and returns that number of molecules. 
        n_to_generate (int): Number of molecules to generate
    
    Returns:
        List of <n_to_generate> smiles strings.
    """

    generated = generator.generate(n_to_generate, verbose=1)

    return generated

#-------------------------------------------

if __name__ == "__main__":

    begin = time.time()
    # ----------------------
    # Command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description=''' Generates molecules using a ML generator. If no generator is
                        provided, uses the original ReLeaSE pre-trained generator.
                    ''')

    # REQUIRED ARGUMENTS: n_to_generate and generator
    parser.add_argument('-n', '--n_to_generate',
                        help="Number of molecules to generate.",
                        type=int,
                        required=True)

    parser.add_argument('-i','--input',
                        help='YAML file for input',
                        required=True)

    # Output options
    parser.add_argument('-o', '--out',
                        help='Path to output file containig the generated SMILES',
                        default=None)

    parser.add_argument('-s', '--sdf',
                        help='Path to SDF output file containig the generated 3D molecules',
                        default=None)

    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Option to print extra information to output',
                        default=False)

    args = parser.parse_args()

    # Parse input
    n_to_generate = args.n_to_generate
    input_file = args.input
    smiles_file = args.out
    sdf_file = args.sdf
    debug = args.debug

    # Read input file
    cfg = input_reader.read_input_file(input_file)

    # Rewards to use:
    reward_properties = cfg['reward_properties']

    # Generator
    generator = release_smiles_generator()
    initial_state = cfg['generator']['initial_state']
    print(f"\nLoading unbiased generator from file {initial_state} ... ", end='')
    generator.load_model(initial_state)
    print("Done.")

    # Activity Predictor
    # Docking Score Predictor
    model_type = reward_properties['prob_active']['model_type']
    model_file = reward_properties['prob_active']['model_file']

    if  model_type == "scikit-learn":
        # Model created with Scikit-Learn, e.g. a random-forest
        from sklearn.ensemble import RandomForestClassifier
        from joblib import load
        print(f"\nInitializing RF Model from file {model_file} ... ", end='')
        activity_model = load(model_file).set_params(n_jobs=1)
        print("Done.")

    elif model_type == "CHEMBERT":
        # CHEMBERT model
        from properties.activity.CHEMBERT.chembert import chembert_model

        print(f"\nInitializiing CHEMBERT with state from file {model_file} ... ", end='')
        activity_model = chembert_model(model_file)
        print("Done.")

    reward_properties['prob_active']['predictor'] = activity_model

    # Scaffold
    if 'scaffold_match' in reward_properties.keys():
        template_smiles_file = reward_properties['scaffold_match']['scaffold_file']
        print(f"\nLoading scaffold from {template_smiles_file}. ")
        with open(template_smiles_file,'r') as tf:
            template = tf.readline().strip() 
        template = Chem.MolFromSmarts(template)
        reward_properties['scaffold_match']['scaffold'] = template

        # This prints info, but also forces the info about rings to be calculated
        print(f"Atom  AtNum  InRing?  Arom?")
        for idx, atom in enumerate(template.GetAtoms()):
            print(f"{idx:>4d}  {atom.GetAtomicNum():5d}  {str(atom.IsInRing()):>7}  {str(atom.GetIsAromatic()):>5}")
        reward_properties['scaffold_match']['scaffold'] = template

    print("*"*80, flush=True)

    # Organize the results in a friendly DataFrame
    results_df = pd.DataFrame()

    # Generates the molecules
    generated = generate_new(generator, n_to_generate)
    results_df["SMILES"] = generated

    # Gets properties
    properties = properties_and_rewards.estimate_properties_parallel(generated, reward_properties)

    for prop, value in properties.items():
        results_df[prop] = value

    #All done, prints the results
    if debug: 
        print(results_df.to_string(index=False))

    if smiles_file is not None:
        print(f"\nSaving results to file: <<{smiles_file}>>")
        results_df.to_csv(smiles_file, index=False)

    elapsed = time.time() - begin
    print(f"\nElapsed time: {elapsed} seconds.")
    print("Execution finished.")
