"""
# Entry point for all Elion calculations
# --------------------------------------
# All details should be in the input file (YAML format). 
# Here we will read this input file, and direct the calculations
# accordingly.
"""
# ── Import-path pin (added) ────────────────────────────────────────────────
# The box has multiple copies of the `properties` package (e.g.
# src/TS/properties and src/elion/properties). Only src/elion/properties has
# the GPU-enabled ChemBERT. Pin THIS file's directory (src/elion) to the FRONT
# of sys.path so `import properties...` and `from generators...` always resolve
# to the copies that live next to this elion.py — making the imported ChemBERT
# deterministic regardless of PYTHONPATH ordering.
import os as _os, sys as _sys
# Resolve this file's directory robustly. Under the TS wrapper, elion.py is run
# via exec() with __file__='elion.py' (relative), so abspath() uses cwd — which
# the launcher sets to src/elion. Guard all three cases.
try:
    _ELION_DIR = _os.path.dirname(_os.path.abspath(__file__))
except Exception:
    _ELION_DIR = ""
# If that didn't yield a dir containing the properties package, fall back to cwd,
# then to the canonical absolute path.
if not _ELION_DIR or not _os.path.isdir(_os.path.join(_ELION_DIR, "properties")):
    _cwd = _os.getcwd()
    if _os.path.isdir(_os.path.join(_cwd, "properties")):
        _ELION_DIR = _cwd
    elif _os.path.isdir("/home/huangzihang/repos/elion/src/elion/properties"):
        _ELION_DIR = "/home/huangzihang/repos/elion/src/elion"
if _ELION_DIR:
    if _ELION_DIR in _sys.path:
        _sys.path.remove(_ELION_DIR)
    _sys.path.insert(0, _ELION_DIR)
    print(f"[ELION] sys.path pinned to: {_ELION_DIR}", flush=True)
# ───────────────────────────────────────────────────────────────────────────
import argparse
from pathlib import Path

from rdkit import Chem
import input_reader
import reward_function
import utils
from generators.Generator import Generator
from properties.Estimators import Estimators


def calculate_properties(config):
    """Given a SMILES file, calculate the properties of the molecules.	"""	
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
    
def generate_mols(config):
    """Generate new molecules"""	
    # Initialize
    output_file = Path(config['Control']['smiles_file'])
    generator = Generator(config['Generator']).generator
    estimator = Estimators(config['Reward_function'])
    
    # Generate molecules
    mols = generator.generate_mols()
    smis = [Chem.MolToSmiles(mol) for mol in mols]
    
    # Calculates & prints Properties
    predictions = estimator.estimate_properties(mols)
    if config['Control']['verbosity'] > 0:
        utils.print_results(smis, predictions, header="PROPERTIES")
    else:
        utils.print_stats(predictions, header="STATISTICS", print_header=True)
    utils.save_smi_file(output_file, smis, predictions)

def bias_generator(config):
    """Biases a Generator"""
    generator = Generator(config['Generator']).generator
    estimator = Estimators(config['Reward_function'])
    generator.bias_generator(config['Control'], estimator)

def post_process(config):
    """TO-DO"""
    pass

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

    #-- Calculation Type --#
    run_type = config['Control']['run_type']

    if run_type == 'calculate_properties':
        # Given a SMILES file, just calculate the properties
        calculate_properties(config)

    elif run_type == 'generate':
        # Generate new molecules
        generate_mols(config)
                
    elif run_type == 'bias_generator':
        bias_generator(config)
        
    elif run_type == 'post_process':
        post_process(config)

    else:
        raise ValueError((f"Invalid run_type: {run_type}\n"
                          f"Valid options are: calculate_properties, generate, bias_generator, post_process")
                         )
    
if __name__ == '__main__':
    main()