import pandas as pd
from pathlib import Path

def filter_molecules(control_options):
    ...
    return

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=''' Filters molecules based on given criteria.
                    ''')

    parser.add_argument('-i', '--input_file',
                        help="Path to input YML file",
                        required=True
                        )

    args = parser.parse_args()
    smiles_file  = args.smiles_file
    output_stem = args.output_stem
    admet_predictions = Path(args.admet_predictions)
    #----------------------
    filter_smiles(control_opts)