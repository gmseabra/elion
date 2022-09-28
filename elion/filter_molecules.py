import pandas as pd
from pathlib import Path

def filter_smiles(smiles_file, admet_predictions_file, output_stem):
    """ Filter the SMILES based on the given criteria.

    Args:
        smiles_file (file): Path to SMILES file created by Elion
        admet_predictions_file (file): path to file containing ADMET Predictor results
        output_stem (str): Stem of output file. a 'smi' will be added to the end.
    """
    # reads the SMILES file from Elion
    smiles_data = pd.read_csv(smiles_file,
                              dtype={'SMILES':str,
                                     'Name':str,
                                     'prob_active':float,
                                     'scaffold_match':float,
                                     'synthetic_accessibility':float,
                                     'drug_likeness':float})
    admet_data  = pd.read_csv(admet_predictions_file, sep='\t',
                              usecols=['Name','MUT_Risk','TOX_Risk','ADMET_Risk','%Fb_hum-100.0'],
                              dtype={'Name':str,
                                     'SMILES':str,
                                     'MUT_Risk':float,
                                     'TOX_Risk':float,
                                     'ADMET_Risk':float,
                                     '%Fb_hum-100.0':float})
    data = smiles_data.merge(admet_data, on=['Name'])

    data = data[(data['scaffold_match'] > 0.95 ) &
                (data['prob_active']    < -7.0 ) &
                (data['MUT_Risk']       < 1.0  ) &
                (data['TOX_Risk']       < 2.0  ) &
                (data['ADMET_Risk']     < 7.0  ) &
                (data['%Fb_hum-100.0']  > 0.80 )
               ]

    data.to_csv(f"{output_stem}.smi", index=None)
    return
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=''' Filters molecules based on given criteria.
                    ''')

    parser.add_argument('-i', '--smiles_file',
                        help="Path to input (SMILES) file",
                        required=True
                        )
    parser.add_argument('-a','--admet_predictions',
                        help="Path to the file with ADMET Predictor results (if any)",
                        default=None)
    parser.add_argument('-o','--output_stem',
                        help="Basename of output file",
                        default="filtered_smiles")

    args = parser.parse_args()
    smiles_file  = args.smiles_file
    output_stem = args.output_stem
    admet_predictions = Path(args.admet_predictions)
    #----------------------
    filter_smiles(smiles_file, admet_predictions, output_stem)