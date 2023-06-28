import pandas as pd
import subprocess
from pathlib import Path

def convert_smiles(smiles_file, output_stem, name_prefix="Lig", ):
    """Formats the SMILES file to a format suitable
       to use with ADMET Predictor.

       ADMET Predictor expects a SMILES file with:
       (i)  NO title line, and
       (ii) Only SMILES and NAME, separated by TAB.

       This scripts reads a comma-separated SMILES file 
       and writes only the SMILES and NAME to a file
       with the format expected by ADMET Predictor.

    Args:
        smiles_file (file): A CSV SMILES file from Elion, that MUST have a
                            named "SMILES". It *may* also have a "Name" column.
                            All other columns will be ignored.
    """
    with open(smiles_file, 'r') as infile:
        this_cpd = 0
        new_file = []
        header_line = infile.readline().split(',')
        header_line = [x.strip().upper() for x in header_line]

        print("Found columns:", header_line)

        if 'SMILES' in header_line:
            smiles_col = header_line.index('SMILES')
        else:
            smiles_col = 0
            infile.seek(0)
        print('SMILES columns: ', smiles_col)
        
        name_col = header_line.index('NAME') if 'NAME' in header_line else -1
        print("Name column: ", name_col)
        for line in infile:
            tokens = line.split(',')
            this_cpd += 1

            if name_col == -1:
                cpd_name = f"{name_prefix}-{this_cpd:04d}"
            else:
                cpd_name = tokens[name_col]

            new_line = f"{tokens[0]}\t{cpd_name}\n"
            new_file.append(new_line)

    with open(f"{output_stem}.smi",'w') as nf:
        for line in new_file:
            nf.write(line)
    return


def predict_admet(smiles_file, admet_predictor, output_stem, keep_smi):
    """Predict ADMET properties for molecules in file <filename>

    Args:
        smiles_file (file): A CSV SMILES file from Elion, that MUST have a
                            named "SMILES". It *may* also have a "Name" column.
                            All other columns will be ignored.
    """

    # Fist, convert the SMILES file to the format required
    # by ADMET predictor
    convert_smiles(smiles_file, output_stem)

    # Now, runs ADMET-Predictor with this file
    _ = subprocess.run([admet_predictor,
                        "-t", "SMI",
                        f"{output_stem}.smi",
                        "-m","TOX,GLB,SimFaFb",
                        "-N","16",
                        "-out",output_stem])
    if not keep_smi:
        Path(f"{output_stem}.smi").unlink()

    return

# #### This must be moved to a different routine.
#     # Reads the results into a Pandas Dataframe for filtering
#     results = pd.read_csv("AP.dat", sep='\t',
#                          usecols=['MUT_Risk','TOX_Risk','ADMET_Risk','%Fb_hum-100.0'])

#     # Filter
#     approved = results[ (results['MUT_Risk']      <  1.0) &
#                         (results['TOX_Risk']      <  2.0) &
#                         (results['ADMET_Risk']    <  7.0) &
#                         (results['%Fb_hum-100.0'] > 80.0)
#                       ]

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=''' Calculates ADMET properties using 
                        an existing installation of SimulationsPlus
                        ADMET Predictor.
                    ''')

    parser.add_argument('-i', '--smiles_file',
                        help="Path to input (SMILES) file",
                        required=True
                        )
    parser.add_argument('-x','--admet_predictor',
                        help="Path to the ADMET Predictor executable",
                        default='/opt/SimulationsPlus/bin/RunAP.sh'
                        )
    parser.add_argument('-o','--output_stem',
                        help="Basename of output file",
                        default="ADMET_properties")
    parser.add_argument('-k', '--keep_smi',
                        help="Keep intermediary SMILES file.",
                        default=False)

    args = parser.parse_args()
    smiles_file  = args.smiles_file
    output_stem = args.output_stem
    keep_smi = args.keep_smi
    admet_predictor = Path(args.admet_predictor)
    #----------------------
    print(f"Predicting properties using: {admet_predictor}")
    predict_admet(smiles_file, admet_predictor, output_stem, keep_smi)