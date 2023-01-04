"""
Scans a database using a CHEM-BERT model
"""
import time
import numpy as np
import pandas as pd
from CHEMBERT.chembert import chembert_model, SMILES_Dataset
from pathlib import Path
module_dir = Path(__file__).parent

class activity_predictor:
    
    def __init__(self, properties):
        
        self.base = properties['predictor']
        self.type = properties['predictor_type']

        if self.base == 'CHEM-BERT':
            self.model = chembert_model(properties['state'])
            
    def predict(self, dataset):
        return self.model.predict(dataset)

def read_smiles_file(smiles_file, smiles_col='SMILES'):
    """
    Reads SMILES form a file and returns a DataFrame object.
    This checks if the file has a header before reading, and
    makes sure there will be a column named "SMILES" for processing.
    
    If the header is not present, assume the SMILES column to be 
    the first column. otherwise, we locate the SMILES column and move
    it to the first.
    """
    header = None
    with open(smiles_file,'r') as smi_file:
        line1 = smi_file.readline().upper()
        if "SMILES" in line1:
            header = 0
            print("Header found: ", line1)
        else:
            print("WARNING: Header not found. Assuming SMILES in 1st column.")
        smi_file.seek(0)

    data_df = pd.read_csv(smiles_file, header=header)
    
    if header == None: 
        data_df.rename(columns={0:"SMILES"}, inplace=True)
    else:
        data_df.rename(columns={smiles_col:"SMILES"}, inplace=True)

    # Bring SMILES column to first
    smiles = data_df.pop('SMILES')
    data_df.insert(0,'SMILES',smiles)
    print(f"Read {len(data_df)} molecules from file {smiles_file}:")
    print(data_df.head(5))
    return data_df
       
if __name__ == '__main__':
    import argparse
    
    #-- Command line arguments
    parser = argparse.ArgumentParser(description='''Tests a CHEMBERT model''')

    parser.add_argument('smiles_file',
                        help='Path to the input SMILES file. Must have at least a SMILES column')

    parser.add_argument('-m', '--model',
                        help='trained model',
                        default=f'{module_dir}/CHEMBERT/model/pretrained_model.pt')

    parser.add_argument('-s', '--smiles_col',
                        help='Name of column containing SMILES',
                        default='SMILES')

    parser.add_argument('-p', '--plot_results',
                        help='If present, make histplots of the results.',
                        action='store_true')

    args = parser.parse_args()
    
    smiles_file   = Path(args.smiles_file)
    trained_model = Path(args.model)
    smiles_col    = args.smiles_col
    plot_results  = args.plot_results

    #---
    output_name = f'{smiles_file.stem}_{trained_model.stem}'
    properties = {'predictor':'CHEM-BERT',
                  'predictor_type':'regressor',
                  'state':trained_model}
    
    start_time = time.time()
    predictor = activity_predictor(properties)

    data   = read_smiles_file(smiles_file, smiles_col=smiles_col)

    # Removes duplicate SMILES strings.
    start_time_cleandb = time.time()
    initial_size = len(data)
    data.drop_duplicates(subset=['SMILES'], inplace=True, ignore_index=True)
    final_size = len(data)
    if final_size != initial_size:
        print(f"WARNING: Dropped {initial_size - final_size} duplicate SMILES.")
        print(f"         The final results will have only {final_size} points.")
    elapsed_cleandb = time.time() - start_time_cleandb
    print(f"ELAPSED TIME (Filtering DB): {elapsed_cleandb:.5f} seconds)")

    # Calculates the predictions
    start_time_calc = time.time()
    dataset = SMILES_Dataset(np.asarray(data['SMILES'].values))
    results = pd.DataFrame({"Prediction":predictor.predict(dataset)})

    if len(results) == len(data):
        results = data.merge(results, left_index=True, right_index=True)
        results = results.sort_values(by=["Prediction"])
    else:
        print("Danger Will Robinson! Molecules lost is space!!")
        print(f"results: {len(results)}, data:{len(data)}.")

    results.to_csv(f'{output_name}.csv', float_format='%.2f', index=None)

    elapsed_calc = time.time() - start_time_calc
    print(F"ELAPSED TIME (Predictions) : {elapsed_calc:.5f} seconds ({elapsed_calc/len(data):.5f} sec/molecule.)")

    # Plot the results
    if plot_results:
        start_time_plot = time.time()

        print("Preparing histogram plot.... ", end="", flush=True)
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots()
        ax=sns.histplot(data=results, x="Prediction", kde=True)
        fig.suptitle(f"Predicted Vina Scores for file \"{smiles_file}\"")
        fig.tight_layout()
        fig.savefig(f'{output_name}.png')
        elapsed_plot = time.time() - start_time_plot
        print(f"Done.\nELAPSED TIME (Plotting)    : {elapsed_plot:.5f} seconds.")

    elapsed_total = time.time() - start_time
    print(f"ELAPSED TIME (TOTAL)       : {elapsed_calc:.5f} seconds")
    print("Have a nice day!")
