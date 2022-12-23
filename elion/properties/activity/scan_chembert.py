"""
Scans a database using a CHEM-BERT model
"""
from CHEMBERT.chembert import chembert_model, SMILES_Dataset_from_file
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
        
if __name__ == '__main__':
    import time
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import argparse
    
    #-- Command line arguments
    parser = argparse.ArgumentParser(description='''Tests a CHEMBERT model''')

    parser.add_argument('smiles_file',
                        help='Path to the input SMILES file. Must have 2 columns: SMILES, data')

    parser.add_argument('-m', '--model',
                        help='trained model',
                        default=f'{module_dir}/CHEMBERT/model/pretrained_model.pt')

    parser.add_argument('-s', '--smiles_col',
                        help='Name of column containing SMILES',
                        default='SMILES')

    parser.add_argument('-l', '--labels_col',
                        help='Name of column containing labels',
                        default='LABELS')

    args = parser.parse_args()
    
    smiles_file   = Path(args.smiles_file)
    trained_model = Path(args.model)
    smiles_col    = args.smiles_col
    labels_col    = args.labels_col

    #---
    output_name = f'{smiles_file.stem}_{trained_model.stem}'
    properties = {'predictor':'CHEM-BERT',
                  'predictor_type':'regressor',
                  'state':trained_model}
    
    start_time = time.time()
    predictor = activity_predictor(properties)

    dataset = SMILES_Dataset_from_file(smiles_file, smiles_col=smiles_col, labels_col=labels_col)
    labels = dataset.label.flatten()
    smiles = dataset.adj_dataset

    assert len(labels) == len(smiles), "Length of SMILES and LABELS column mismatch!!!" 

    # Do the predictions:
    predictions = predictor.predict(dataset)

    results = pd.DataFrame({"Prediction":predictions})
    if len(labels) == len(predictions):
        results[labels_col] = labels
        results[smiles_col] = smiles
        results = results[[labels_col,smiles_col,"Prediction"]]
        results = results.sort_values(by=["Prediction",labels_col])
    else:
        print("Danger Will Robinson! Molecules lost is space!!")

    results.to_csv(f'{output_name}.csv', float_format='%.2f', index=None)

    elapsed = time.time() - start_time
    print(F"ELAPSED TIME: {elapsed:.5f} seconds ({elapsed/len(labels):.5f} sec/molecule.)")

    # Plot the results
    plt.figure(3)
    plt.hist(predictions)
    plt.savefig(f'{output_name}.png')

    print("Have a nice day!")
