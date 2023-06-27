
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
                        default=f'{module_dir}/properties/activity/CHEMBERT/model/pretrained_model.pt')

    args = parser.parse_args()
    
    smiles_file   = Path(args.smiles_file)
    trained_model = Path(args.model)

    #---
    output_name = f'{smiles_file.stem}_{trained_model.stem}'
    properties = {'predictor':'CHEM-BERT',
                  'predictor_type':'regressor',
                  'state':trained_model}
    
    start_time = time.time()
    predictor = activity_predictor(properties)

    dataset = SMILES_Dataset_from_file(smiles_file)
    labels = dataset.label.flatten()

    # Finally, do the predictions:
    predictions = predictor.predict(dataset)

    results = pd.DataFrame({"Prediction":predictions})
    if len(labels) == len(predictions):
        results['Label'] = labels
        error = predictions - labels
        rmse = np.linalg.norm(error) / np.sqrt(len(labels))
        r2 = (np.corrcoef(predictions,labels)[0,1])**2
        print(f"{rmse = :5.2f}, {r2 = :5.2f}")

    results.to_csv(f'{output_name}.csv', float_format='%.2f', index=None)

    elapsed = time.time() - start_time
    print(F"ELAPSED TIME: {elapsed:.5f} seconds ({elapsed/len(labels):.5f} sec/molecule.)")

    # Plot the results
    plt.figure(3)
    plt.scatter(labels, predictions, s=10, alpha=0.5)

    # Linear regression to get a fit line
    b,a = np.polyfit(labels, predictions, deg=1)
    x_fit = np.linspace(min(labels), max(labels))
    plt.plot(x_fit, (a+b*x_fit), color='red', lw=2.5)
 
    # Adds an x=y line.
    xpoints = ypoints = plt.xlim()
    plt.margins(0,0)
    plt.grid(color='lightgrey', linestyle='-', lw=1.0)
    plt.plot(xpoints,ypoints,
             color='grey', label=f'(X=Y)', linestyle='--', lw=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.suptitle(f"Predictions on: {smiles_file.name}", size='x-large')
    plt.title(f'Model: {trained_model.name}')
    plt.legend(loc='lower right', title=f'$R^2$ = {r2:0.2f}')
    plt.savefig(f'{output_name}.png')

    print("Have a nice day!")