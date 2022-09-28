#!/usr/bin/env python
__description__="""
    Estimate binding affinity using CHEM-BERT 
    pre-trained model
"""
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import glob
import numpy as np
import os, re, time
from tqdm import tqdm

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from .model import Smiles_BERT, BERT_base

#from pathlib import Path
#module_dir = Path(__file__).parent

class Vocab(object):
    """
    Define the SMILES vocabulary
    """
    def __init__(self):
        self.pad_index = 0
        self.mask_index = 1
        self.unk_index = 2
        self.start_index = 3
        self.end_index = 4

        # check 'Na' later
        self.voca_list = ['<pad>', '<mask>', '<unk>', '<start>', '<end>'] + ['C', '[', '@', 'H', ']', '1', 'O', \
                            '(', 'n', '2', 'c', 'F', ')', '=', 'N', '3', 'S', '/', 's', '-', '+', 'o', 'P', \
                             'R', '\\', 'L', '#', 'X', '6', 'B', '7', '4', 'I', '5', 'i', 'p', '8', '9', '%', '0', '.', ':', 'A']

        self.dict = {s: i for i, s in enumerate(self.voca_list)}

    def __len__(self):
        return len(self.voca_list)

class SMILES_Dataset(Dataset):
    '''
    A Dataset of SMILES.
    '''
    def __init__(self, smiles_list, label_list=[], vocab=Vocab(), seq_len=256):
        self.vocab = vocab
        self.seq_len = seq_len

        self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P',
                           'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
        
        self.smiles_dataset = []
        self.adj_dataset = []
        if len(label_list) == 0:
            label_list  = np.zeros_like(smiles_list, dtype=np.float64)
        self.label = label_list.reshape(-1,1)

        for i in smiles_list:
            self.adj_dataset.append(i)
            self.smiles_dataset.append(self.replace_halogen(i))

    def __len__(self):
        return len(self.smiles_dataset)

    def __getitem__(self, idx):
        item   = self.smiles_dataset[idx]
        label  = self.label[idx]
        smiles = self.adj_dataset[idx]

        input_token, input_adj_masking = self.CharToNum(item)

        input_data = [self.vocab.start_index] + input_token + [self.vocab.end_index]
        input_adj_masking = [0] + input_adj_masking + [0]

        smiles_bert_input = input_data[:self.seq_len]
        smiles_bert_adj_mask = input_adj_masking[:self.seq_len]

        padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
        smiles_bert_input.extend(padding)
        smiles_bert_adj_mask.extend(padding)

        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        RDLogger.EnableLog('rdApp.*')
        if mol:
            adj_mat = GetAdjacencyMatrix(mol)
            smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))
        else:
            print("BAD SMILES:", smiles)
            smiles_bert_adjmat = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)

        output = {"smiles_bert_input": smiles_bert_input, "smiles_bert_label": label,  \
                  "smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat}

        return {key:torch.tensor(value) for key, value in output.items()}

    def CharToNum(self, smiles):
        tokens = [i for i in smiles]
        adj_masking = []

        for i, token in enumerate(tokens):
            if token in self.atom_vocab:
                adj_masking.append(1)
            else:
                adj_masking.append(0)

            tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

        return tokens, adj_masking

    def replace_halogen(self,string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        sn = re.compile('Sn')
        na = re.compile('Na')
        string = br.sub('R', string)
        string = cl.sub('L', string)
        string = sn.sub('X', string)
        string = na.sub('A', string)
        return string

    def zero_padding(self, array, shape):
        if array.shape[0] > shape[0]:
            array = array[:shape[0],:shape[1]]
        padded = np.zeros(shape, dtype=np.float32)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded

class chembert_model:
    """
    A model for prediction of binding energies, based on 
    the CHEM-BERT model. Gets as parameter a 'model_state',
    a fine-tuned CHEM-BERT model for binding energies.
    """
    
    def __init__(self,model_state):
        # Use GPU if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # prepare model
        Smiles_vocab = Vocab()
        params = {'batch_size':16, 'dropout':0, 'learning_rate':0.00001,
                'epoch':15, 'optimizer':'Adam', 'model':'Transformer'}
        model = Smiles_BERT(len(Smiles_vocab), max_len=256, nhead=16,
                            feature_dim=1024, feedforward_dim=1024, nlayers=8,
                            adj=True, dropout_rate=params['dropout'])
        output_layer = nn.Linear(1024, 1)
        model = BERT_base(model, output_layer)

        model.load_state_dict(torch.load(model_state, map_location=device))
        model.to(device)

        self.model = model
        self.device = device
        self.params = params

    def predict(self,dataset):
        """
        Receives a 'Dataset' object, and returns an array of
        same length as the dataset, with predictions on every
        point of the dataset.
        """

        # Run the predictions
        device = self.device
        model  = self.model
        params = self.params

        dataloader = DataLoader(dataset, batch_size=params['batch_size'], num_workers=4)

        predictions  = []
        with torch.no_grad():
            pbar_update_n = params['batch_size']
            with tqdm(dataloader, total=len(dataset), desc="Predicting activities ", leave=False) as pbar:
                for i, data in enumerate(dataloader):
                    data = {key:value.to(device) for key, value in data.items()}
                    position_num = torch.arange(256).repeat(data["smiles_bert_input"].size(0),1).to(device)
                    output = model.forward(data["smiles_bert_input"], position_num,
                                        adj_mask=data["smiles_bert_adj_mask"],
                                        adj_mat=data["smiles_bert_adjmat"])
                    output = output[:,0]
                    predictions.extend(output[:,0].tolist())
                    pbar.update(len(data["smiles_bert_input"]))
            return predictions

def SMILES_Dataset_from_file(smiles_file):
    """
    Reads SMILES form a file and returns a Dataset object.
    Expects the file to have SMILES in the first column.
    If present, a second column with labels (true values)
    is also read. Ignores any other columns.
    """
    import pandas as pd

    has_header = False
    with open(smiles_file,'r') as smi_file:
        has_header = csv.Sniffer().has_header(smi_file.read(1024))
        smi_file.seek(0)
    text = pd.DataFrame()
    if has_header:
        text = pd.read_csv(smiles_file, header=0, names=['SMILES','LABELS'])
    else:
        text = pd.read_csv(smiles_file, names=['SMILES','LABELS'])

    smiles_list = np.asarray(text['SMILES'].values)
    label_list  = np.asarray(text['LABELS'].values)

    print(f"Read {len(text)} molecules from file {smiles_file}.")
    return SMILES_Dataset(smiles_list, label_list)

def main():
    import pandas as pd
    import argparse
    parser = argparse.ArgumentParser(description='Evaluates predictions based on a pre-trained model')
    parser.add_argument('trained_model', type=str, help='Path to the pre-trained model')
    parser.add_argument('input_file'   , type=str, help="Path to SMILES file with the molecules to evaluate.")
    parser.add_argument('--output_file', '-o'  , type=str, help="path for the .csv file with results.", default="predictions.dat")
    args = parser.parse_args()

    trained_model = args.trained_model
    input_file    = args.input_file
    output_file   = args.output_file

    model = chembert_model(trained_model)
    dataset = SMILES_Dataset_from_file(input_file)
    labels = dataset.label.flatten()
    
    # Finally, do the predictions:
    predictions = np.array(model.predict(dataset))
    print(f"LABELS: {len(labels)}, PREDICTIONS:{len(predictions)}")
    results = pd.DataFrame({"Prediction":predictions})
    if len(labels) == len(predictions):
        results['Label'] = labels
        error = predictions - labels
        rmse = np.linalg.norm(error) / np.sqrt(len(labels))
        r2 = (np.corrcoef(predictions,labels)[0,1])**2
        print(f"RMSE = {rmse:5.2f}, R^2 = {r2:5.2f}")

    results.to_csv('predictions.csv', index=None)

if __name__ == '__main__':
    main()
