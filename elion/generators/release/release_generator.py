#Basic stuff
import pandas as pd
import numpy as np
import warnings
import pickle
import time
import sys

# Chemistry
from rdkit import Chem, DataStructs, RDLogger

# Initialize tqdm progress bar
from tqdm import tnrange, tqdm_notebook
from tqdm.auto import tqdm

# CUDA / PyTorch
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F

# ReLeaSE specific classes
from .stackRNN import StackAugmentedRNN
from .data     import GeneratorData
from .utils    import canonical_smiles, time_since

def print_torch_info():
    print("torch.cuda.is_available()          =", torch.cuda.is_available())
    print("torch.cuda.device_count()          =", torch.cuda.device_count())
    print("torch.cuda.current_device()        =", torch.cuda.current_device())
    print("torch.cuda.device('cuda')          =", torch.cuda.device('cuda'))
    print("torch.cuda.get_device_name()       =", torch.cuda.get_device_name(0))
    print("torch.cuda.get_device_capability() =", torch.cuda.get_device_capability(0))
    return

class release_smiles_generator(StackAugmentedRNN):
    """
    Defines a SMILES generator for the ReLeaSE package:
    https://github.com/isayev/ReLeaSE

    This version uses a stack augmented generative GRU as a generator. The model was trained 
    to predict the next symbol from SMILES alphabet using the already generated prefix.
    Model was trained to minimize the cross-entropy loss between predicted symbol 
    and ground truth symbol.

    Once instantiated, the generator can either be trained with raw data, or read a pre-trained
    instance previously stored.

    When calling the "generate" function wiht a number of desired SMILES strings, the generator
    with keep generating new strings until reaching the required number of valid SMILES strings,
    according the the conditions / filtering requested.
    """

    # For debug only. For *very* verbose, set > 5.
    verbosity = 0

    # Those are defaults, and can be overridden when creating a generator object
    gen_data_path = '/home/seabra/work/li/hitopt/hitopt/generator/release/data/chembl_22_clean_1576904_sorted_std_final.smi'
    gen_tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                 '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                 '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
    
    # The details of the RNN. Those are hardcoded here, for now, 
    # but later we can make them variable.

    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    lr = 0.001
    n_layers=1
    is_bidirectional=False
    has_stack=True
    use_cuda=True,


    optimizer_instance = torch.optim.Adadelta


    def __init__(self, data_path = gen_data_path, tokens = gen_tokens):
        """ 
        Initializes the generator
        """
        
        # Initialize stack-augmented generative RNN:
        print("Initializing the generator. Please wait a minute...")
        print("--> data_path = ",data_path)
        print("--> tokens    = ",tokens)

        begin = time.time()

        self.data_path = data_path
        self.tokens = tokens
        self.gen_data = GeneratorData(training_data_path=data_path, delimiter='\t', 
                                cols_to_read=[0], keep_header=True, tokens=tokens)

        super().__init__(input_size=self.gen_data.n_characters, 
                         hidden_size=self.hidden_size,
                         output_size=self.gen_data.n_characters, 
                         layer_type=self.layer_type,
                         n_layers=self.n_layers, 
                         is_bidirectional=self.is_bidirectional, 
                         has_stack=self.has_stack,
                         stack_width=self.stack_width, 
                         stack_depth=self.stack_depth, 
                         use_cuda=self.use_cuda, 
                         optimizer_instance=self.optimizer_instance, 
                         lr=self.lr)
        
        print(f"Generator ready to be used. Elapsed time: {time.time() - begin:.2f} seconds.")

    def generate(self, n_to_generate, verbose=0):
        """
        Generates n_to_generate number of SMILES strings
        """
        
        generated, unique_smiles = [], []
        with tqdm(total=n_to_generate) as pbar:
            pbar.set_description("Generating molecules")
            total_generated = 0
            total_unique = 0

            while(total_unique < n_to_generate):

                # Generate a new SMILES string
                new_smiles = self.evaluate(self.gen_data, predict_len=120)[1:-1]
                total_generated += 1

                # Check that this SMILES is valid.
                # Sometimes a problem arises only after trying to
                # generate a new molecule from the sanitized smiles.
                # So, we need to:
                # 1. Create a Mol object from the raw SMILES and sanitize
                # 2. Create SMILES from this Mol object
                # 3. Try to create a new Mol from teh sanitized SMILES.
                # If the sanitization encounter an error, it will fail step #3. 
                mol = None
                RDLogger.DisableLog('rdApp.*')
                mol = Chem.MolFromSmiles(new_smiles, sanitize=True)
                if mol:
                    new_canonic = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                    mol = Chem.MolFromSmiles(new_canonic, sanitize=True)
                RDLogger.EnableLog('rdApp.*')

                if mol:
                    if self.filter_smiles(new_canonic):
                        generated.append(new_canonic)
                        unique_smiles = set(generated)
                        #unique_smiles = list(np.unique(generated))

                        # Only update the bar if we got a new molecule.
                        new_size = len(unique_smiles)
                        if (new_size > total_unique):
                            total_unique = new_size
                            pbar.update(1)

                if '' in unique_smiles:
                    unique_smiles.remove('')


        n_valid = len(generated)
        n_unique = len(unique_smiles)
        print(f"Generated    : {total_generated}")
        print(f"Valid SMILES : {n_valid} ({(n_valid / total_generated):0.2%} of the total)")
        print(f"Unique SMILES: {n_unique} ({(n_unique / total_generated):0.2%} of the total)")

        return list(unique_smiles)

    def filter_smiles(self, smiles_string):
        """Applies a filter to the SMILES string.
           At the moment, only checks if len(smiles) >= 6.
           Other checks may be added later.

        Args:
            smiles_string (str): One SMILES string.

        Returns:
            Boolean: True if approved by filters, False otherwise
        """

        approved = True

        # Molecule size: > 6 carbon atoms
        mol = Chem.MolFromSmiles(smiles_string)
        if (not mol) or (len(smiles_string) < 6):
            approved = False

        return approved

    def train(self, model_path):
        """
        Train the SMILES generator, using the `gen_data` defined above. 
        
        Warning: takes a while!
        The training is set to 1,500,000 iterations (epochs) which, in my computer 
        (2018 Acer Predator Helios 300 with 1 NVIDIA GEFORCE GTX 1060, 6GB memory), 
        takes ~5h 54min ( approx. 6h), to run only 3% (47,726) of the total of 1,500,000 iterations.
        For running all the iterations on my computer it would take ~12,000 minutes, or 200h.

        Alternatively, one can use the pre-trained models from ReLeaSE github.

        Args:
            model_path (Path): Path to write the trained generator.
 
        """

        # fit, evaluate and save_model are methods from the parent class, stackRNN.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            losses = self.fit(self.gen_data, 1500000)
            losses = self.fit(self.gen_data, 10)
        self.evaluate(self.gen_data)
        self.save_model(model_path)
        return

if __name__ == "__main__":
    smi_gen = release_smiles_generator()
    model_path = './release/checkpoints/generator/checkpoint_biggest_rnn'
    smi_gen.load_model(model_path)

    n_to_generate = int(sys.argv[1])
    new_mols = smi_gen.generate(n_to_generate)
    num = 0
    with open("generated_smiles.smi",'w') as output:
        for smiles in new_mols:
            num = num + 1
            output.write(f"{smiles}, Gen-{num:04d}\n")