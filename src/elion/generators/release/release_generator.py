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
    gen_data_path = '/blue/lic/huangzihang/repos/elion/src/elion/generators/release/data/chembl_22_clean_1576904_sorted_std_final.smi'
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
    use_cuda=None   # FIX: was hardcoded True, which bypasses stackRNN's auto-detect and crashes
                    # on CPU-only PyTorch builds. None triggers the fallback:
                    #   if self.use_cuda is None: self.use_cuda = torch.cuda.is_available()
                    # -> False on CPU-only builds (works), True when CUDA PyTorch is installed (uses GPU)


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

        # [PRINT] Show vocabulary and architecture summary so you can see the
        # full generative search space and model capacity before any learning.
        print(f"[init] n_characters (vocab size) = {self.gen_data.n_characters}")
        print(f"[init] tokens = {self.gen_data.all_characters}")
        print(f"[init] architecture: layer_type={self.layer_type}, n_layers={self.n_layers}, "
              f"hidden_size={self.hidden_size}, stack_width={self.stack_width}, "
              f"stack_depth={self.stack_depth}, is_bidirectional={self.is_bidirectional}, "
              f"has_stack={self.has_stack}")
        print(f"[init] optimizer={self.optimizer_instance.__name__}, lr={self.lr}, use_cuda={self.use_cuda}")

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

        # [PRINT] Parameter count tells you the total number of learnable weights
        # being updated during RL fine-tuning — the neural analogue of TS's reagent pool size.
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[init] total parameters      = {total_params:,}")
        print(f"[init] trainable parameters  = {trainable_params:,}")
        
        print(f"Generator ready to be used. Elapsed time: {time.time() - begin:.2f} seconds.")

    def generate(self, n_to_generate, verbose=0):
        """
        Generates n_to_generate number of SMILES strings
        """
        
        # [PRINT] Generation session header — mirrors TS's "--- cycle_id ---" header.
        # Shows what the policy is being asked to produce before the loop starts.
        print(f"\n[generate] === generation session: requesting {n_to_generate} unique valid SMILES ===")

        generated, unique_smiles = [], []
        with tqdm(total=n_to_generate,leave=False, ncols=80, unit='mols') as pbar:
            pbar.set_description("Generating molecules")
            total_generated = 0
            total_unique = 0

            while(total_unique < n_to_generate):

                # Generate a new SMILES string
                new_smiles = self.evaluate(self.gen_data, predict_len=120)[1:-1]
                total_generated += 1

                # [PRINT] Raw policy output — the token sequence sampled from the GRU.
                # Analogous to TS printing the winner_idx: you see what the neural policy
                # actually produced before any chemistry validation filters it.
                print(f"[generate] attempt={total_generated} | raw_smiles='{new_smiles}' | len={len(new_smiles)}")

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

                # [PRINT] Validity gate outcome — shows whether the policy's output
                # was chemically valid. Tracking the invalid rate reveals whether
                # RL fine-tuning is destabilizing the generator (mode collapse warning).
                if not mol:
                    print(f"[generate] attempt={total_generated} | INVALID (RDKit parse failed) | smiles='{new_smiles}'")
                else:
                    print(f"[generate] attempt={total_generated} | valid | canonical='{new_canonic}'")

                if mol:
                    if self.filter_smiles(new_canonic):
                        generated.append(new_canonic)
                        unique_smiles = set(generated)

                        # [PRINT] Filter pass + uniqueness tracking — mirrors TS's score_counts
                        # (how many times each arm has been pulled). Here, n_total_accepted and
                        # n_unique show exploitation breadth: a stagnant n_unique vs growing
                        # n_total_accepted signals the policy is looping over the same molecules.
                        new_size = len(unique_smiles)
                        duplicate = (new_size == total_unique)
                        print(f"[generate] attempt={total_generated} | filter=PASS | "
                              f"n_total_accepted={len(generated)} | n_unique={new_size} | "
                              f"duplicate={'YES' if duplicate else 'NO'}")

                        if (new_size > total_unique):
                            total_unique = new_size
                            pbar.update(1)
                    else:
                        # [PRINT] Filter rejection — shows which valid-but-unwanted molecules
                        # the policy is generating (e.g. too short). Frequent rejections here
                        # suggest the policy is generating fragments, not drug-like molecules.
                        print(f"[generate] attempt={total_generated} | filter=REJECT | canonical='{new_canonic}'")

                if '' in unique_smiles:
                    unique_smiles.remove('')

                # [PRINT] Running efficiency summary every 50 attempts — analogous to TS's
                # mu/std range summary per cycle. Shows validity rate and uniqueness rate
                # so you can track whether the policy is improving, stagnating, or collapsing.
                if total_generated % 50 == 0:
                    validity_rate = len(generated) / total_generated if total_generated > 0 else 0.0
                    uniqueness_rate = total_unique / total_generated if total_generated > 0 else 0.0
                    print(f"[generate] --- progress @ attempt {total_generated} ---")
                    print(f"[generate]   valid_accepted / total_attempts = {len(generated)} / {total_generated} "
                          f"({validity_rate:.2%})")
                    print(f"[generate]   unique / total_attempts         = {total_unique} / {total_generated} "
                          f"({uniqueness_rate:.2%})")
                    print(f"[generate]   still needed                    = {n_to_generate - total_unique}")


        n_valid = len(generated)
        n_unique = len(unique_smiles)

        # [PRINT] Session-end summary — mirrors TS's final posterior state printout.
        # validity_rate and uniqueness_rate together diagnose policy health:
        #   low validity   → RL has destabilized the chemical language model
        #   low uniqueness → policy has collapsed to a narrow region of chemical space
        print(f"\n[generate] === session complete ===")
        print(f"[generate] total_attempted   = {total_generated}")
        print(f"[generate] valid_accepted    = {n_valid}  ({(n_valid / total_generated):0.2%})")
        print(f"[generate] unique_valid      = {n_unique} ({(n_unique / total_generated):0.2%})")
        print(f"[generate] redundancy        = {n_valid - n_unique} duplicates discarded")

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

        # [PRINT] Filter decision trace — shows exactly which criterion killed the molecule.
        # When the policy starts generating many short fragments, this pinpoints the threshold.
        if not approved:
            reason = "mol_invalid" if not mol else f"len={len(smiles_string)}<6"
            print(f"[filter_smiles] REJECT | smiles='{smiles_string}' | reason={reason}")

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

        # [PRINT] Pre-training parameter snapshot — records the weight distribution
        # before any gradient updates. Comparing this to post-training norms reveals
        # how much the RL loop has shifted the policy away from the pretrained prior.
        # This is the ReLeaSE analogue of TS's warmup prior_mean/prior_std snapshot.
        print(f"\n[train] === training session start ===")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"[train] pre-train | layer={name} | "
                      f"mean={param.data.mean().item():.6f} | "
                      f"std={param.data.std().item():.6f} | "
                      f"norm={param.data.norm().item():.4f}")

        # fit, evaluate and save_model are methods from the parent class, stackRNN.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            losses = self.fit(self.gen_data, 1500000)
            losses = self.fit(self.gen_data, 10)
        self.evaluate(self.gen_data)

        # [PRINT] Post-training parameter snapshot — the delta between pre and post norms
        # shows how much the RL loop has shifted the neural policy weights, analogous to
        # watching delta_mu and delta_std in TS after add_score().
        print(f"\n[train] === training session complete ===")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"[train] post-train | layer={name} | "
                      f"mean={param.data.mean().item():.6f} | "
                      f"std={param.data.std().item():.6f} | "
                      f"norm={param.data.norm().item():.4f}")

        self.save_model(model_path)
        print(f"[train] model saved to {model_path}")
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