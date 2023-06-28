#!/usr/bin/env python

"""
Generates a batch of molecules using the MoLeR generator
"""

import numpy as np
from pathlib import Path
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'  # or any {'0', '1', '2'}

# Local
from molecule_generation import VaeWrapper

# -----------------------------------------------------------------------
def main():

    #-- Command line arguments
    import argparse
    import sys
    # print("COMMAND:", ' '.join(sys.argv))
    parser = argparse.ArgumentParser(
        description=''' Generate molecles with MoLeR generator
                    ''')

    parser.add_argument('model_dir',
                        help='Path to the model dir. REQUIRED')

    parser.add_argument('n_batch', type=int,
                        help='Number of molecules to generate. REQUIRED')

    parser.add_argument('-s', '--scaffold',
                        help='Scaffold SMILES',
                        default=None)

    parser.add_argument('-c', '--center',
                        help='SMILES at center of latent space.',
                        default=None)

    parser.add_argument('-d', '--stdev', type=float,
                        help='Standard deviation around center',
                        default=1.0)

    args         = parser.parse_args()
    n_batch      = args.n_batch
    model_dir    = args.model_dir
    scaffold_smi = args.scaffold
    center_smi   = args.center
    stdev        = args.stdev
    # -----------------------------------------------------

    seed = np.random.randint(100_000_000)
    with VaeWrapper(model_dir, seed=seed) as generator:

        if center_smi is None:
            embeddings = generator.sample_latents(n_batch)
        else:
            [latent_center] = generator.encode([center_smi])
            embeddings = latent_center + stdev * np.random.randn(n_batch, latent_center.shape[0]).astype(np.float32)
        
        if scaffold_smi is None:
            decoded_smiles = generator.decode(embeddings)
        else:
            decoded_smiles = generator.decode(embeddings,[scaffold_smi]*n_batch)

    for smi in decoded_smiles: print(smi)
if __name__ == "__main__":
    main()
