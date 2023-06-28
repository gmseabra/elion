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

    parser.add_argument('-s', '--scaffolds_file',
                        help='Scaffolds SMILES',
                        default=None)

    parser.add_argument('-c', '--centers_file',
                        help='SMILES file for centers of latent space.',
                        default=None)

    parser.add_argument('-d', '--stdev', type=float,
                        help='Standard deviation around center',
                        default=1.0)

    args           = parser.parse_args()
    n_batch        = args.n_batch
    model_dir      = args.model_dir
    scaffolds_file = args.scaffolds_file
    centers_file   = args.centers_file
    stdev          = args.stdev
    # -----------------------------------------------------

    seed = np.random.randint(100_000_000)

    with VaeWrapper(model_dir, seed=seed) as generator:
        if centers_file is None:
            embeddings = generator.sample_latents(n_batch)
        else:
            # Here, centers_file is a filename, containing SMILES for different centers
            # Read the centers into a list
            centers_smi = []
            with open(centers_file.strip(), 'r') as cf:
                for line in cf:
                    centers_smi.append(line.replace('\n',''))

            # Now, generate an n_batch long vector of SMILES from
            # the centers' SMILES.

            # If the length of the centers_smi is the samas n_batch,
            # we assume the user knows what they want. Otherwise, we 
            # build the batch_centers vector by sampling from the 
            # centers_smi with repetition, so as to not priviledge any
            # partcular molecule.

            batch_centers = []
            if len(centers_smi) == n_batch:
                batch_centers = centers_smi
            else:
                batch_centers = np.random.choice(centers_smi,size=n_batch).tolist()
            
            latent_center = generator.encode(batch_centers)
            latent_center = np.array(latent_center)

            embeddings = latent_center + stdev * np.random.randn(n_batch, latent_center.shape[1]).astype(np.float32)
        
        if scaffolds_file is None:
            decoded_smiles = generator.decode(embeddings)
        else:
            scaffolds_smis = []
            with open(scaffolds_file.strip(), 'r') as sf:
                for line in sf:
                    scaffolds_smis.append(line.replace('\n',''))

            scaffolds = []
            if len(scaffolds_smis) == n_batch:
                scaffolds = scaffolds_smis
            else:
                scaffolds = np.random.choice(scaffolds_smis,size=n_batch).tolist()
            
            decoded_smiles = generator.decode(embeddings,scaffolds)

    for smi in decoded_smiles: print(smi)
if __name__ == "__main__":
    main()
