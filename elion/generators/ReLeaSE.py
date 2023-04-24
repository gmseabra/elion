#!/usr/bin/env python
__description__="""
    Simply generate new molecule SMILES using
    some generator.

    This code uses the ReLeaSE program from:
    https://github.com/isayev/ReLeaSE
"""

import time
import pandas as pd

from generators.Generator import Generator
from generators.release.release_generator import release_smiles_generator, print_torch_info

import importlib
from pathlib import Path

class ReLeaSE(Generator):
    """The ReLeaSE Generator
    https://github.com/isayev/ReLeaSE
    """
    #-------------------------------------------

    def __init__(self,generator_properties):

        initial_state = "generators/release/checkpoints/generator/checkpoint_biggest_rnn"
        if 'initial_state' in generator_properties.keys():
            initial_state = generator_properties['initial_state']
        self.initial_state = Path(initial_state)

        self.generator = release_smiles_generator()
        print(f"\nLoading generator from file {initial_state} ... ", end='')
        self.generator.load_model(self.initial_state)
        print("Done.")

    def bias_generator(self, prop_values):
        pass