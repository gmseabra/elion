#!/bin/bash

# Finetunes the MoLeR generator
#
# Use this script to activate the appropriate conda env
# as needed, then run MoLeR inside this env.
#
# Usage:
# finetune_moler.sh <model_path> <n_to_generate>
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate moler-env

MOLER_DIR=`dirname $0`
FINETUNED_DIR=$1
PRETRAINED_DIR=$2

# 1. Preprocess SMILES
molecule_generation preprocess INPUT_DIR OUTPUT_DIR TRACE_DIR

# 2. Fine-tune the generator
molecule_generation train --save-dir FINETUNED_DIR \
						  --load-saved-model PRETRAINED_DIR \
						  --load-weights-only \

