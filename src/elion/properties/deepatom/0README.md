Installation of DeepAtom in HiPerGator
======================================

1. Environment:
---------------
Install the binding_affinities_27 environment with the 
enviriment spec file, e.g.:

$ mamba create --name binding_affinity_27 --file binding_affinity_27_spec-file.txt

2. Paths:
---------
Make sure the paths in the files:

DeepAtom location:
- bin/predict_binding_affinity_v4_2_data_split.sh:SCRIPTS_DIR="/home/seabra/work/source/deepatom/deepatom/model_split_data"
- bin/run_deepatom.sh:deepatom="/home/seabra/work/source/deepatom/deepatom/bin/predict_binding_affinity_v4_2_data_split.sh"

Chimera location
- model_split_data/00_preprocess/02b_augment_in_Chimera_VS_opt.sh:chimera="/apps/chimera/1.12/sbin/chimera"

point to the correct location.

Running DeepAtom on HiPerGator
==============================

In the run script, before running DA, 
1. Make sure to load chimera: `mocule load chimera`
2. make sure to activate the environemnt: `conda activate binding_affinity_27`
