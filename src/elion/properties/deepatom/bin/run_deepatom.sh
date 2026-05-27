#!/bin/bash

#  _____                         _
# |  __ \                   /\  | |
# | |  | | ___  ___ _ __   /  \ | |_ ___  _ __ ___
# | |  | |/ _ \/ _ \ '_ \ / /\ \| __/ _ \| '_ ` _ \
# | |__| |  __/  __/ |_) / ____ \ || (_) | | | | | |
# |_____/ \___|\___| .__/_/    \_\__\___/|_| |_| |_|
#                  | |
#                  |_|
#

# To run DeepAtom:
# 
# 1. Make a folder for you data containing 2 subfolders,
#    named `protein` and `ligands_in_bound_pose`.
# 2. (Optional) Copy this script to the directory you just created
# 2. Put the PDB file for the protein in the `protein` folder
# 3. Put the ligand PDB files in the `ligands_in_bound_pose` folder:
#
#      My_Data/
#      ├── ligands_in_bound_pose
#      │   ├── ligand_1.pdb
#      │   ├── ligand_2.pdb
#      │   ├── ligand_3.pdb
#      │   ├── ligand_4.pdb
#      │   └── ligand_5.pdb
#      └── protein
#      │   └── ABCD.pdb
#      └── run_deepatom.sh
#
# 4. Modify the only the "data-dir" line here with
#    the full path pointing to your data.
# 
# 5. Run the script
#

if [ $# -eq 0 ]
  then
    echo "Usage:"
    echo "run_deepatom.sh my_data_dir"
    echo "argument <my_data_dir> is required."
    exit -1
fi
echo " Runnning DeepAtom for files in folder >> $1"


data_dir=$1

###############################################################################
#                                                                             #
#                   DO NOT MODIFY ANYTHING BELOW THIS LINE                    #
#                 (Unless you are sure of what you are doing)                 #
#                                                                             #
###############################################################################

# Program
# deepatom="/usr/local/deepatom/bin/predict_binding_affinity_v4_2_data_split.sh"
deepatom="/blue/lic/share/local/deepatom/bin/predict_binding_affinity_v4_2_data_split.sh"

# Runs the DeepAtom Model
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate binding_affinity_27

time $deepatom -t 'vs' -d $data_dir
# rm -rf ${HOME}/DEEP_MODEL_temp
# rm -rf /blue/lic/seabra/DEEP_MODEL_temp

echo "#####################################################################"
echo "Done"

