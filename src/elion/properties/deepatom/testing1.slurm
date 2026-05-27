#!/bin/bash
#SBATCH --job-name=DeepAtom
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=email
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=1-00:00:00


# This script starts with the binding poses in PDBQT format
# (AutoDock/Vina outputs) 
#
# Change DATA_DIR to your desired folder. 
# Inside this folder put:
#  - One folder called 'pdbqt', with the PDBQT files of the
#    docked poses, and 
#  - One folder called 'protein' with the protein PDB file.

DATA_DIR="/blue/lic/huangzihang/repos/TEAD3/pocketflow"

#
# -- No need to change anything beow this line.
#
echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

echo "#####################################################################"
echo "Done"
