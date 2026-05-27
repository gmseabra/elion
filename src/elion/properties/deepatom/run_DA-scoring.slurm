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


# Programs

# Creates the data structure
mkdir -p "${DATA_DIR}/ligands_in_bound_pose"

# If needed, expand the pdbqt files.
module load openbabel 
OBABEL="obabel"
PDBQT_DIR="$DATA_DIR/pdbqt"
echo "Converting PDBQT files to PDB files"
cd ${DATA_DIR}/ligands_in_bound_pose/
echo "Changed to directory: " `pwd -P`
COUNTER=0
TOTAL=`ls -1 ${PDBQT_DIR} | wc -l`
for PDBQT_FILE in `ls $PDBQT_DIR/*.pdbqt`;
do
    (( COUNTER++ ))
    FILENAME=$(basename -- "$PDBQT_FILE")
    FILENAME="${FILENAME%.*}"
    DEST="${DATA_DIR}/ligands_in_bound_pose/${FILENAME}.pdb"
    printf "\r%b" "\033[2K"
    printf "Converting file ${COUNTER} of ${TOTAL}... " 
    $OBABEL -ipdbqt $PDBQT_FILE -opdb -O${DEST} -p7.4 -e1 &> /dev/null
done
# Remove underscores, which are problematic for DeepAtom
rename -- _ - *_*
echo "Done."

cd ${DATA_DIR}
echo "Changed to directory: " `pwd -P`

# Runs DeepAtom for those molecules
module load conda chimera
conda activate /blue/lic/seabra/.conda/envs/binding_affinity_27
DEEPATOM="/blue/lic/share/local/deepatom/bin/predict_binding_affinity_v4_2_data_split.sh"
echo "Running DeepAtom"
$DEEPATOM -t 'vs' -d $DATA_DIR 1> deepatom.log 2> deepatom.err
echo "#####################################################################"
echo "Done"
