#!/bin/bash


while getopts “d:” OPTION
do
     case $OPTION in
         d)
             DATASET_DIR=$OPTARG
             ;;
         ?)
             usage
             exit
             ;;
     esac
done

if [ -z "$DATASET_DIR" ]
then
     exit 1
fi

#===================================================================

cd  "${DATASET_DIR}/ligands_in_bound_pose"

for lig in $(ls -1 *.pdb)
do
	bar=${lig%.pdb}
	mkdir -p ../Dataset_VS/$bar
	cp $lig ../Dataset_VS/$bar/${bar}_ligand.pdb
	cp ../protein/*.pdb ../Dataset_VS/$bar/${bar}_protein.pdb
done

