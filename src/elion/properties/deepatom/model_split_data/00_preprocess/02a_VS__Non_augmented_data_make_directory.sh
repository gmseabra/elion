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

ROOT_DIR=${DATASET_DIR}
dataset_name="Dataset_VS"
mkdir -p "${dataset_name}_Non_augmented" # create this directory only if it doesn't exist
cd   "$dataset_name"

NumCmplx=$(ls  -1 |  wc -l)

#=======================================================================================

complex_counter=0

for pdbid in $(ls  -1)
do
	complex_counter=$((complex_counter + 1))
	echo    "COMPLEX  ${pdbid} :   ${complex_counter}  out  of  ${NumCmplx}"
	cd  ${pdbid}
	cp     "${pdbid}_complex.pdb"     "${ROOT_DIR}/${dataset_name}_Non_augmented/${pdbid}.pdb"
	echo   "------------------------------------------------------"
	cd  ..
done

#=======================================================================================

