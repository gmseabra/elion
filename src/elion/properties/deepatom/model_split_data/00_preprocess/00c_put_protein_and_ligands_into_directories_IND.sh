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

mkdir ../Dataset_VS
mv * ../Dataset_VS
mv ../Dataset_VS .

