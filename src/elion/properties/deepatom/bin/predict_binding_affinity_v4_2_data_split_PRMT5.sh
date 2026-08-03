#!/bin/bash
#------------------------------------------------------------------------
#       Script created by: Mohammad A. Rezaei [2020.02.10]
#       Biophysics Graduate Program, Ohio State University
#       Chemistry Graduate Program, University of Florida
#
#       Script updated and maintained by: 
#           Gustavo Seabra
#           Department of Medicinal Chemistry
#           College of Pharmacy
#           University of Florida
#------------------------------------------------------------------------


###############################################################################
#                                                                             #
#                                CONFIGURATION                                #
#                                                                             #
###############################################################################
# Only need to modify the following 2 lines according to your installation:

# 1. Where do you want the scratch files to be written. Make sure the folder exists and
#    is writeable by you.
SCRATCH_ROOT_DIR="/home/huangzihang/repos/user_data/deepatom"

# 2. Location of the DeepAtom installation.
DEEPATOM_ROOT_DIR="/home/huangzihang/repos/elion/src/elion/properties"

###############################################################################
#                NO NEED TO MODIFY ANYTHING BELOW THIS POINT                  #
###############################################################################

usage()
{
cat << EOF


OPTIONS:
   -h      Show this message
   -t      Type of input (vs / dlg / ind / ind_prep)
   -d      directory to be predicted (full address)


EOF
}

while getopts “:ht:d:” OPTION
do
     case $OPTION in
         h)
             usage
             exit 1
             ;;
         t)
             input_type=$OPTARG
             ;;
         d)
             input_dir=$OPTARG
             ;;
         ?)
             usage
             exit
             ;;
     esac
done

if [ -z "$input_type" ] || [ -z "$input_dir" ]
then
     usage
     exit 1
fi

#===================================================================

start_whole=$(date +%s)
SCRATCH_DIR="$SCRATCH_ROOT_DIR/DEEP_MODEL_temp.$$"
SCRIPTS_DIR="$DEEPATOM_ROOT_DIR/deepatom/model_split_data"
PRE_DIR="${SCRIPTS_DIR}/00_preprocess"
GEN_DIR="${SCRIPTS_DIR}/01_generate_channels"
PRED_DIR="${SCRIPTS_DIR}/02_pytorch"

mkdir -p ${SCRATCH_DIR}

#conda activate binding_affinity_27

cd "${input_dir}/ligands_in_bound_pose"
ls -1 > "${input_dir}/ligands_list.txt"

foo=${input_dir%/}
dataset_name=${foo##*/}
echo "dataset_name:  ${dataset_name}"
DATASET_DIR="${SCRATCH_DIR}/${input_type}/${dataset_name}"
echo "DATASET_DIR:  ${DATASET_DIR}"
output_list_splits_dir="${DATASET_DIR}/lists"
mkdir -p "${output_list_splits_dir}"


num_ligs=$(cat "${input_dir}/ligands_list.txt" | wc -l)
echo "num_ligs:  ${num_ligs}"

#=======================================================================================
# find number of CPUs on the machine
if [[ -z $NR_CPUS ]]
then
	NR_CPUS=$(grep -c "^processor" /proc/cpuinfo)
fi

# allow 4 cores be free
NR_CPUS=$((NR_CPUS-4))

#=======================================================================================


#batch_size=$(grep -c "^processor" /proc/cpuinfo)  # equal to number of cores on computer
batch_size="${NR_CPUS}"
echo "batch_size:  ${batch_size}"

#----------------------------------------------------------------------------------------

split    -d     \
         -a   5    \
         -l   "$batch_size"   \
         "${input_dir}/ligands_list.txt"   \
         "${output_list_splits_dir}"/split_

#  -a, --suffix-length=N   use suffixes of length N (default 2)
#  -b, --bytes=SIZE        put SIZE bytes per output file
#  -C, --line-bytes=SIZE   put at most SIZE bytes of lines per output file
#  -d, --numeric-suffixes  use numeric suffixes instead of alphabetic
#  -l, --lines=NUMBER      put NUMBER lines per output file

# rm  -f   "${input_dir}/ligands_list.txt"

#----------------------------------------------------------------------------------------

cd "${output_list_splits_dir}"


for split_list in $(ls -1 "${output_list_splits_dir}"/split_*)
do

     start_batch=$(date +%s)

     batch_name=${split_list##*split_}
     echo "batch_name:  ${batch_name}"

     batch_dir="${DATASET_DIR}/${batch_name}"
     echo "batch_dir:  ${batch_dir}"

     mkdir -p "${batch_dir}"

     cp -r "${input_dir}/protein"  "${batch_dir}/"
     mkdir -p "${batch_dir}/ligands_in_bound_pose"

     while read -r lig
     do

          echo "LIG:  ${lig}"
          ln -s "${input_dir}/ligands_in_bound_pose/${lig}"  "${batch_dir}/ligands_in_bound_pose/${lig}"

     done < "${output_list_splits_dir}/split_${batch_name}"



     cd "${batch_dir}"

     if [ "$input_type" = "vs" ]  # i.e. multiple ligands will bind to a single protein
     then
          #-------------------------------------------------------------------------------------------------------
          /bin/bash "${PRE_DIR}/00_put_protein_and_ligands_into_directories.sh"  -d "${batch_dir}"

          python "${PRE_DIR}/01_preprocess_complexes_VS.py"  "${batch_dir}"

          python "${PRE_DIR}/pipeline_VS.py" --batch-dir "${batch_dir}" --pre-dir "${PRE_DIR}" --scripts-dir "${SCRIPTS_DIR}"

          # /bin/bash "${PRE_DIR}/02a_VS__Non_augmented_data_make_directory.sh"  -d "${batch_dir}"

          # /bin/bash "${PRE_DIR}/02b_augment_no_Chimera_VS_opt.sh"  -d "${batch_dir}"  -p "${PRE_DIR}"

          # /bin/bash "${PRE_DIR}/03a_generate_atomtypes_VS_Non_augmented.sh"  -d "${batch_dir}"  -s "${SCRIPTS_DIR}"

          # /bin/bash "${PRE_DIR}/03b_generate_atomtypes_VS_augmented.sh"  -d "${batch_dir}"  -s "${SCRIPTS_DIR}"

          python "${GEN_DIR}/make_grid_mp.py"  "${batch_dir}"  "${dataset_name}"

          python "${GEN_DIR}/make_grid_for_aug_mp.py"  "${batch_dir}"  "${dataset_name}"

          python "${PRED_DIR}/test.py"  \
             --batch_dir="${batch_dir}"  \
             --test_type="${input_type}"  \
             --test_dataset="${dataset_name}"  \
             --model_dir="${SCRIPTS_DIR}/model"  \
             --out_csv_dir="${input_dir}"
          #-------------------------------------------------------------------------------------------------------

     fi

     cd "${output_list_splits_dir}"
     # rm -rf  "${batch_dir}"

     end_batch=$(date +%s)
     runtime_batch=$((end_batch-start_batch))
     echo "Runtime (Batch) = ${runtime_batch} seconds."

done


# rm -rf "${DATASET_DIR}"

end_whole=$(date +%s)
runtime_whole=$((end_whole-start_whole))
echo "Runtime (Total) = ${runtime_whole} seconds."


