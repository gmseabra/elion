#!/bin/bash

# Generates a batch of molecules with the MoLeR generator
#
# Use this script to activate the appropriate conda env
# as needed, then run MoLeR inside this env.
#
# Usage:
# generate_moler.sh <model_path> <n_to_generate>
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate moler-env


POSITIONAL_ARGS=()
while [ $OPTIND -le "$#" ]
do
    if getopts m:n:s:c:d: option
    then
        case $option
        in
            m) MODEL_DIR="$OPTARG";;
            n) N_TO_GENERATE="$OPTARG";;
            s) SCAFFOLD_SMI="$OPTARG";;
			c) CENTER_SMI="$OPTARG";;
			d) STDEV="$OPTARG";;
		   \?) echo "Unrecognized option: $OPTARG"
		   	   exit 1;;
        esac
    else
        POSITIONAL_ARGS+=("${!OPTIND}")
        ((OPTIND++))
    fi
done

#echo "$script" "${POSITIONAL_ARGS[@]}"

if [[ -f "gen.err" ]]; then
    rm gen.err
fi


# Defaults to required parameters
MOLER_DIR=`dirname $0`
DEFAULT_MODEL_DIR="$MOLER_DIR/PRETRAINED_MODEL"
DEFAULT_N_TO_GENERATE=10

# Runs the generator
python ${MOLER_DIR}/generate_moler_batch.py ${MODEL_DIR:-$DEFAULT_MODEL_DIR} \
									  ${N_TO_GENERATE:-$DEFAULT_N_TO_GENERATE} \
	   								  ${SCAFFOLD_SMI:+"-s $SCAFFOLD_SMI"} \
									  ${CENTER_SMI:+"-c $CENTER_SMI"} \
									  ${STDEV:+"-e $STDEV"} \
									  2> gen.err
