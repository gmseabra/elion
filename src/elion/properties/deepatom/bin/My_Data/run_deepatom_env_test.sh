#!/bin/bash

# Test script to run the binding affinity prediction script 10 times
# and check if the mean of outputs is within an acceptable standard deviation

# Path to the original script and output file
# SCRIPT="/blue/lic/share/local/deepatom/deepatom/bin/predict_binding_affinity_v4_2_data_split.sh"
SCRIPT="/blue/lic/huangzihang/repos/deepatom/bin/predict_binding_affinity_v4_2_data_split.sh"
SCRIPT_ARGS="-t vs -d /blue/lic/huangzihang/repos/deepatom/bin/My_Data"
OUTPUT_FILE="/blue/lic/huangzihang/repos/deepatom/bin/My_Data/vs_My_Data.csv"

# Clear the output file to start fresh
> "$OUTPUT_FILE"

# Array to store binding affinity values
declare -a values

# Run the script 10 times
for i in {1..10}
do
    echo "Running iteration $i..."
    # Run the script in the specified conda environment
    conda activate /blue/lic/huangzihang/repos/miniconda3/envs/binding_affinity_27
    bash $SCRIPT $SCRIPT_ARGS
    conda deactivate
    
    # Extract the last line's binding affinity value (second column of CSV)
    value=$(tail -n 1 "$OUTPUT_FILE" | cut -d',' -f2)
    values+=("$value")
done

# Calculate mean using awk
mean=$(printf "%s\n" "${values[@]}" | awk '{sum+=$1} END {print sum/NR}')

# Calculate standard deviation using awk
stddev=$(printf "%s\n" "${values[@]}" | awk -v mean="$mean" \
    '{sum+=($1-mean)^2} END {print sqrt(sum/(NR-1))}')

echo "Outputs: ${values[@]}"
echo "Mean: $mean"
echo "Standard Deviation: $stddev"

# Define acceptable standard deviation
ACCEPTABLE_STDDEV=0.5

# Check if standard deviation is within acceptable range
if (( $(echo "$stddev <= $ACCEPTABLE_STDDEV" | bc -l) )); then
    echo "Test Passed: Standard deviation ($stddev) is within acceptable range (<= $ACCEPTABLE_STDDEV)."
else
    echo "Test Failed: Standard deviation ($stddev) exceeds acceptable range ($ACCEPTABLE_STDDEV)."
    exit 1
fi