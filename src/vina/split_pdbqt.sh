#!/bin/bash

# Define paths
SRC_DIR="docking_score/LGBM_suzuki/3/pdbqt"
OUT_BASE="docking_score/LGBM_suzuki/3"
NUM_SPLITS=390

# 1. Collect all pdbqt files into an array
mapfile -t files < <(find "$SRC_DIR" -maxdepth 1 -name "*.pdbqt")
total_files=${#files[@]}

if [ "$total_files" -eq 0 ]; then
    echo "No .pdbqt files found in $SRC_DIR. Check your path!"
    exit 1
fi

# 2. Calculate distribution
# Using ceiling division: (total + (n-1)) / n
files_per_folder=$(( (total_files + NUM_SPLITS - 1) / NUM_SPLITS ))

echo "Total files: $total_files"
echo "Splitting into $NUM_SPLITS folders (~$files_per_folder files each)"

# 3. Create folders and copy files
# Loop from 0 to 31
for (( i=0; i<$NUM_SPLITS; i++ )); do
    # Creating padded folder names (e.g., split_part_1)
    folder_num=$((i + 1))
    target_dir="$OUT_BASE/split_part_$folder_num"
    
    mkdir -p "$target_dir"
    
    # Calculate array slice indices
    start=$((i * files_per_folder))
    
    # Check if we have files left to copy for this batch
    if [ $start -lt $total_files ]; then
        # Copy the slice of the array to the target directory
        # This handles the last folder automatically even if it has fewer files
        cp "${files[@]:$start:$files_per_folder}" "$target_dir/"
        echo "Copied batch $folder_num to $(basename "$target_dir")"
    fi
done

echo "Process complete. 32 folders created in $OUT_BASE"