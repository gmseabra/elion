#!/bin/bash

# Define the output file in the current directory
OUTPUT_FILE="./LGBM_suzuki_vina_3_no_smile.csv"

# Initialize the file with a header (optional, remove if not needed)
echo "Ligand_ID,Affinity" > "$OUTPUT_FILE"

# Loop through all directories matching the pattern *_p*_results
for dir in docking_score/LGBM_suzuki/3/3_p*_results; do
    # Check if it is actually a directory to avoid errors
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir..."
        
        # Loop through all output pdbqt files in the current directory
        for file in "$dir"/*_out.pdbqt; do
            # Check if file exists (handles empty directories)
            if [ -f "$file" ]; then
                
                # 1. Extract the ID: Get filename and remove "_out.pdbqt"
                filename=$(basename "$file")
                id="${filename%_out.pdbqt}"
                
                # 2. Extract the Score: Get the first Model's affinity ($4)
                # We use awk to find the line, print column 4, and exit immediately for speed
                score=$(awk '/REMARK VINA RESULT/ {print $4; exit}' "$file")
                
                # 3. Append to the summary CSV file
                if [ ! -z "$score" ]; then
                    echo "$id,$score" >> "$OUTPUT_FILE"
                fi
            fi
        done
    fi
done

echo "Done! Summary saved to $OUTPUT_FILE"