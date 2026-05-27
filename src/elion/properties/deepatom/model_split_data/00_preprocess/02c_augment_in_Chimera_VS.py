import os
import sys
import numpy as np
from random import randint
from random import uniform
from random import random
from chimera import runCommand as rc  # use 'rc' as shorthand for runCommand


augmented_dir = os.path.join('../../Dataset_VS_augmented')

NUMBER_OF_SCANNED_ROT_ANGLES = 6
NUMBER_OF_SCANNED_TURN_AXES = 6


def get_ligand_center(lig_pdb):
    # Validate file existence
    if not os.path.exists(lig_pdb):
        raise FileNotFoundError("Ligand PDB file not found: {}".format(lig_pdb))

    lig_coords = []
    
    # Open file in text mode (or decode if binary mode is required)
    with open(lig_pdb, 'r') as ligFile:  # Changed from 'rb' to 'r'
        for line in ligFile:
            # Ensure line is a string (decode if necessary)
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if line.startswith(("ATOM ", "HETATM")):
                try:
                    x_coord = float(line[30:38].strip())
                    y_coord = float(line[38:46].strip())
                    z_coord = float(line[46:54].strip())
                    lig_coords.append((x_coord, y_coord, z_coord))
                except ValueError:
                    print("Warning: Skipping invalid coordinate line in {}: {}".format(lig_pdb, line.strip()))
                    continue
    
    # Convert to NumPy array and validate
    lig_coords = np.array(lig_coords)
    if lig_coords.size == 0:
        raise ValueError("No valid ATOM or HETATM coordinates found in {}".format(lig_pdb))
    
    print("lig_coords shape:", lig_coords.shape)  # Debug output
    maxc = np.squeeze(np.max(lig_coords, axis=0))
    minc = np.squeeze(np.min(lig_coords, axis=0))
    lig_center = (maxc + minc) / 2.0
    
    return lig_center


# Validate command-line arguments
if len(sys.argv) < 3:
    raise ValueError("Usage: python script.py <ligand_pdb> <complex_pdb>")

lig_pdb = sys.argv[1]
cmplx_pdb = sys.argv[2]

pdb_code = cmplx_pdb.split('_')[0]

# Ensure output directory exists
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Get ligand center
lig_center_x, lig_center_y, lig_center_z = get_ligand_center(lig_pdb)

START_ANGLE = -180
END_ANGLE = 180
step = (END_ANGLE - START_ANGLE) / NUMBER_OF_SCANNED_ROT_ANGLES

rc("open " + cmplx_pdb)

print("COMPLEX: " + str(cmplx_pdb))
print("="*20)

sample_id = -1  # so first index for samples starts at zero

for rot_angle in range(START_ANGLE, END_ANGLE, step):
    rot_angle += randint(-step, step)
    for turn_axis in range(NUMBER_OF_SCANNED_TURN_AXES):
        sample_id += 1
        cmplx_pdb_augmented = "{0}_augmented_{1}.pdb".format(pdb_code, str(sample_id))
        output_file = os.path.join(augmented_dir, cmplx_pdb_augmented)

        # Axis used for rotation
        rot_x = int(1000 * random())
        rot_y = int(1000 * random())
        rot_z = int(1000 * random())

        turn_command = "turn " + str(rot_x) + "," + str(rot_y) + "," + str(rot_z) + \
                       " " + str(rot_angle) + " center " + str(lig_center_x) + \
                       "," + str(lig_center_y) + "," + str(lig_center_z) + " models #0"

        rc(turn_command)

        # Axis used for translation
        move_x = int(1000 * random())
        move_y = int(1000 * random())
        move_z = int(1000 * random())

        move_length = uniform(-1.0, 1.0)  # Random float x, -1.0 <= x < 1.0

        move_command = "move " + str(move_x) + "," + str(move_y) + "," + str(move_z) + \
                       " " + str(move_length) + " models #0"

        rc(move_command)

        rc("select #0")
        rc("write selected #0 " + output_file)
        
rc("close all")