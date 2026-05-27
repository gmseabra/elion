#!/usr/bin/env python

#  _____                         _
# |  __ \                   /\  | |
# | |  | | ___  ___ _ __   /  \ | |_ ___  _ __ ___
# | |  | |/ _ \/ _ \ '_ \ / /\ \| __/ _ \| '_ ` _ \
# | |__| |  __/  __/ |_) / ____ \ || (_) | | | | | |
# |_____/ \___|\___| .__/_/    \_\__\___/|_| |_| |_|
#                  | |
#                  |_|
#
"""
This code will test DeepAtom on a database, and calculate
some basic statistics. It takes:

# Path to original data files
data_path = Path("/home/seabra/work/source/deepatom/DATA")
deepatom  = Path("/home/seabra/work/source/deepatom/deepatom/bin/run_deepatom.sh")
csv_file  = Path("PDBbind_v2016_refined_core_INDEX.csv")
database  = Path(data_path,"INDEX",csv_file)
pdb_path  = Path(data_path,"pdbbind_v2016_refined")

# Path to test folder
test_path = Path("/home/seabra/work/source/deepatom/test")
test_results_file = Path(test_path,f"deepatom_test_{test_time}.tsv")

# Testing parameters
n_test = 100      # Number of complexes to use for testing
n_reps = 5        # Number of repetitions for each complex

"""

from pathlib import Path
import time
import numpy as np
import shutil
import subprocess

begin_time = time.time()
test_time  = time.strftime("%Y-%m-%d_%H-%M")


def fmt_time(t):
    hours, rem = divmod(t, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}"

# Path to original data files
data_path = Path("/home/seabra/work/source/deepatom/DATA")
deepatom  = Path("/home/seabra/work/source/deepatom/deepatom/bin/run_deepatom.sh")
csv_file  = Path("PDBbind_v2016_refined_core_INDEX.csv")
database  = Path(data_path,"INDEX",csv_file)
pdb_path  = Path(data_path,"pdbbind_v2016_refined")

# Path to test folder
test_path = Path("/home/seabra/work/source/deepatom/test")
test_results_file = Path(test_path,f"deepatom_test_{test_time}.tsv")

with open(test_results_file,'a') as tf:
    tf.write(f"{'System':<8} \t {'Exp':>6} \t {'Calc':>6} \t {'Error':>6} \n")

# Testing parameters
n_test = 100      # Number of complexes to use for testing
n_reps = 5        # Number of repetitions for each complex

#
# Builds a test set of n_test complexes
#
pdb_data = []
with open(database,'r') as infile:
    for line in infile:
        pdb_data.append( line.strip().split(",") )

rng = np.random.default_rng(seed=42)
pdb_data_idx = list( range( len(pdb_data) ) )
#idx_selection = rng.choice(len(pdb_data),n_test, replace=False)

test_results = []
mae = 0
mse = 0

#for this_pdb_idx in idx_selection:
while (len(test_results) < n_test) & (len(pdb_data_idx) > 0):

    this_pdb_idx = pdb_data_idx.pop(rng.choice(pdb_data_idx))
    this_test_name = pdb_data[this_pdb_idx][0]
    this_data_path = Path(data_path,pdb_path,this_test_name)
    print(pdb_data[this_pdb_idx], this_data_path.exists())

    if this_data_path.exists():

        this_data_delG = -1.36 * float(pdb_data[this_pdb_idx][1])
        this_results = [this_test_name, this_data_delG] 

        this_test_path = Path(test_path,this_test_name)
        res_file = Path(this_test_path,f"vs_{this_test_name}.csv")
        if res_file.exists(): res_file.unlink() # Removes the results csv file, if it already exists.

        # Copy the files to test_path, already building the correct structure
        prt_path = Path(this_test_path,'protein')
        lig_path = Path(this_test_path,'ligands_in_bound_pose')

        prt_path.mkdir(parents=True, exist_ok=True)
        lig_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(Path(this_data_path,this_test_name+"_protein.pdb"), prt_path)

        # Prepare 5 copies of the ligand to run
        for lig_copy in range(n_reps):
            shutil.copy(Path(this_data_path,this_test_name+"_ligand.pdb" ), 
                        Path(lig_path,f"copy-{lig_copy}.pdb"))

        # Now, runs deepatom for this test case 
        execute_deepatom = subprocess.run([deepatom,this_test_path], stdin=None, 
                                          input=None, stdout=None, stderr=None, 
                                          shell=False, timeout=None, check=False)
        print("DeepAtom Exit code = ", execute_deepatom)

        # Reads the csv file created
        with open(res_file,'r') as rf:
            delG = 0.0
            for line in rf:
                tokens = line.strip().split(",")
                #name = tokens[0]
                delG += float(tokens[1])

        # Take the average of the repetitions
        delG_avg = delG/n_reps
        this_results.append(delG_avg)

        # Error
        delG_err = this_data_delG - delG_avg
        this_results.append(delG_err)
        
        # Accumulate for error calculations
        mse += delG_err**2
        mae += delG_err

        # Finally, add this results to test_results
        test_results.append(this_results)

# Calculates the average error
final_size = len(test_results)
mse = mse / final_size
mae = mae / final_size

# writes results in test_results_file
with open(test_results_file,'a') as tf:
    for item in sorted(test_results):
        tf.write(f"{item[0]:8} \t {item[1]:6.2f} \t {item[2]:6.2f} \t {item[3]:6.2f}\n")
    tf.write(f"# {'-'*53} \n")
    tf.write(f"# Finished test on: {time.strftime('%Y-%m-%d_%H-%M')} \n")
    tf.write(f"# TOTAL: {final_size} systems with {n_reps} repetitions each.\n")
    tf.write(f"# MAE =  {mae:6.2f} \n")
    tf.write(f"# MSE =  {mse:6.2f} \n")

    # Print total timings
    elapsed_time = time.time() - begin_time
    tf.write(f"# Total time = {fmt_time(elapsed_time)} \n")
    tf.write(f"# Per system = {fmt_time(elapsed_time / n_test)} \n")

