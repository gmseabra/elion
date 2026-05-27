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
This code will test DeepAtom on the test directory.

# Testing parameters
n_test = 100      # Number of complexes to use for testing
n_reps = 5        # Number of repetitions for each complex

"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import shutil
import subprocess

def fmt_time(t):
    hours, rem = divmod(t, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}"

def test_deepatom(deepatom_home,orig_file_name,n_test, threshold):
    begin_time = time.time()

    # Files
    deepatom  = Path(deepatom_home,"deepatom/bin/run_deepatom.sh")
    test_path = Path(deepatom_home,"test")
    orig_file = Path(test_path, orig_file_name)

    # output file
    test_time  = time.strftime("%Y-%m-%d_%H-%M")
    test_results_file = Path(test_path,f"deepatom_test_{test_time}.tsv")

    #
    # Builds a test set of n_test complexes
    #
    orig_data_df = pd.read_csv(orig_file,delim_whitespace=True,comment="#")
    to_test_df   = orig_data_df.sample(n_test)

    n_reps = 0
    final_size = 0
    test_results = pd.DataFrame(columns=['System','orig','new'])

    for idx, row in to_test_df.iterrows():
        this_test = {}
        this_test['System'] = row.System.strip()
        this_test['orig']   = row.Calc

        this_test_path = Path(test_path,this_test['System'])
        print(this_test_path, this_test_path.exists())

        if this_test_path.exists():
            final_size += 1
            res_file = Path(this_test_path,f"vs_{this_test['System']}.csv")
            if res_file.exists():
                # Move the file to a new name
                shutil.move(res_file, Path(this_test_path,f"orig_vs_{this_test['System']}.csv"))

            # Now, runs deepatom for this test case 
            execute_deepatom = subprocess.run([deepatom,this_test_path], stdin=None, 
                                            input=None, stdout=None, stderr=None, 
                                            shell=False, timeout=None, check=False)
            print("DeepAtom Exit code = ", execute_deepatom)

            # move the results_file to a new name, to protect the original
            shutil.move(res_file, Path(this_test_path,f"new_vs_{this_test['System']}.csv"))
            res_file = Path(this_test_path,f"new_vs_{this_test['System']}.csv")
            # Reads the csv file created
            with open(res_file,'r') as rf:
                delG = 0.0
                n_reps = 0
                for line in rf:
                    tokens = line.strip().split(",")
                    delG += float(tokens[1])
                    n_reps += 1

            # Take the average of the repetitions
            this_test['new'] = delG/n_reps
                    
            # Finally, add this results to test_results
            test_results = test_results.append(this_test, ignore_index=True)

    # Calculates the errors
    test_results["diff" ] = test_results['orig'] - test_results['new']
    test_results["PASS?"] = test_results["diff"].apply(lambda a: 
                                                        "ok" if (abs(a) < threshold)
                                                        else "FAIL")
    # writes results in test_results_file
    test_results.to_csv(test_results_file, sep="\t", index=False, float_format="%5.2f")

    # Writes some extra stats to the output file
    test_pass = (test_results["PASS?"] == "ok"  ).sum()
    test_fail = (test_results["PASS?"] == "FAIL").sum()

    # Largest error:
    idx_max_err = test_results["diff"].idxmax()

    with open(test_results_file,'a') as tf:
        tf.write(f"# {'-'*53} \n")
        tf.write(f"# Finished test on: {time.strftime('%Y-%m-%d_%H-%M')} \n")
        tf.write(f"# TOTAL: {final_size} systems with {n_reps} repetitions each.\n")
        tf.write(f"# The largest difference was {test_results.iloc[idx_max_err]['diff']:6.2f}, "
                 f"for the case of {test_results.iloc[idx_max_err]['System']}.\n")
        tf.write(f"# {test_pass} tests passed.\n")
        tf.write(f"# {test_fail} tests failed.\n")

        # Print total timings
        elapsed_time = time.time() - begin_time
        tf.write(f"# Total time = {fmt_time(elapsed_time)} \n")
        tf.write(f"# Per system = {fmt_time(elapsed_time / n_test)} \n")

if __name__ == "__main__":

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='''Tests DeepAtom against a certain
                       number of known outputs, to check
                       if it is working as expected.''')
    parser.add_argument("--deepatom_home", type=Path,
                        default="~/work/source/deepatom",
                        help=("Full path to thh DeepAtom instalation "
                              "(default: %(default)s)."))
    parser.add_argument("--orig_data_file", type=Path,
                        default="deepatom_pdbbind_v2016_refined_2021-05-20_10-32.tsv",
                        help=("Name of the original data TSV file in the test_dir "
                              "(default: %(default)s)."))
    parser.add_argument("--n_test", '-n', type=int, default=10,
                        help="Number of systems to test (default: %(default)s).")
    parser.add_argument("--threshold", '-t', type=float,default="0.5",
                        help=("Maximum acceptable difference between the new and orig values "
                              "(default: %(default)s)."))

    args = parser.parse_args()

    deepatom_home  = Path(args.deepatom_home).expanduser()
    orig_data_file_name = Path(args.orig_data_file)
    n_test = args.n_test
    threshold = args.threshold


    test_deepatom(deepatom_home, orig_data_file_name, n_test, threshold)
