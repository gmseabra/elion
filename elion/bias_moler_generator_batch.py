#!/usr/bin/env python
"""
    Bias the MoLeR generator.

    This code uses the MoLeR molecular generator from:
    https://github.com/microsoft/molecule-generation
"""

import time
import yaml
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from typing import List, Dict

# Local
from properties import properties_and_rewards
from utils import print_progress_table, save_smi_file

##
"""
# Removed code left here in case I need it back.
        # #
        # # FINE TUNING THE GENERATOR
        # #
        # if n_improved_mols > 0:
        #     # Hurray, we detected new molecules with good rewards!
        #     print(f"\nFound {n_improved_mols} molecules with good rewards ")
        #     _smi = np.array(smiles_cur)
        #     [best_smiles.append(s) for s in _smi[rew_np == 15]]
        #     new_mols_since_last_train = new_mols_since_last_train + n_improved_mols

        #     # When enough new molecules have been generated, retrain the generator on the best ones
        #     print(f"New molecules since last training: {new_mols_since_last_train}")
        #     if (new_mols_since_last_train) > n_best:
        #         # We reached a number of good estimates, so lets retrain the generator.
        #         print(f"\n   --> Found {new_mols_since_last_train} new molecules below threshold.")
        #         print(   "   --> Retraining generator...")

        #         # 1. Obtain the reward estimates for each molecule currently in best_smiles
        #         #    Notice the use of 'uncapped' rewards here. We want a full value for all rewards
        #         #    to sort the molecules.
        #         best_predictions = properties_and_rewards.estimate_properties_parallel(best_smiles,reward_properties)
        #         best_rewards_total     = properties_and_rewards.estimate_rewards_batch(best_predictions,reward_properties)["TOTAL"]

        #         # 2. Get the indices of the n_best molecules with best reward.
        #         #    We will keep only the 'n_best' molecules with the best total rewards
        #         best_indices = np.argsort(-np.array(best_rewards_total))[:n_best]

        #         # 3. Substitutes the old best_smiles with only those with best estimates:
        #         new_best_smiles = []
        #         new_rewards     = []
        #         for molecule in best_indices:
        #             new_best_smiles.append(best_smiles[molecule])
        #             new_rewards.append(best_rewards_total[molecule])
                
        #         best_smiles = new_best_smiles
        #         best_rewards_total = new_rewards

        #         if cfg['control']['verbosity'] > 1:
        #             print("The new best molecules are:")
        #             for mol,rew in zip(best_smiles,best_rewards_total):
        #                 print(mol,rew)
        #         print(f"   --> Average rewards for the new molecules: {np.average(best_rewards_total)}")

        #         # Finally, fine-tune the generator with the n_best molecules
        #         #print(f"Fine-tuning generator with the best {n_best} molecules.")

        #         print("Setting the center of the latent space to ", best_smiles[0])
                
        #         #############################################
        #         ######           Fine-tuning           ######
        #         #############################################
        #         #fine_tune_moler(best_smiles)

        #         #quit()
        #         # RESET moler_state
        #         #print("\n")

        #         new_mols_since_last_train = 0

        # Finally, check and adjust the thresholds for the next iterations.
        # Only consider molecules that pass the mandatory checks (rewards_cur['TOTAL'] == 15)

"""

def bias_generator(cfg:Dict):

    reward_properties = cfg['reward_properties']
    moler_state       = cfg['generator']['initial_state']
    template_smi      = cfg['generator']['seed_smi']
    n_batch           = cfg['generator']['batch_size']
    gen_start         = cfg['control']['gen_start']
    n_iterations      = cfg['control']['max_iter']

    predictor = reward_properties['prob_active']['predictor']
    moler_sh  = Path(cfg['elion_root_dir'],'generators/moler/generate_moler_batch.sh')

    # Checkpoints dir
    chk_dir = Path("./chk")
    Path.mkdir(chk_dir,exist_ok=True, parents=True)

    # Keep the n_best molecules
    max_size = 200
    best_smiles = []
    best_rewards = {}
    best_rewards_total = []
    best_predictions = {}

    # Keep a history per iteration
    history = {}
    for prop in reward_properties.keys():
        history[f'{prop}_thresh'] = []
        history[prop] = []

    # Prepares a file for dumping partial stats
    sep = ','
    if not cfg['control']['restart']:
        with open("biasing_history.csv",'w') as f:
            f.write("Iteration" + sep + sep.join(history.keys()) + '\n')
    else:
        # If it is a restart, try to figure the next step number
        # ctrl_opts['restart'] == True:
        # Tries to find the info from the biasing_history file
        history_file = Path('biasing_history.csv')
        assert history_file.is_file(), f"File {history_file} does not exist !!"
        with open('biasing_history.csv','r') as f:

            # Finds the last line
            last_line = ""
            for line in f:
                last_line = line
            last_iteration = last_line.split(',')[0]
            gen_start  = int(last_iteration) + 1

        print("RESTARTING JOB FROM ITERATION", gen_start)

    # ---------------------------------------------------------------
    # INITIAL (UNBIASED) BATCH
    # ---------------------------------------------------------------
    print("\nUNBIASED GENERATOR")
    print(  "==================")

    print(f"Generating an initial (unbiased) batch of {n_batch} molecules... ", end='')
    result = subprocess.run([moler_sh, 
                            "-m", moler_state, 
                            "-n", str(n_batch), 
                            "-s", template_smi], capture_output=True)
    print("Done.")
    smiles_unbiased = result.stdout.decode('utf-8').split()
    smiles_unbiased = np.unique(np.array(smiles_unbiased))

    print(f"Obtained {len(smiles_unbiased)} unique molecules ",
          f"from a total of {n_batch} generated.")
    n_best = min(max_size,len(smiles_unbiased))

    # Properties and Rewards
    predictions_unbiased = properties_and_rewards.estimate_properties_parallel(smiles_unbiased,reward_properties)
    rewards_unbiased = properties_and_rewards.estimate_rewards_batch(predictions_unbiased,reward_properties)
    
    # Dump stats on the unbiased mols
    print_progress_table(reward_properties, predictions_unbiased, rewards_unbiased)

    # Save history
    for prop in reward_properties.keys():
        history[f'{prop}_thresh'].append(reward_properties[prop]["threshold"])
        history[prop].append(np.average(predictions_unbiased[prop]))

    # Dump history
    with open("biasing_history.csv",'a') as f:
        out_str = (f"{0:4d}" + sep
                    + sep.join(f"{history[key][0]:0.2f}" for key in history.keys()) 
                    + '\n')
        f.write(out_str)

    # Review and update the thresolds if needed
    properties_and_rewards.check_and_adjust_thresholds(predictions_unbiased, rewards_unbiased, reward_properties)
 
    # Select the n_best molecules by TOTAL reward.
    best_indices = np.argsort(-np.array(rewards_unbiased['TOTAL']))[:n_best]
    for molecule in best_indices:
        best_smiles.append(smiles_unbiased[molecule])
        best_rewards_total.append(rewards_unbiased['TOTAL'][molecule])

    # Save the best smiles to a file
    for prop in predictions_unbiased.keys():
        best_predictions[prop] = []
        best_rewards[prop] = []

        for molecule in best_indices:
            best_predictions[prop].append(predictions_unbiased[prop][molecule])
            best_rewards[prop].append(rewards_unbiased[prop][molecule])

    save_smi_file("./chk/unbiased.smi", best_smiles, best_predictions)

    #
    # Keep a list off molecules with that maximized the rewards
    # with current thresolds. This list will be updated anytime
    # the threshold changes.
    #
    max_rew = np.max(best_rewards_total)
    avg_rew = np.average(best_rewards_total)
    lowest_rew = np.min(best_rewards_total)

    max_mols = []
    max_rews = []
    max_idxs = []

    for idx in range(len(best_smiles)):
        if best_rewards_total[idx] == max_rew:
            max_idxs.append(idx)  
            max_mols.append(best_smiles[idx])
            max_rews.append(best_rewards_total[idx])

    print(f"\n  --> There are currently {len(best_smiles)} molecules ",
                f"stored in the 'best_smiles' array ")
    print(f"  --> The highest reward of the best_smiles is {max_rew}")
    print(f"  --> The average reward of the best_smiles is {avg_rew}")
    print(f"  --> The lowest  reward of the best_smiles is {lowest_rew}")
    print(f"  --> There are {len(max_mols)} molecules with the top rewards.")

    # ---------------------------------------------------------------
    # OPTIMIZATION CYCLE
    # ---------------------------------------------------------------

    iteration_accumulated_time = 0
    new_mols_since_last_train  = 0
    center_smi = best_smiles[0]
    center_rew = best_rewards_total[0]
    lowest_rew = best_rewards_total[-1]
    center_predictions = {}
    for prop in best_predictions.keys():
        center_predictions[prop] = best_predictions[prop][0]
    all_converged = False

    begin_iteration = gen_start + 1
    final_iteration = begin_iteration + n_iterations

    #for optimization_iteration in range(begin_iteration, final_iteration):

    optimization_iteration = begin_iteration
    while( (not all_converged) and optimization_iteration < final_iteration):
        iteration_start_time = time.time()
        print("\n")
        print("#"*80)
        print(f"Optimization Iteration {optimization_iteration} of {gen_start + n_iterations}.")
        print("-"*80, flush=True)
        print("Current thresholds:")
        for prop in reward_properties.keys():
            print(f"  ---> {prop:<25s}: {reward_properties[prop]['threshold']:>8.3f}")
        print(f"  ---> {'Current highest reward':<25s}: {center_rew}")
        print(f"  ---> {'Current lowest  reward':<25s}: {lowest_rew} \n")

        print(f"There are {len(max_mols)} molecules with the top rewards.")

        # Generate new molecules and predictions
        generated_smi = set()
        print(f"Generating a new batch of {n_batch} molecules (total).")
        print("-"*80, flush=True)

        # We will pass *all* molecules that maximize the rewards to 
        # MoLeR, and those will be the centers of the latent space for
        # the generation of new molecules.

        with open('max_mols.smi','w') as centers_file:
            centers_file.write('\n'.join(max_mols))
            centers_file.write('\n')
            
        # Now, we call MoLeR with this file
        print(f"\nGenerating a new batch of {n_batch} molecules: ")
        result = subprocess.run([moler_sh,
                                "-m", moler_state, 
                                "-n", str(n_batch), 
                                "-s", template_smi, 
                                "-c", 'max_mols.smi'],
                                capture_output=True)

        smiles_cur = result.stdout.decode('utf-8').split()
        smiles_cur = np.unique(np.array(smiles_cur))

        print(f"Done. Obtained {len(smiles_cur)} unique molecules ",
              f"from a total of {n_batch} molecules generated.")
        n_best = min(max_size,len(smiles_cur))

        predictions_cur = properties_and_rewards.estimate_properties_parallel(smiles_cur,reward_properties)
        rewards_cur     = properties_and_rewards.estimate_rewards_batch(predictions_cur,reward_properties)
        
        prob_actives_cur = np.array(predictions_cur['prob_active'])

        # Print progress
        print("\nPROPERTIES:  ", ', '.join([str(x) for x in predictions_cur]))
        if cfg['control']['verbosity'] > 1:
            # ----- (DEBUG) Print predictions -------------------------------------------------------
            print(f"\n(DEBUG): PREDICTIONS  ({len(smiles_cur)})")
            print("(DEBUG): SMILES  ", *[str(x) for x in predictions_cur])
            i=1
            for smi, *pred in zip(smiles_cur, *[predictions_cur[x] for x in predictions_cur]):
                print(f"(DEBUG): {i:3d} {smi:<100s}", ' '.join([f"{x:6.2f}" for x in pred]))
                i+=1
            print(f"(DEBUG): {'LENGTH':<104s}", ' '.join([f"{len(predictions_cur[x]):6d}" for x in predictions_cur]))
            # ----- (DEBUG) Print rewards ----------------------------------------------------------
            print(f"\n(DEBUG): REWARDS  ({len(smiles_cur)})")
            print("(DEBUG): SMILES  ", *[str(x) for x in rewards_cur])
            i=1
            for smi, *rewd in zip(smiles_cur, *[rewards_cur[x] for x in rewards_cur]):
                print(f"(DEBUG): {i:3d} {smi:<100s}", ' '.join([f"{x:6.2f}" for x in rewd]))
                i+=1
            print(f"(DEBUG): {'LENGTH':<104s}", ' '.join([f"{len(rewards_cur[x]):6d}" for x in rewards_cur]))

        # Statistics on the predicitons
        pred_np  = np.array([x for x in predictions_cur.values()])
        pred_avg = np.average(pred_np,1)
        pred_std = np.std(pred_np,1)
        print(f"PREDICTIONS: {'AVERAGES >':>103s} ", ' '.join([f"{x:6.2f}" for x in pred_avg]))
        print(f"             {'STANDARD DEVIATIONS >':>103s} ", ' '.join([f"{x:6.2f}" for x in pred_std]), flush=True)

        # Statistics on the rewards
        rew_np  = np.array([x for x in rewards_cur.values()])
        rew_avg = np.average(rew_np,1)
        rew_std = np.std(rew_np,1)
        print(f"REWARDS    : {'AVERAGES >':>103s} ", ' '.join([f"{x:6.2f}" for x in rew_avg]))
        print(f"             {'STANDARD DEVIATIONS >':>103s} ", ' '.join([f"{x:6.2f}" for x in rew_std]), flush=True)

        all_converged = properties_and_rewards.all_converged(reward_properties)
        if all_converged:
            print("All properties converged to the final intervals. Exiting.")
            break
        elif optimization_iteration >= final_iteration:
            print("Maximum number of iterations reached. Exiting.")
            break
        else:
            print("Thresholds not converged yet... Continue iterating.\n")

        # Adjusting the goal
        # ---------------------
        rew_np  = np.array(rewards_cur['TOTAL'])
        rew_avg = np.average(rew_np)
        rew_std = np.std(rew_np) 

        # Trying to accept only molecules with REWARD > rew_avg.
        # We are only interested in new molecules with rewards higher 
        # than the average.
        n_improved_mols = np.sum(rew_np > rew_avg)
        print("Current batch:")
        print(f"  --> NUMBER OF MOLECULES   : {len(prob_actives_cur)}")
        print(f"  --> TOTAL REWARDS AVERAGE : {rew_avg:4.1f}")
        print(f"  --> TOTAL REWARDS STD_DEV : {rew_std:4.2f}")
        print(f"  --> N IMPROVED MOLECULES  : {n_improved_mols} ({n_improved_mols /len(rew_np):4.1%})\n")         

        # If molecules were improved, we check if it's time to change the
        # thresholds and recalculate the rewards to choose the new LS center

        if n_improved_mols > 0:

            # 1. Update the list of n_best molecules
            print(f"\nFound {n_improved_mols} molecules with good rewards ")
            _smi = np.array(smiles_cur)

            best_len = len(best_smiles)
            print(f"The `best_smiles` array started with {best_len} unique molecules.")

            # Copies the new improved molecules to the 'best_smiles' array
            improved_idx = np.argsort(-np.array(rew_np))[:n_improved_mols]
            [best_smiles.append(s)  for s in _smi[improved_idx]  ]

            # Copies the preditions of the new improved molecules
            for prop in predictions_cur.keys():
                prop_np = np.array(predictions_cur[prop])
                rew_np  = np.array(rewards_cur[prop])
                [best_predictions[prop].append(p) for p in prop_np[improved_idx]]
                [best_rewards[prop].append(r) for r in rew_np[improved_idx]]
            [best_rewards_total.append(rt) for rt in np.array(rewards_cur['TOTAL'])[improved_idx]]
            best_rewards['TOTAL'] = best_rewards_total

            # Filter for unique molecules
            best_smiles, unique_idx = np.unique(best_smiles, return_index=True)
            for prop in predictions_cur.keys():
                prop_np = np.array(best_predictions[prop])
                rew_np  = np.array(best_rewards[prop])

                best_predictions[prop] = np.array(best_predictions[prop])[unique_idx]
                best_rewards[prop] = np.array(best_rewards[prop])[unique_idx]
            best_rewards_total = np.array(best_rewards['TOTAL'])[unique_idx]
            best_rewards['TOTAL'] = best_rewards_total

            print(f"From the {n_improved_mols} with rewards higher than the "
                  f"minimum, {len(best_smiles) - best_len} were new, "
                  f"and were added to the best_smiles array.")
            best_len = len(best_smiles)
            print(f"The `best_smiles` array now has {best_len} unique molecules.")
            n_best = min(max_size, best_len)
            
            # 2. Check and update the thresholds
            #properties_and_rewards.check_and_adjust_thresholds(predictions_cur, rewards_cur, reward_properties)
            properties_and_rewards.check_and_adjust_thresholds(best_predictions, best_rewards, reward_properties)

            # # 3. If the threshold was adjusted in any property, we recalculate the rewards
            moved_threshold = False
            for prop in reward_properties.keys():
                if reward_properties[prop]['moved_threshold']: moved_threshold = True

            if moved_threshold:
                print("\nSince thresholds changed, we need to recalculate rewards")

                # a. Recalculate the rewards given the new tresholds
                best_predictions = properties_and_rewards.estimate_properties_parallel(best_smiles,reward_properties)
                best_rewards = properties_and_rewards.estimate_rewards_batch(best_predictions,reward_properties)
                best_rewards_total = best_rewards['TOTAL']
                properties_and_rewards.check_and_adjust_thresholds(best_predictions, best_rewards, reward_properties)

            # Get the indices of the n_best molecules with best reward.
            #    We will keep only the 'n_best' molecules with the best total rewards

            # BEFORE TRIMMING, WE SHOULD SHUFFLE THE SMILES, SO THAT
            # THE ORDER OF THE SMILES IS RANDOM.
            # WHY? The new SMILES were just append to the end of the best_smiles array.
            #      If they have the same rewards as other SMILES, they will always come
            #      AFTER the old ones in the array and, when we trim it, we are always
            #      keeping the old SMILES. This will make the kept smilies random within
            #      the same reward range.
            random_order = np.random.permutation(len(best_smiles))
            best_smiles  = np.array(best_smiles)[random_order]
            for prop in best_predictions:
                best_predictions[prop] = np.array(best_predictions[prop])[random_order]
                best_rewards[prop]     = np.array(best_rewards[prop])[random_order]
            best_rewards['TOTAL'] = np.array(best_rewards['TOTAL'])[random_order]
            best_rewards_total = best_rewards['TOTAL']
            #       
            print(f"\nTRIMMING: Trimming to the best {n_best} molecules.")
            best_indices = np.argsort(-np.array(best_rewards_total))[:n_best]

            # c. Substitutes the old best_smiles with only those with best estimates:
            new_predictions = {}
            new_rewards     = {}
            
            new_best_smiles = [smi for smi in np.array(best_smiles)[best_indices]]
            for prop in best_predictions:
                new_predictions[prop] = [p for p in np.array(best_predictions[prop])[best_indices]]
                new_rewards[prop]     = [r for r in np.array(best_rewards[prop])[best_indices]]

            new_rewards['TOTAL'] = [tr for tr in np.array(best_rewards['TOTAL'])[best_indices]]
            best_smiles = new_best_smiles
            best_predictions = new_predictions
            best_rewards = new_rewards
            best_rewards_total = best_rewards['TOTAL']

            max_rew = np.max(best_rewards_total)
            avg_rew = np.average(best_rewards_total)
            lowest_rew = np.min(best_rewards_total)
        print(f"\n  --> There are currently {len(best_smiles)} molecules ",
                    f"stored in the 'best_smiles' array ")
        print(f"  --> The highest reward of the best_smiles is {max_rew}")
        print(f"  --> The average reward of the best_smiles is {avg_rew:0.2f}")
        print(f"  --> The lowest  reward of the best_smiles is {lowest_rew}")

        # Re-center the latent space of the generator.
        # With the MoLeR generator, instead of re-training we can just re-center
        # the latent space search around the best molecule.

        # Explore options
        # Make a selection of all molecules that maximized rewards

        # We now need to re-uild the list of best molecules.
        max_mols = []
        max_rews = []
        max_idxs = []

        for idx in range(len(best_smiles)):
            if best_rewards_total[idx] == max_rew:
                max_idxs.append(idx)  
                max_mols.append(best_smiles[idx])
                max_rews.append(best_rewards_total[idx])

        print(f"  --> There are {len(max_mols)} molecules out of {len(best_smiles)} ",
                    f"with the maximum reward of {max_rew}.")

        # Now, randomly select one molecule from max_mols for the next step
        idx = np.random.choice(len(max_mols))
        center_smi = max_mols[idx]
        center_rew = max_rews[idx]
        center_idx = max_idxs[idx]

        for prop in best_predictions.keys():
            center_predictions[prop] = best_predictions[prop][center_idx]

        # Dump stats on the generated mols
        print(f"\nFinished optimizing iteration {optimization_iteration}")
        print_progress_table(reward_properties,best_predictions,best_rewards)

        iteration_elapsed_time = time.time() - iteration_start_time
        iteration_accumulated_time = iteration_accumulated_time + iteration_elapsed_time
        print(f"|--> Elapsed time (this iteration) = {iteration_elapsed_time:0.3f} sec.")
        print(f"|--> Average time (all iterations) = {iteration_accumulated_time/(optimization_iteration):0.3f} sec./it")

        # Save history
        for prop in reward_properties.keys():
            history[f'{prop}_thresh'].append(reward_properties[prop]["threshold"])
            history[prop].append(np.average(best_predictions[prop]))

        # Dump history
        with open("biasing_history.csv",'a') as f:
            out_str = (f"{optimization_iteration:4d}" + sep
                        + sep.join(f"{history[key][optimization_iteration-gen_start-1]:0.2f}" for key in history.keys()) 
                        + '\n')
            f.write(out_str)
        
        # Save a checkpoint file
        if (optimization_iteration % 10) == 0:

            chk_path = f"./chk/restart.yml"
            smi_path = f"./chk/generated_smiles_{optimization_iteration:03}.smi"

            print('\n-------------- Checkpoint --------------')
            print(f"   optimization Iteration {optimization_iteration}:")
            print(f"   Saving generated SMILES to  {smi_path}.")
            print('-----------------------------------------\n')

            save_smi_file(smi_path, best_smiles, best_predictions)

            #with open(chk_path,'w') as rst_file:
            #    yaml.dump(cfg, rst_file, indent=5)
        
        optimization_iteration += 1
    # --- END OF OPTIMIZATION ITERATION.

    print("Optimization process reached the end.")
    print("Generating final set of molecules:")
    n_final = 1_000
    result = subprocess.run([moler_sh,
                            "-m", moler_state, 
                            "-n", str(n_final), 
                            "-s", template_smi, 
                            "-c", 'max_mols.smi'],
                            capture_output=True)
    print("Done.")

    smiles_cur = result.stdout.decode('utf-8').split()
    smiles_cur = np.unique(np.array(smiles_cur))

    print(f"Obtained {len(smiles_cur)} unique molecules ",
          f"from a total of {n_final} generated.")
    predictions_cur = properties_and_rewards.estimate_properties_parallel(smiles_cur,reward_properties)
    smi_path = f"./chk/generated_smiles_final.smi"
    save_smi_file(smi_path, smiles_cur, predictions_cur)
    print(f"Saved final results to file {smi_path}")
    print("-- Have a nice day! --")
    return
#----

def main():
    import input_reader
    import argparse

    #-- Command line arguments
    parser = argparse.ArgumentParser(
        description=''' Bias a generator using optimization learning
                    ''')

    parser.add_argument('-i','--input_file',
                        help='Path to the input file',
                        default='./input.yml')

    args = parser.parse_args()
    input_file = args.input_file
    # ---

    # Input File
    cfg = input_reader.read_input_file(input_file) 

    # Bias the generator
    bias_generator(cfg)

if __name__ == "__main__":
    main()





