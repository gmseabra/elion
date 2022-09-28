#!/usr/bin/env python
"""
    Bias a molecular generator using Reinforcement Learning.

    This code uses the ReLeaSE program from:
    https://github.com/isayev/ReLeaSE
"""

import sys
import warnings
from typing import List, Dict
import time
import numpy as np
import pandas as pd

from generators.release.release_generator import release_smiles_generator, print_torch_info
from generators.release.reinforcement import Reinforcement
from pathlib import Path


# Chemistry
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

# Local 
from utils import get_fingerprint_from_smiles, print_dict, print_progress_table, save_smi_file
from properties import properties_and_rewards

#-------------------------------------------

def print_preamble():
    print('\n')
    print("="*80)
    print("""
             Bias a molecular generator using Reinforcement Learning.
             This code uses the ReLeaSE program from:
             https://github.com/isayev/ReLeaSE
          """)
    print("="*80)
    print_torch_info()
    print("="*80)
    print('\n', flush=True)

    return

def get_fingerprint_list(smiles_list):

    fp_list = [get_fingerprint_from_smiles(x) for x in smiles_list]

    return fp_list

def generate_and_estimate(generator, predictor, n_to_generate):
    """
    1) generates n_to_generate number of SMILES strings
    2) Generates fingerprints for each SMILES string
    3) Generates the predicted score for each
    """

    #
    # THIS FUNCTION IS NO LONGER BEING USED
    #

    print("(DEBUG) function: generate_and_estimate ")
    print("(DEBUG) ------------------------------- ")
    print("(DEBUG) predictor     = ", predictor)
    print("(DEBUG) generator     = ", generator)
    print("(DEBUG) n_to_generate = ", n_to_generate)
    print("(DEBUG)  ")    
    
    generated = generator.generate(n_to_generate)
    #fingerprints = get_fingerprint_list(generated)
    #prediction = predictor.predict(fingerprints)  

    # CHEMBERT predicts directly from SMILES
    prediction = np.array(predictor.predict(generated))
    
    return generated, prediction

def generate_smiles_batch(generator, n_to_generate:int) -> List[str]:
    """Generates a new batch of SMILES strings using the 
       generator provided.

    Args:
        generator (generator): Generator objetc. Must have a 'generate' method
                               implemeted, that gets an int as the number of
                               molecules to generate.

        n_to_generate (int):   Number of new molecules to generate

    Returns:
        List[str]: The list of SMILES strings for the new molecules.
    """

    generated_smiles = generator.generate(n_to_generate)
    return generated_smiles

def get_total_reward_one(smiles:str, reward_properties:dict) -> int:
    """
    The reward function. Returns the reward for ONE molecule.
    """

    properties = properties_and_rewards.estimate_properties_one(smiles, reward_properties)
    reward = properties_and_rewards.estimate_capped_rewards_one(properties,reward_properties)["TOTAL"]

    return reward

def get_total_rewards_batch(smiles_list:List[str], reward_properties) -> list:
    print("Estimating rewards for", len(smiles_list), "new molecules.")
    properties = properties_and_rewards.estimate_properties_parallel(smiles_list, reward_properties)
    rewards = properties_and_rewards.estimate_capped_rewards_batch(properties, reward_properties)["TOTAL"]
    return rewards

def save_checkpoint():
    pass

def bias_generator(generator:release_smiles_generator, 
                   config_opts:Dict, 
                   gen_start:int=0, batch_size:int=100, n_best:int=100, 
                   n_to_generate:int=200, n_policy:int=15, n_iterations:int=100,
                   n_gen0:int=100):

    # General options
    verbosity = 1
    ctrl_opts = config_opts['control']
    seed_generator = 'seed_smi' in config_opts['generator'].keys()
    if seed_generator:
        seed_smi = config_opts['generator']['seed_smi'] 
        print("Will use the seed to bias the generator:")
        print("<"+seed_smi+">")

    # If it is a restart, try to figure the next step number
    if ctrl_opts['restart'] == True:

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

    # Rewards to use:
    reward_properties = config_opts['reward_properties']
    predictor = reward_properties['prob_active']['predictor']
   
    # Checkpoints dir
    chk_dir = Path("./chk")
    Path.mkdir(chk_dir,exist_ok=True, parents=True)

    # First, let's generate 1000 random molecules, and estimate 
    # their activity. This function also saves a SMI file with the generated molecules
    print("GENERATING UNBIASED BATCH")
    smiles_unbiased = generate_smiles_batch(generator,n_gen0)

    predictions_unbiased = properties_and_rewards.estimate_properties_parallel(smiles_unbiased,reward_properties)
    rewards_unbiased = properties_and_rewards.estimate_capped_rewards_batch(predictions_unbiased,reward_properties)

    # Dump stats on the generated mols
    print("\nUNBIASED GENERATOR")
    print_progress_table(reward_properties, predictions_unbiased, rewards_unbiased)
    

    # Create a list of the 'best' molecules generated by the unbiased generator, 
    # then use them to start biasing the generator. At every iteration,
    # this list gets updated with the best n_best molecules overall.

    best_smiles_unbiased = []
    best_predictions_unbiased = {}

    best_indices = np.argsort(-np.array(rewards_unbiased['TOTAL']))[:n_best]
    for molecule in best_indices:
        best_smiles_unbiased.append(smiles_unbiased[molecule])

    for prop in predictions_unbiased.keys():
        best_predictions_unbiased[prop] = []

        for molecule in best_indices:
            best_predictions_unbiased[prop].append(predictions_unbiased[prop][molecule])

    # Save the best smiles to a file
    save_smi_file("./chk/unbiased.smi", best_smiles_unbiased, best_predictions_unbiased)

    # Now, reset the gen_data.file to hold those n_best molecules
    # GenData requires the SMILES to be contained in <>:
    if seed_generator:
        # We seed the generator with an initial molecule
        for molecule in range(len(best_smiles_unbiased)):
            best_smiles_unbiased[molecule] = '<'+seed_smi+'>'
    else:
        for molecule in range(len(best_smiles_unbiased)):
            best_smiles_unbiased[molecule] = '<'+best_smiles_unbiased[molecule]+'>'

    generator.gen_data.file = best_smiles_unbiased
    generator.gen_data.file_len = len(best_smiles_unbiased)

    # #####################################################
    #            Reinforcement Learning Action
    # #####################################################
    # This defines the step of learning, and is the core
    # of the RL idea
    # RL_action = Reinforcement(generator, predictor, get_total_reward_one)
    RL_action = Reinforcement(generator, predictor, get_total_rewards_batch)

    # Now, bias the generator for 100 iterations towards the best molecules
    # (The filterwarnings is necessary to avoid litter)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if seed_generator:
            print("FITTING GENERATOR TO SEED MOLECULE:")
            print('<'+seed_smi+'>')
        else:
            print("FITTING GENERATOR TO BEST MOLECULES FROM UNBIASED SET.")
        losses = RL_action.generator.fit(generator.gen_data, 100, print_every=1000)
    
    ####################################################
    #               Reinforcement loop
    ####################################################
    # This is the main loop, the heart of the program.

    # Keep a history per iteration
    history = {}
    history['sco_thresh'] = []
    for prop in reward_properties.keys():
        history[prop] = []

    # Prepares a file for dumping partial stats
    sep = ','
    if not ctrl_opts['restart']:
        with open("biasing_history.csv",'w') as f:
            f.write("Iteration" + sep + sep.join(history.keys()) + '\n')

    reinforcement_iteration = 0
    #best_smiles, best_predictions = [] , []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        RL_start_time = time.time()

        iteration_accumulated_time = 0
        new_mols_since_last_train = 0

        for reinforcement_iteration in range(gen_start, gen_start + n_iterations):
            print("\n")
            print("#"*80)
            print(f"Reinforcement Iteration {reinforcement_iteration} of {gen_start + n_iterations}.")
            print("-"*80, flush=True)
            print("Thresholds:")
            for prop in reward_properties.keys():
                print(f"    ---> {prop:<25s}: {reward_properties[prop]['threshold']:>8.3f}")
            iteration_start_time = time.time()

            for policy_iteration in range(n_policy):
                print("\n")
                print(f"Reinforcement Iteration {reinforcement_iteration} of {gen_start + n_iterations}.")
                print(f"Policy Iteration {policy_iteration+1} of {n_policy}.")
                print(ctrl_opts['comment'].upper())
                print("-"*80, flush=True)

                # 1. Train the generator with the latest gen_data
                print("Training generator with gen_data")
                cur_reward, cur_loss = RL_action.policy_gradient(generator.gen_data, std_smiles=True, **reward_properties)
                print("  Average rewards = ", cur_reward)
                print("  Average loss    = ", cur_loss)
                print("\n")

                # 2. Generate new molecules and predictions
                smiles_cur = generate_smiles_batch(generator,n_to_generate)
                predictions_cur = properties_and_rewards.estimate_properties_parallel(smiles_cur,reward_properties)
                rewards_cur = properties_and_rewards.estimate_capped_rewards_batch(predictions_cur,reward_properties)
                prob_actives_cur = np.array(predictions_cur['prob_active'])

                print("PROPERTIES  ", *[str(x) for x in predictions_cur])
                if verbosity > 1:
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
                pred_np = np.array([x for x in predictions_cur.values()])
                pred_avg = np.average(pred_np,1)
                pred_std = np.std(pred_np,1)
                print(f"PREDICTIONS: {'AVERAGES >':>103s} ", ' '.join([f"{x:6.2f}" for x in pred_avg]))
                print(f"             {'STANDARD DEVIATIONS >':>103s} ", ' '.join([f"{x:6.2f}" for x in pred_std]), flush=True)

                # Statistics on the rewards
                rew_np = np.array([x for x in rewards_cur.values()])
                rew_avg = np.average(rew_np,1)
                rew_std = np.std(rew_np,1)
                print(f"REWARDS    : {'AVERAGES >':>103s} ", ' '.join([f"{x:6.2f}" for x in rew_avg]))
                print(f"             {'STANDARD DEVIATIONS >':>103s} ", ' '.join([f"{x:6.2f}" for x in rew_std]), flush=True)

                # 3. Adjusting the goal
                # ---------------------
                # At every iteration, we use the best results to retrain the generator
                # fine-tuning it to generate more active molecules
                rew_np  = np.array(rewards_cur['TOTAL'])
                rew_avg = np.average(rew_np)
                rew_std = np.std(rew_np) 

                # Trying to accept only molecules with REWARD == 15.
                n_approved_mols = np.sum(rew_np == 15)
                print(f" NUMBER OF MOLECULES   : {len(prob_actives_cur)}")
                print(f" TOTAL REWARDS AVERAGE : {rew_avg:4.1f}")
                print(f" TOTAL REWARDS STD_DEV : {rew_std:4.2f}")
                print(f" N APPROVED MOLECULES  : {n_approved_mols} ({n_approved_mols /len(rew_np):4.1%})")            

                #
                # FINE TUNING THE GENERATOR
                #
                # if n_below_thr > 0:
                if n_approved_mols > 0:
                    # Hurray, we detected new molecules with good rewards!
                    print(f"\nFound {n_approved_mols} molecules with good rewards ")
                    _smi = np.array(smiles_cur)
                    [generator.gen_data.file.append('<'+s+'>') for s in _smi[rew_np == 15]]
                    new_mols_since_last_train = new_mols_since_last_train + n_approved_mols
                    generator.gen_data.file_len = len(generator.gen_data.file)

                    # When enough new molecules have been generated, retrain the generator on the best ones
                    print(f"New molecules since last training: {new_mols_since_last_train}")
                    if (new_mols_since_last_train) > n_best:
                        # We reached a number of good estimates, so lets retrain the generator.
                        print(f"\n   --> Found {new_mols_since_last_train} new molecules below threshold.")
                        print(   "   --> Retraining generator...")

                        # 1. Obtain the reward estimates for each molecule currently in gen_data
                        #    (We've been adding molecules to the gen_data.file)
                        #    Notice the use of 'uncapped' rewards here. We want a full value for all rewards
                        #    to sort the molecules.
                        gen_data_mols = []
                        for mol in generator.gen_data.file:
                            gen_data_mols.append(mol.strip('<>'))
                        gen_data_predictions = properties_and_rewards.estimate_properties_parallel(gen_data_mols,reward_properties)
                        gen_data_rewards     = properties_and_rewards.estimate_rewards_batch(gen_data_predictions,reward_properties)["TOTAL"]

                        # 2. Get the indices of the n_best molecules with best reward.
                        #    We will keep only the 'n_best' molecules with the best total rewards
                        best_indices = np.argsort(-np.array(gen_data_rewards))[:n_best]

                        # 3. Substitutes the old gen_data.file with only those with best estimates:
                        new_data_file = []
                        new_rewards   = []
                        for molecule in best_indices:
                            new_data_file.append(f"<{gen_data_mols[molecule]}>")
                            new_rewards.append(gen_data_rewards[molecule])
                        
                        # If we are seeding the generator, make sure the seed is in the smiles_set:
                        if seed_generator:
                            new_data_file[ -1 ] = ('<'+seed_smi+'>')
                            new_rewards[   -1 ] = 0

                        generator.gen_data.file = new_data_file
                        generator.gen_data.file_len = len(generator.gen_data.file)
                        print(f"   --> LENGTH OF gen_data.file = {generator.gen_data.file_len}")

                        if verbosity > 1:
                            print("The new molecules in the gen_data are:")
                            for mol,rew in zip(new_data_file,new_rewards):
                                print(mol,rew)
                        print(f"   --> Average rewards for the new gen_data: {np.average(new_rewards)}")

                        # Finally, retrain the generator for 100 iterations
                        _ = RL_action.generator.fit(generator.gen_data, 100, print_every=1000)
                        new_mols_since_last_train = 0

                # Finally, check and adjust the thresholds for the next iterations.
                # Only consider molecules that pass the mandatory checks (rewards_cur['TOTAL'] == 15)
                properties_and_rewards.check_and_adjust_thresholds(predictions_cur, rewards_cur, reward_properties)

            # --- END OF POLICY ITERATION.

            # Generate some molecules for stats. No need to save.
            smi = generate_smiles_batch(generator,n_to_generate)
            # predictions = properties_and_rewards.estimate_properties_batch(smi,reward_properties)
            predictions = properties_and_rewards.estimate_properties_parallel(smi,reward_properties)
            rewards = properties_and_rewards.estimate_capped_rewards_batch(predictions,reward_properties)

            # Dump stats on the generated mols
            print(f"\nFINISHED BIASING ITERATION {reinforcement_iteration}")
            print("-"*55)
            print_progress_table(reward_properties,predictions,rewards)

            iteration_elapsed_time = time.time() - iteration_start_time
            iteration_accumulated_time = iteration_accumulated_time + iteration_elapsed_time
            print(f"|--> Elapsed time (this iteration) = {iteration_elapsed_time:0.3f} sec.")
            print(f"|--> Average time (all iterations) = {iteration_accumulated_time/(reinforcement_iteration+1):0.3f} sec./it")

            # Save history
            history['sco_thresh'].append(reward_properties['prob_active']["threshold"])

            for prop in reward_properties.keys():
                history[prop].append(np.average(predictions[prop]))

            # Dump history
            with open("biasing_history.csv",'a') as f:
                out_str = (f"{reinforcement_iteration:4d}" + sep
                            + sep.join(f"{history[key][reinforcement_iteration-gen_start]:0.3f}" for key in history.keys()) 
                            + '\n')
                f.write(out_str)
            
            # Save a checkpoint file
            if (reinforcement_iteration % 2) == 0:

                chk_path = f"./chk/biased_generator_{reinforcement_iteration:03}.chk"
                smi_path = f"./chk/generated_smiles_{reinforcement_iteration:03}.smi"

                print('\n-------------- Checkpoint --------------')
                print(f"   Reinforcement Iteration {reinforcement_iteration}:")
                print(f"   Saving generator to file {chk_path}" )
                print(f"   and generated SMILES to  {smi_path}.")
                print('-----------------------------------------\n')

                save_smi_file(smi_path, smi, predictions)
                generator.save_model(chk_path)
            
            # Check Convergence
            # if properties_and_rewards.all_converged(reward_properties): 
            #     print("ALL PROPERTIES CONVERGED.")
                
            #     break


    # RL Cycle finished, print results and quit.
    print("\n")
    print("#"*60)
    print(" FINISHED REINFORCED LEARNING CYCLE")
    print(f" TOTAL: {gen_start + reinforcement_iteration} iterations")
    print("#"*60)
    print(f"Generating final batch of {n_to_generate} molecules.")

    # Generate and save a final batch
    smi_file="./chk/generated_smiles_final.smi"
    chk_path="./chk/biased_generator_final.chk"

    smi = generate_smiles_batch(generator,n_to_generate)
    #predictions_final = properties_and_rewards.estimate_properties_batch(smi,reward_properties)
    predictions_final = properties_and_rewards.estimate_properties_parallel(smi,reward_properties)
    rewards_final = properties_and_rewards.estimate_capped_rewards_batch(predictions_final,reward_properties)
    
    generator.save_model(chk_path)
    print(f"Final batch of molecules saved to file {smi_file}.")

    # Dump stats on the generated mols
    print(f"\nFINISHED BIASING GENERATOR")
    print(f"|--> Total number of iterations    = {gen_start + reinforcement_iteration}")
    print("-"*55)
    print_progress_table(reward_properties,predictions_final,rewards_final)

    # Total elapsed time
    RL_elapsed_time = time.time() - RL_start_time
    print(f"|--> Total Elapsed time            = {RL_elapsed_time:0.3f} sec.")
    print("\n\n===================================")
    print("  EXECUTION FINISHED SUCCESSFULLY")
    print("===================================")

    return
#----

def main():
    import input_reader
    import argparse

    #-- Command line arguments
    parser = argparse.ArgumentParser(
        description=''' Bias a generator using reinforcement learning
                    ''')

    parser.add_argument('-i','--input_file',
                        help='Path to the input file',
                        default='./input.yml')

    args = parser.parse_args()
    input_file = args.input_file
    #----------------------

    print_preamble()

    # -----------------------------------------------------
    #
    #                     Input File
    #
    # -----------------------------------------------------
    cfg = input_reader.read_input_file(input_file)

    # General options
    ctrl_opts    = cfg['control']
    n_iterations = ctrl_opts['max_iter']

    # Rewards to use:
    reward_properties = cfg['reward_properties']

    # Docking Score Predictor
    model_type = reward_properties['prob_active']['model_type']
    model_file = reward_properties['prob_active']['model_file']

    if  model_type == "scikit-learn":
        # Model created with Scikit-Learn, e.g. a random-forest
        from sklearn.ensemble import RandomForestClassifier
        from joblib import load
        print(f"\nInitializing RF Model from file {model_file} ... ", end='')
        activity_model = load(model_file).set_params(n_jobs=1)
        print("Done.")

    elif model_type == "CHEMBERT":
        # CHEMBERT model
        from properties.activity.CHEMBERT.chembert import chembert_model

        print(f"\nInitializiing CHEMBERT with state from file {model_file} ... ", end='')
        activity_model = chembert_model(model_file)
        print("Done.")

    reward_properties['prob_active']['predictor'] = activity_model

    # Generator
    generator_opts = cfg['generator']
    generator = release_smiles_generator()
    initial_state = generator_opts['initial_state']
    print(f"\nLoading unbiased generator from file {initial_state} ... ", end='')
    generator.load_model(initial_state)
    print("Done.", flush=True)
    

    # Scaffold
    template_smiles_file = reward_properties['scaffold_match']['scaffold_file']
    print(f"\nLoading scaffold from {template_smiles_file}. ")
    with open(template_smiles_file,'r') as tf:
        template = tf.readline().strip() 
    template = Chem.MolFromSmarts(template)

    # This prints info, but also forces the info about rings to be calculated.
    # It is necessary because (a bug?) in RDKit that does not calculate the
    # infor about rings until they are requested (or printed)
    print(f"Atom  AtNum  InRing?  Arom?")
    for idx, atom in enumerate(template.GetAtoms()):
        print(f"{idx:>4d}  {atom.GetAtomicNum():5d}  {str(atom.IsInRing()):>7}  {str(atom.GetIsAromatic()):>5}")
    reward_properties['scaffold_match']['scaffold'] = template
    

    # Bias the generator
    bias_generator(generator,cfg, n_iterations=n_iterations)

if __name__ == "__main__":
    main()





