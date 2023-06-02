#!/usr/bin/env python
__description__="""
    Simply generate new molecule SMILES using
    some generator.

    This code uses the ReLeaSE program from:
    https://github.com/isayev/ReLeaSE
"""

import time
import pandas as pd
import numpy as np
from rdkit import Chem
import warnings

from utils import print_results,save_smi_file
from generators.AbstractGenerator import AbstractGenerator
from generators.release.release_generator import release_smiles_generator, print_torch_info
from generators.release.reinforcement import Reinforcement

import importlib
from pathlib import Path

class ReLeaSE(AbstractGenerator):
    """The ReLeaSE Generator
    https://github.com/isayev/ReLeaSE
    """
    #-------------------------------------------

    def __init__(self,generator_properties):

        # General properties
        self.name = generator_properties['name']

        # Batch Size for generation and training
        self.batch_size = 200
        if "batch_size" in generator_properties.keys():
            self.batch_size = int(generator_properties['batch_size'])

        # Number of "best molecules" to keep in the training bucket.
        # Defaults to teh batch size.
        self.n_best = self.batch_size
        if "n_best" in generator_properties.keys():
            n_best = int(generator_properties['n_best'])
            if n_best > self.batch_size:
                msg = f"ERROR: n_best must be <= batch_size ({self.batch_size})"
                self.bomb_input(self.name, msg)

        # Number of policy microiterations per iteration
        self.n_policy=15
        if "n_policy" in generator_properties.keys():
            self.n_policy = int(generator_properties["n_policy"])

        # Maximum number of full iterations
        self.max_iterations=100
        if "max_iterations" in generator_properties.keys():
            self.max_iterations = int(generator_properties['max_iterations'])

        # Seed forthe generator
        self.seed_smi = False
        if 'seed_smi' in generator_properties.keys():
            self.seed_smi = generator_properties['seed_smi']

        # Initial state of the generator. Defaults to the original (unbiased) ReLeaSE
        # (Check the ReLeaSE GitHub page for details)
        initial_state = "generators/release/checkpoints/generator/checkpoint_biggest_rnn"
        if 'initial_state' in generator_properties.keys():
            initial_state = generator_properties['initial_state']
        self.initial_state = Path(initial_state)

        # The molecule generator from ReLeaSE
        self.generator = release_smiles_generator()
        print(f"\nLoading generator from file {initial_state} ... ", end='')
        self.generator.load_model(self.initial_state)
        print("Done.")

    def generate_mols(self):
        """
        Generates a sample of n_to_generate molecules using the provided generator.

        Args:
            generator (generator): Generator object to use. Must have a 'generate' function that
                                gets as argument a number and returns that number of molecules. 
            n_to_generate (int): Number of molecules to generate
        
        Returns:
            List of <n_to_generate> smiles strings.
        """
        generated = self.generator.generate(self.batch_size, verbose=1)
        return generated

    # def bias_generator(self,
    #                    config_opts:Dict, 
    #                    n_to_generate:int=200, 
    #                    n_gen0:int=100):

    def bias_generator(self, ctrl_opts, estimator):
        """Bias the generator.

        Args:
            prop_values (_type_): _description_
        """
        print("\n")
        print("#"*80)
        print("#",f"{'BIASING GENERATOR':^76s}","#")
        print("#"*80)
        print("#",f"Generator: {self.name:<65s}","#")
        print("#",f"Bucket size: {self.n_best:<63d}","#")
        print("#",f"Policy microiterations: {self.n_policy:<52d}","#")
        print("#",f"Max iterations: {self.max_iterations:<60d}","#")
        print("#",f"SMILES seed: {self.seed_smi:<63s}","#")
        print("#"*80)
        
        # Gets the generator
        generator  = self.generator

        # Generator parameters. All of those can be set by keywords.
        batch_size = self.batch_size
        n_best     = self.n_best
        n_policy   = self.n_policy
        seed_smi   = self.seed_smi
        max_iterations = self.max_iterations
        
        # General options
        verbosity = ctrl_opts['verbosity']
        # If it is a restart, try to figure the next step number
        gen_start = 0
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
    
        # Checkpoints dir
        chk_dir = Path("./chk")
        Path.mkdir(chk_dir,exist_ok=True, parents=True)

        # First, let's generate a batch of random molecules, and estimate 
        # their activity. This function also saves a SMI file with the generated molecules
        print("GENERATING UNBIASED BATCH")
        smis_unbiased = self.generate_mols()
        mols_unbiased = [ Chem.MolFromSmiles(x) for x in smis_unbiased ] 

        # Predict properties for initial batch
        predictions_unbiased = estimator.estimate_properties(mols_unbiased)
        rewards_unbiased = estimator.estimate_rewards(predictions_unbiased)

        # Print the predictions:
        print("\nUNBIASED GENERATOR")
        print_results(smis_unbiased, predictions_unbiased, header="UNBIASED PROPERTIES")
        print_results(smis_unbiased, rewards_unbiased, header="UNBIASED REWARDS")
        
        # Bucket
        # ------
        # Create a list of the 'best' molecules generated by the unbiased generator,
        # then use them to start biasing the generator. At every iteration,
        # this list gets updated with the best n_best molecules overall.
        #
        # Initially, it will contain the unbiased SMILES
        #
        # The list is kept sorted from highest to lowest rewards.

        bucket_smiles = []
        bucket_predictions = {}

        # Sort by rewards, and collect only the n_best results
        best_indices = np.argsort(-np.array(rewards_unbiased['TOTAL']))[:n_best]
        for molecule in best_indices:
            bucket_smiles.append(smis_unbiased[molecule])

        for prop in predictions_unbiased.keys():
            bucket_predictions[prop] = []

            for molecule in best_indices:
                bucket_predictions[prop].append(predictions_unbiased[prop][molecule])

        # Save the best smiles to a file
        save_smi_file("./chk/unbiased.smi", bucket_smiles, bucket_predictions)

        if seed_smi:
            print("Seeding the generator with SMILES:")
            print("< "+seed_smi+" >")

            # predict seed properties
            predictions_seed = estimator.estimate_properties([Chem.MolFromSmiles(seed_smi)])
            rewards_seed = estimator.estimate_rewards(predictions_seed)

            # Print the predctions for the seed:
            print_results([seed_smi], predictions_seed, header="SEED PROPERTIES",include_stats=False)
            print_results([seed_smi], rewards_seed, header="SEED REWARDS",include_stats=False)

            # Now we seed the generator with the initial molecule
            for idx, molecule in enumerate(bucket_smiles):
                # TO-DO: we can also seed only a fraction of the positions
                bucket_smiles[idx] = '<'+seed_smi+'>'
        else:
            for idx, molecule in enumerate(bucket_smiles):
                bucket_smiles[idx] = '<'+molecule+'>'

        # Bucket
        # Reset the gen_data.file to hold those n_best molecules
        generator.gen_data.file = bucket_smiles
        generator.gen_data.file_len = len(bucket_smiles)

        # #####################################################
        #            Reinforcement Learning Action
        # #####################################################
        # This defines the step of learning, and is the core
        # of the RL idea
        # RL_action = Reinforcement(generator, predictor, get_total_reward_one)
        RL_action = Reinforcement(generator, 
                                  estimator.estimate_properties,
                                  estimator.smiles_reward_pipeline)

        # Now, bias the generator for 100 iterations towards the best molecules
        # (The filterwarnings is necessary to avoid litter)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if seed_smi:
                print("FITTING GENERATOR TO SEED MOLECULE:")
                print('<'+seed_smi+'>')
            else:
                print("FITTING GENERATOR TO BEST MOLECULES FROM UNBIASED SET.")
            losses = RL_action.generator.fit(generator.gen_data, 100, print_every=1000)
        
        ####################################################
        #               Reinforcement loop
        ####################################################
        # This is the main loop, the heart of the program.
        #quit("STOP HERE")
        # Keep a history per iteration
        history = {}
        #history['sco_thresh'] = []
        for prop in estimator.properties.keys():
            history[f"{prop}_thr"] = []
            history[f"{prop}_avg"] = []
            #history[prop] = []

        # Prepares a file for dumping partial stats
        sep = ','
        if not ctrl_opts['restart']:
            with open("biasing_history.csv",'w') as f:
                f.write("Iteration" + sep + sep.join(history.keys()) + '\n')

        reinforcement_iteration = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            RL_start_time = time.time()

            iteration_accumulated_time = 0
            new_mols_since_last_train = 0

            for reinforcement_iteration in range(gen_start, gen_start + max_iterations):
                print("\n")
                print("#"*80)
                print(f"Reinforcement Iteration {reinforcement_iteration} of {gen_start + max_iterations}.")
                print("-"*80, flush=True)
                print("Thresholds:")
                for prop in estimator.properties.keys():
                    print(f"    ---> {prop:<25s}: {estimator.properties[prop].threshold:>8.3f}")
                iteration_start_time = time.time()

                for policy_iteration in range(n_policy):
                    print("\n")
                    print(f"Reinforcement Iteration {reinforcement_iteration} of {gen_start + max_iterations}.")
                    print(f"Policy Iteration {policy_iteration+1} of {n_policy}.")
                    print(ctrl_opts['comment'].upper())
                    print("-"*80, flush=True)

                    # 1. Train the generator with the latest gen_data
                    print("Training generator with gen_data")
                    cur_reward, cur_loss = RL_action.policy_gradient(generator.gen_data, std_smiles=True)#, **estimator.properties)
                    print("  Average rewards = ", cur_reward)
                    print("  Average loss    = ", cur_loss)
                    print("\n")

                    # 2. Generate new molecules and predictions
                    #smiles_cur = generate_smiles_batch(generator,n_to_generate)
                    #predictions_cur = properties_and_rewards.estimate_properties_parallel(smiles_cur,estimator.properties)
                    #rewards_cur = properties_and_rewards.estimate_capped_rewards_batch(predictions_cur,estimator.properties)
                    #prob_actives_cur = np.array(predictions_cur['prob_active'])

                    smiles_cur = self.generate_mols()
                    mols = [Chem.MolFromSmiles(smi) for smi in smiles_cur]
                    predictions_cur = estimator.estimate_properties(mols)
                    rewards_cur = estimator.estimate_rewards(predictions_cur)

                    print("PROPERTIES  ", *[str(x) for x in predictions_cur])
                    # Statistics on the predicitons
                    if verbosity > 1:
                        # ----- (DEBUG) Print predictions -------------------------------------------------------
                        print(f"\n(DEBUG): PREDICTIONS  ({len(smiles_cur)})")
                        print("(DEBUG): SMILES  ", *[str(x) for x in predictions_cur])
                        i=1
                        for smi, *pred in zip(smiles_cur, *[predictions_cur[x] for x in predictions_cur]):
                            print(f"(DEBUG): {i:3d} {smi:<100s}", ' '.join([f"{x:6.2f}" for x in pred]))
                            i+=1
                    pred_np = np.array([x for x in predictions_cur.values()])
                    pred_avg = np.average(pred_np,1)
                    pred_std = np.std(pred_np,1)
                    print(f"PREDICTIONS: {'AVERAGES >':>99s} ", ' '.join([f"{x:6.2f}" for x in pred_avg]))
                    print(f"             {'STANDARD DEVIATIONS >':>99s} ", ' '.join([f"{x:6.2f}" for x in pred_std]), flush=True)


                    # Statistics on the rewards
                    if verbosity > 1:
                        print(f"(DEBUG): {'LENGTH':<104s}", ' '.join([f"{len(predictions_cur[x]):6d}" for x in predictions_cur]))
                        # ----- (DEBUG) Print rewards ----------------------------------------------------------
                        print(f"\n(DEBUG): REWARDS  ({len(smiles_cur)})")
                        print("(DEBUG): SMILES  ", *[str(x) for x in rewards_cur])
                        i=1
                        for smi, *rewd in zip(smiles_cur, *[rewards_cur[x] for x in rewards_cur]):
                            print(f"(DEBUG): {i:3d} {smi:<100s}", ' '.join([f"{x:6.2f}" for x in rewd]))
                            i+=1
                        print(f"(DEBUG): {'LENGTH':<104s}", ' '.join([f"{len(rewards_cur[x]):6d}" for x in rewards_cur]))
                    rew_np = np.array([x for x in rewards_cur.values()])
                    rew_avg = np.average(rew_np,1)
                    rew_std = np.std(rew_np,1)
                    print(f"REWARDS    : {'AVERAGES >':>99s} ", ' '.join([f"{x:6.2f}" for x in rew_avg]))
                    print(f"             {'STANDARD DEVIATIONS >':>99s} ", ' '.join([f"{x:6.2f}" for x in rew_std]), flush=True)

                    # 3. Adjusting the goal
                    # ---------------------
                    # At every iteration, we use the best results to retrain the generator
                    # fine-tuning it to generate more active molecules
                    rew_np  = np.array(rewards_cur['TOTAL'])
                    rew_avg = np.average(rew_np)
                    rew_std = np.std(rew_np) 

                    # Trying to accept only molecules with REWARD == 15.
                    n_approved_mols = np.sum(rew_np == 15)
                    print(f" NUMBER OF MOLECULES   : {len(smiles_cur)}")
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
                            gen_data_predictions = properties_and_rewards.estimate_properties_parallel(gen_data_mols,estimator.properties)
                            gen_data_rewards     = properties_and_rewards.estimate_rewards_batch(gen_data_predictions,estimator.properties)["TOTAL"]

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
                    properties_and_rewards.check_and_adjust_thresholds(predictions_cur, rewards_cur, estimator.properties)

                # --- END OF POLICY ITERATION.

                # Generate some molecules for stats. No need to save.
                smi = generate_smiles_batch(generator,n_to_generate)
                # predictions = properties_and_rewards.estimate_properties_batch(smi,estimator.properties)
                predictions = properties_and_rewards.estimate_properties_parallel(smi,estimator.properties)
                rewards = properties_and_rewards.estimate_capped_rewards_batch(predictions,estimator.properties)

                # Dump stats on the generated mols
                print(f"\nFINISHED BIASING ITERATION {reinforcement_iteration}")
                print("-"*55)
                print_progress_table(estimator.properties,predictions,rewards)

                iteration_elapsed_time = time.time() - iteration_start_time
                iteration_accumulated_time = iteration_accumulated_time + iteration_elapsed_time
                print(f"|--> Elapsed time (this iteration) = {iteration_elapsed_time:0.3f} sec.")
                print(f"|--> Average time (all iterations) = {iteration_accumulated_time/(reinforcement_iteration+1):0.3f} sec./it")

                # Save history
                for prop in estimator.properties.keys():
                    history[f"{prop}_thr"].append(estimator.properties[prop]['threshold'])
                    history[f"{prop}_avg"].append(np.average(predictions[prop]))

                #history['sco_thresh'].append(estimator.properties['prob_active']["threshold"])

                #for prop in estimator.properties.keys():
                #    history[prop].append(np.average(predictions[prop]))

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
                # if properties_and_rewards.all_converged(estimator.properties): 
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
        #predictions_final = properties_and_rewards.estimate_properties_batch(smi,estimator.properties)
        predictions_final = properties_and_rewards.estimate_properties_parallel(smi,estimator.properties)
        rewards_final = properties_and_rewards.estimate_capped_rewards_batch(predictions_final,estimator.properties)
        
        generator.save_model(chk_path)
        print(f"Final batch of molecules saved to file {smi_file}.")

        # Dump stats on the generated mols
        print(f"\nFINISHED BIASING GENERATOR")
        print(f"|--> Total number of iterations    = {gen_start + reinforcement_iteration}")
        print("-"*55)
        print_progress_table(estimator.properties,predictions_final,rewards_final)

        # Total elapsed time
        RL_elapsed_time = time.time() - RL_start_time
        print(f"|--> Total Elapsed time            = {RL_elapsed_time:0.3f} sec.")
        print("\n\n===================================")
        print("  EXECUTION FINISHED SUCCESSFULLY")
        print("===================================")

        return
