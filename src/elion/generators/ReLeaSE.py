#!/usr/bin/env python
__description__="""
    Simply generate new molecule SMILES using
    some generator.

    This code uses the ReLeaSE program from:
    https://github.com/isayev/ReLeaSE
"""

import time
import warnings
from pathlib import Path
import numpy as np
from rdkit import Chem

from elion import utils
from .AbstractGenerator import AbstractGenerator
from .release.release_generator import release_smiles_generator
from .release.reinforcement import Reinforcement


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

    def generate_smis(self):
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

    def generate_mols(self):
        """Generate molecules using the generator, and returns them as RDKit molecules."""
        generated = self.generate_smis()
        mols = [Chem.MolFromSmiles(smi) for smi in generated]
        return mols
    
    def bias_generator(self, ctrl_opts, estimator):
        """Bias the generator.

        Args:
            prop_values (_type_): _description_
        """
        print("\n")
        print("#"*80)
        print("#",f"{'BIASING GENERATOR':^76s}","#")
        print("#"*80)
        print("#",f"{ctrl_opts['comment'].upper():<76s}", "#")
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

        gen_start = 0
        # -------------------------------------------
        # History File
        # -------------------------------------------
        history_file = Path(ctrl_opts['history_file'])
        sep = ',' # Creates a CSV file
        history = {}
        for prop in estimator.properties.keys():
            history[f"{prop}_thr"] = []
            history[f"{prop}_avg"] = []

        if ctrl_opts['restart']:
            
            # Tries to find the info from the biasing_history file
            assert history_file.is_file(), f"File {history_file} does not exist !!"
            with open(history_file,'r', encoding='utf-8') as f:

                # Finds the last line
                last_line = ""
                for line in f:
                    last_line = line
                last_iteration = last_line.split(',')[0]
                gen_start  = int(last_iteration) + 1
            print("RESTARTING JOB FROM ITERATION", gen_start)
            
        else:
            # This is *not* a restart: Initialize the history file
            with open(history_file,'w', encoding='utf-8') as f:
                f.write("Iteration" + sep + sep.join(history.keys()) + '\n')

        # Checkpoints dir
        chk_dir = Path("./chk")
        Path.mkdir(chk_dir,exist_ok=True, parents=True)

        # -------------------------------------------
        # Initial (Unbiased) Batch
        # -------------------------------------------
        # First, let's generate a batch of random molecules,
        # and estimate their activity. This function also 
        # saves a SMI file with the generated molecules.
        # Unless it is a restart run, those molecules will
        # bw completely unbiased. Otherwie, they are from the
        # same distribution as the final batch of the previous
        # run.
        
        print("GENERATING INITIAL BATCH")
        smis_unbiased = self.generate_smis()
        mols_unbiased = [ Chem.MolFromSmiles(x) for x in smis_unbiased ] 

        # Predict properties for initial batch
        predictions_unbiased = estimator.estimate_properties(mols_unbiased)
        rewards_unbiased = estimator.estimate_rewards(predictions_unbiased)

        # Save history
        for prop in estimator.properties.keys():
            history[f"{prop}_thr"].append(estimator.properties[prop].threshold)
            history[f"{prop}_avg"].append(np.average(predictions_unbiased[prop]))
        # Dump to history
        with open(history_file,'a', encoding='utf-8') as f:
            out_str = (f"{gen_start:4d}" + sep
                        + sep.join(f"{history[key][0]:0.3f}" for key in history.keys()) 
                        + '\n')
            f.write(out_str)
        gen_start += 1

        # Print the predictions:
        print("\nUNBIASED GENERATOR")
        utils.print_results(smis_unbiased, predictions_unbiased, header="UNBIASED PROPERTIES")
        utils.print_results(smis_unbiased, rewards_unbiased, header="UNBIASED REWARDS")
        
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
        utils.save_smi_file("./chk/unbiased.smi", bucket_smiles, bucket_predictions)

        if seed_smi:
            print("Seeding the generator with SMILES:")
            print("< "+seed_smi+" >")

            # predict seed properties
            predictions_seed = estimator.estimate_properties([Chem.MolFromSmiles(seed_smi)])
            rewards_seed = estimator.estimate_rewards(predictions_seed)

            # Print the predctions for the seed:
            utils.print_results([seed_smi], predictions_seed, header="SEED PROPERTIES",include_stats=False)
            utils.print_results([seed_smi], rewards_seed, header="SEED REWARDS",include_stats=False)

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
        RL_action = Reinforcement(generator, 
                                  estimator.estimate_properties,
                                  estimator.smiles_reward_pipeline)

        # Bias the generator for 100 iterations towards the best molecules
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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            RL_start_time = time.time()

            iteration_accumulated_time = 0
            new_mols_since_last_train = 0

            for reinforcement_iteration in range(gen_start, gen_start + max_iterations):
                print("\n")
                print("#"*80)
                print(f"Reinforcement Iteration {reinforcement_iteration} of {gen_start + max_iterations - 1}.")
                print("-"*80, flush=True)
                iteration_start_time = time.time()

                for policy_iteration in range(n_policy):
                    print("\n")
                    print("#",f"{ctrl_opts['comment'].upper():<76s}", "#")
                    print(f"Policy Iteration {policy_iteration + 1} of {n_policy}.")
                    print(f"(RL Iteration {reinforcement_iteration} of {gen_start + max_iterations - 1})")
                    print("-"*80, flush=True)
                    print(f"{'Property':<34s} {'Threshold'}\t{'Limit':>8s}\t{'Converged?'}")
                    for prop in estimator.properties.keys():
                        print((f"    ---> {prop:<25s}: {estimator.properties[prop].threshold:>8.3f}"
                               f"\t{estimator.properties[prop].thresh_limit:>8.3f}"
                               f"\t{estimator.properties[prop].converged!s:>10s}"))
                    print("-"*80, flush=True)

                    # 1. Train the generator with the latest gen_data
                    print("Training generator with gen_data")
                    cur_reward, cur_loss = RL_action.policy_gradient(generator.gen_data, std_smiles=True)
                    print(f"  Average rewards = {cur_reward:8.2f}")
                    print(f"  Average loss    = {cur_loss:8.2f}")
                    print("\n")

                    # 2. Generate new molecules and predictions
                    smiles_cur = self.generate_smis()
                    mols = [Chem.MolFromSmiles(smi) for smi in smiles_cur]
                    predictions_cur = estimator.estimate_properties(mols)
                    rewards_cur = estimator.estimate_rewards(predictions_cur)

                    # Statistics on the predicitons
                    if verbosity > 1:
                        # Prints the complete table
                        utils.print_results(smiles_cur, predictions_cur, header="PROPERTIES", include_stats=True)
                        utils.print_results(smiles_cur, rewards_cur, header="REWARDS", include_stats=True)
                    else:
                        # Only print the stats
                        utils.print_stats(predictions_cur, header="PROPERTIES", print_header=True)
                        utils.print_stats(rewards_cur, header="REWARDS", print_header=True)

                    # 3. Adjusting the goal
                    # ---------------------
                    # At every iteration, we use the best results to retrain the generator
                    # fine-tuning it to generate more active molecules
                    rew_np  = np.array(rewards_cur['TOTAL'])
                    rew_avg = np.average(rew_np)
                    rew_std = np.std(rew_np) 

                    # Accept only molecules with maximum total reward
                    n_approved_mols = np.sum(rew_np == estimator.max_reward)
                    print(f" NUMBER OF MOLECULES   : {len(smiles_cur)}")
                    print(f" TOTAL REWARDS AVERAGE : {rew_avg:4.1f}")
                    print(f" TOTAL REWARDS STD_DEV : {rew_std:4.2f}")
                    print(f" N APPROVED MOLECULES  : {n_approved_mols} ({n_approved_mols /len(rew_np):4.1%})")            

                    #
                    # FINE TUNING THE GENERATOR
                    #
                    if n_approved_mols > 0:
                        # Hurray, we detected new molecules with good rewards!
                        # We save them in the gen_data.file, so that they will be used to train the generator
                        print(f"\nFound {n_approved_mols} molecules with good rewards ")
                        _smi = np.array(smiles_cur)
                        [generator.gen_data.file.append('<'+s+'>') for s in _smi[rew_np == estimator.max_reward]]
                        new_mols_since_last_train = new_mols_since_last_train + n_approved_mols
                        generator.gen_data.file_len = len(generator.gen_data.file)
                        print(f"New molecules since last training: {new_mols_since_last_train}")

                        # -----------------------------------------------------
                        # Filter gen_data.file and retrain the generator
                        # -----------------------------------------------------
                        # When enough new molecules have been generated, 
                        # remove from gen_data.file the molecules with rewards below the threshold
                        # and retrain the generator on the n_best ones left.
                        if (new_mols_since_last_train) > n_best:
                            # We reached a number of good estimates, so lets retrain the generator.
                            print(f"\n   --> Found {new_mols_since_last_train} new molecules below threshold.")
                            print(   "   --> Retraining generator...")

                            # 1. Obtain the reward estimates for each molecule *currently* in gen_data
                            #    (We've been adding molecules to the gen_data.file)
                            #    Notice the use of 'uncapped' rewards here. We want a full value for all rewards
                            #    to sort the molecules.

                            gen_data_mols = [Chem.MolFromSmiles(smi.strip('<>')) for smi in generator.gen_data.file]
                            gen_data_predictions = estimator.estimate_properties(gen_data_mols)
                            gen_data_rewards = estimator.estimate_rewards(gen_data_predictions)['TOTAL']

                            # 2. Get the indices of the n_best molecules with best reward.
                            #    We will keep only the 'n_best' molecules with the best total rewards
                            best_indices = np.argsort(-np.array(gen_data_rewards))[:n_best]

                            # 3. Substitutes the old gen_data.file with only those with best estimates:
                            new_data_file = []
                            new_rewards   = []
                            for molecule in best_indices:
                                new_data_file.append(generator.gen_data.file[molecule])
                                new_rewards.append(gen_data_rewards[molecule])
                            
                            # If we are seeding the generator, make sure the seed is in the smiles_set:
                            if seed_smi is not None and f"<{seed_smi}>" not in new_data_file:
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
                    estimator.check_and_adjust_thresholds(predictions_cur)
                    if estimator.all_converged:
                        break

                # --- END OF POLICY ITERATION.

                # Generate some molecules for stats. No need to save.
                smis = self.generate_smis()
                mols = [Chem.MolFromSmiles(smi) for smi in smis]
                predictions = estimator.estimate_properties(mols)
                rewards = estimator.estimate_rewards(predictions)

                # Save history
                for prop in estimator.properties.keys():
                    history[f"{prop}_thr"].append(estimator.properties[prop].threshold)
                    history[f"{prop}_avg"].append(np.average(predictions[prop]))

                # Dump history
                with open(history_file,'a', encoding='utf-8') as f:
                    out_str = (f"{reinforcement_iteration:4d}" + sep
                                + sep.join(f"{history[key][reinforcement_iteration-gen_start]:0.3f}" for key in history.keys()) 
                                + '\n')
                    f.write(out_str)
                
                # Dump stats on the generated mols
                print(f"\nFINISHED REINFORCEMENT ITERATION {reinforcement_iteration}")
                print("-"*55)
                utils.print_results(smis, predictions, header="PROPERTIES")
                utils.print_results(smis, rewards,     header="REWARDS")


                iteration_elapsed_time = time.time() - iteration_start_time
                iteration_accumulated_time = iteration_accumulated_time + iteration_elapsed_time
                print(f"|--> Elapsed time (this iteration) = {iteration_elapsed_time:0.3f} sec.")
                print(f"|--> Average time (all iterations) = {iteration_accumulated_time/(reinforcement_iteration):0.3f} sec./it")

                # Save a checkpoint file
                if (reinforcement_iteration % 2) == 0:

                    chk_path = f"./chk/biased_generator_{reinforcement_iteration:03}.chk"
                    smi_path = f"./chk/generated_smiles_{reinforcement_iteration:03}.smi"

                    print('\n-------------- Checkpoint --------------')
                    print(f"   Reinforcement Iteration {reinforcement_iteration}:")
                    print(f"   Saving generator to file {chk_path}" )
                    print(f"   and generated SMILES to  {smi_path}.")
                    print('-----------------------------------------\n')

                    utils.save_smi_file(smi_path, smis, predictions)
                    generator.save_model(chk_path)

                if estimator.all_converged:
                    print("\nAll properties converged. Stopping.")
                    break
                

        # RL Cycle finished, print results and quit.
        print("\n")
        print("#"*60)
        print(" FINISHED REINFORCED LEARNING CYCLE")
        print(f"TOTAL: {reinforcement_iteration - gen_start} iterations")
        print("#"*60)
        print(f"Generating final batch of {batch_size} molecules.")

        # Generate and save a final batch
        smi_file="./chk/generated_smiles_final.smi"
        chk_path="./chk/biased_generator_final.chk"

        smis = self.generate_smis()
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        predictions_final = estimator.estimate_properties(mols)
        rewards_final = estimator.estimate_rewards(predictions_final)
              
        generator.save_model(chk_path)
        utils.save_smi_file(smi_file, smis, predictions_final)
        print(f"Final batch of molecules saved to file {smi_file}.")

        # Dump stats on the generated mols
        print( "\nFINISHED BIASING GENERATOR")
        print(f"|--> Total number of iterations    = {reinforcement_iteration - gen_start}")
        print("-"*55)
        utils.print_results(smis, predictions_final, header="PROPERTIES")
        utils.print_results(smis, rewards_final,     header="REWARDS")

        # Total elapsed time
        RL_elapsed_time = time.time() - RL_start_time
        print(f"|--> Total Elapsed time            = {RL_elapsed_time:0.3f} sec.")
        print("\n\n===================================")
        print("  EXECUTION FINISHED SUCCESSFULLY")
        print("===================================")

        return
