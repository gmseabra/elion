"""
This class implements simple policy gradient algorithm for
biasing the generation of molecules towards desired values of
properties aka Reinforcement Learninf for Structural Evolution (ReLeaSE)
as described in 
Popova, M., Isayev, O., & Tropsha, A. (2018). 
Deep reinforcement learning for de novo drug design. 
Science advances, 4(7), eaap7885.
"""

from generators.release.utils import canonical_smiles
from generators.release.stackRNN import StackAugmentedRNN
from generators.release.data import GeneratorData
import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem, RDLogger
from typing import Any, Tuple, Callable


class Reinforcement(object):
    def __init__(self, generator:StackAugmentedRNN, predictor:Any, get_reward:Callable):
        """
        Constructor for the Reinforcement object.

        Parameters
        ----------
        generator: object of type StackAugmentedRNN
            generative model that produces string of characters (trajectories)

        predictor: object of any predictive model type
            predictor accepts a trajectory and returns a numerical
            prediction of desired property for the given trajectory

        get_reward: function
            custom reward function that accepts a trajectory, predictor and
            any number of positional arguments and returns a single value of
            the reward for the given trajectory
            Example:
            reward = get_reward(trajectory=my_traj, predictor=my_predictor,
                                custom_parameter=0.97)

        Returns
        -------
        object of type Reinforcement used for biasing the properties estimated
        by the predictor of trajectories produced by the generator to maximize
        the custom reward function get_reward.
        """

        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward

    def policy_gradient(self, data:GeneratorData, n_batch:int=10, gamma:float=0.97,
                        std_smiles:bool=False, grad_clipping:bool=None, **kwargs)-> Tuple[float,float]:
        """
        Implementation of the policy gradient algorithm.

        Parameters:
        -----------

        data: object of type GeneratorData
            stores information about the generator data format such alphabet, etc

        n_batch: int (default 10)
            number of trajectories to sample per batch. When training on GPU
            setting this parameter to to some relatively big numbers can result
            in out of memory error. If you encountered such an error, reduce
            n_batch.

        gamma: float (default 0.97)
            factor by which rewards will be discounted within one trajectory.
            Usually this number will be somewhat close to 1.0.


        std_smiles: bool (default False)
            boolean parameter defining whether the generated trajectories will
            be converted to standardized SMILES before running policy gradient.
            Leave this parameter to the default value if your trajectories are
            not SMILES.

        grad_clipping: float (default None)
            value of the maximum norm of the gradients. If not specified,
            the gradients will not be clipped.

        kwargs: any number of other positional arguments required by the
            get_reward function.

        Returns
        -------
        total_reward: float
            value of the reward averaged through n_batch sampled trajectories

        rl_loss: float
            value for the policy_gradient loss averaged through n_batch sampled
            trajectories

        """
        rl_smiles = []
        rl_loss = 0
        self.generator.optimizer.zero_grad()
        total_reward = 0
        

        reward = 0
        trajectory = '<>'
 
        rl_smiles = self.generator.generate(n_batch)

        # # Generates and evaluates n_batch molecules
        # for gen_smi in range(n_batch):

        #     # Generates 1 molecule
        #     mol = None
        #     while mol == None:

        #         trajectory = self.generator.evaluate(data)

        #         # Check if the SMILES string is valid by trying to generate a
        #         # RDKIT Mol object from it. (The RDLogger statement disables 
        #         # throwing an error message if the conversion fails.)
                
        #         # Check that this SMILES is valid.
        #         # Sometimes a problem arises only after trying to
        #         # generate a new molecule from the sanitized smiles.
        #         # So, we need to:
        #         # 1. Create a Mol object from the raw SMILES and sanitize
        #         # 2. Create SMILES from this Mol object
        #         # 3. Try to create a new Mol from teh sanitized SMILES.
        #         # If the sanitization encounter an error, it will fail step #3. 

        #         RDLogger.DisableLog('rdApp.*')
        #         mol = Chem.MolFromSmiles(trajectory[1:-1], sanitize=True)
        #         if mol:
        #             new_canonic = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        #             mol = Chem.MolFromSmiles(new_canonic, sanitize=True)
        #         RDLogger.EnableLog('rdApp.*')
                
        #         if mol:
        #             canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        #             rl_smiles.append(canonical_smi)
        #--> END Molecule generation

        # Calculates rewards for all generated molecules
        rewards = self.get_reward(rl_smiles, kwargs)

        # Now go through the generated smiles and calculate the losses 
        for ind, smi in enumerate(rl_smiles):
            trajectory = '<' + smi + '>'

            # Converting string of characters into tensor
            trajectory_input = data.char_tensor(trajectory)
            discounted_reward = rewards[ind]
            total_reward += rewards[ind]

            # Initializing the generator's hidden state
            hidden = self.generator.init_hidden()
            if self.generator.has_cell:
                cell = self.generator.init_cell()
                hidden = (hidden, cell)
            if self.generator.has_stack:
                stack = self.generator.init_stack()
            else:
                stack = None

            # "Following" the trajectory and accumulating the loss
            for p in range(len(trajectory)-1):

                # This calls "forward" in the StackAugmentedRNN.
                output, hidden, stack = self.generator(trajectory_input[p], 
                                                       hidden, 
                                                       stack)
                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[p+1]
                rl_loss -= (log_probs[0, top_i]*discounted_reward)
                discounted_reward = discounted_reward * gamma
        #-->END MOLECULE GENERATION/EVALUATION

        # Doing backward pass and parameters update
        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch
        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                           grad_clipping)

        self.generator.optimizer.step()
        
        return total_reward, rl_loss.item()
