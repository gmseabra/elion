# Estimator class
# -------------------------------
# Molecular Properties Prediction
# -------------------------------
# The 'self.properties' dict will contain OBJECTS to calculate properties and rewards.
# Each of those objects must implement at least 2 methods: 
#   1) 'predict': Gets an RDKit Mol object and returns a property value; and
#   2) 'reward' : Gets a property value and returns a reward. 

import importlib
import rdkit.Chem as Chem

class Estimators:

    def __init__(self, properties_cfg):
        self.properties = {}
        self.n_mols = 0
        for prop in properties_cfg:
            print("-"*50)
            module = importlib.import_module(f'properties.{prop}')
            module = getattr(module, prop)
            self.properties[prop] = module(prop,**properties_cfg[prop])
        print("Done reading properties.")
        print("="*80)
        print("\n")

    def estimate_properties(self,mols):
        """Calculates the properties for a list of molecules

        Args:
            mols ([RDKit ROMol]): List of RDKit ROMol objects
            properies (dict):  Dictionary with properties as keys and details
                            of predictor functions as values.
        Returns:
            Dict: Dictionary of properties as keys and predictions (floats) as values
        """
        pred = {}
        for _prop in self.properties.keys():
            pred[_prop] = []

        _mols = []
        _mols.extend(mols)
        self.n_mols = len(_mols)
        
        for _prop, _cls in self.properties.items():
            predictions = _cls.predict(mols)
            pred[_prop] = predictions
        return pred

    def estimate_rewards(self, predictions):
        """Calculates the rewards, given a dict of pre-calculated properties.

        Args:
            predictions (dict): Dictionary with properties as keys and lists of
                                predicted values as values.
            properties (_type_): Dictionary with properties as keys and details
                                of the property pbjects as values

        Returns:
            dict: Dict with property names as keys and rewards as values.
        """

        rew = {}
        for _prop, cls in self.properties.items():
            _values = predictions[_prop]

            if len(_values) != self.n_mols:
                msg = ( "ERROR: Something went wrong...\n"
                       f"Expecting {self.n_mols} values, but got only {len(_values)}"
                       f"for property {_prop}.")
                quit(msg)
                
            rew[_prop] = cls.reward(_values)
        rew["TOTAL"] = self.total_reward(rew)
        return rew

    def total_reward(self, rewards):

        total_rew = []
        for mol in range(self.n_mols):
            total_rew_mol = 0.0
            for _prop, cls in self.properties.items():
                this_rew = rewards[_prop][mol] * cls.rew_coeff
                total_rew_mol += this_rew
                    
            total_rew.append(total_rew_mol)
        return total_rew

    def smiles_reward_pipeline(self, smis, **kwargs):
        """
        Sometimes the RL process needs to pass the molecules as SMILES and needs
        to get the reward. This function does that.
        
        In sequence, 
           1) Generate molecules from SMILES
           2) Estimate properties
           3) Estimate rewards
           4) Calculate total reward
           Returns the total reward for each molecule in the list.

        Args:
            smis ([str]): SMILES for the molecules to be evaluated
            kwargs: Any other arguments to be passed to the property objects

        Returns:
            [float]: Total reward for each molecule in the list
        """

        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        predictions = self.estimate_properties(mols)
        rewards = self.estimate_rewards(predictions)
        total_reward = self.total_reward(rewards)
        return total_reward
                        
