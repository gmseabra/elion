# Estimator class
# -------------------------------
# Molecular Properties Prediction
# -------------------------------
# The 'self.properties' dict will contain OBJECTS to calculate properties and rewards.
# Each of those objects must implement at least 2 methods: 
#   1) 'predict': Gets an RDKit Mol object and returns a property value; and
#   2) 'reward' : Gets a property value and returns a reward. 

import importlib

class Estimators:

    def __init__(self, properties_cfg):
        self.properties = {}
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
        for _prop, _cls in self.properties.items():
            predictions = _cls.predict(mols)
            pred[_prop] = predictions
        return pred

    def estimate_rewards(self, n_mols, predictions):
        """Calculates the rewards, given a dict of pre-calculated properties.

        Args:
            n_mols (int): number of molecules
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
            rew[_prop] = cls.reward(_values)
        return rew

