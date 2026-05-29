# Estimator class
# -------------------------------
# Molecular Properties Prediction
# -------------------------------
# The 'self.properties' dict will contain OBJECTS to calculate properties and rewards.
# Each of those objects must implement at least 2 methods: 
#   1) 'predict': Gets an RDKit Mol object and returns a property value; and
#   2) 'reward' : Gets a property value and returns a reward. 

import importlib
import logging
import rdkit.Chem as Chem


class Estimators:

    def __init__(self, properties_cfg, logger: logging.Logger | None = None):
        self.properties = {}
        self.n_mols = 0
        self.all_converged = False
        self._logger = logger or logging.getLogger(__name__)

        for prop in properties_cfg:
            self._logger.info("-" * 80)
            self._logger.info(f"Loading Property: {prop.upper()}")
            module = importlib.import_module(f'properties.{prop}')
            module = getattr(module, prop)
            self.properties[prop] = module(prop, **properties_cfg[prop],
                                           logger=self._logger)
            self._logger.info(f"Done Loading Property: {prop.upper()}")

        # Maximum possible reward per molecule
        max_reward = 0.0
        for _prop, _cls in self.properties.items():
            max_reward += _cls.rew_coeff * _cls.max_reward
        self.max_reward = max_reward
        self._logger.info("-" * 80)
        self._logger.info("Done reading properties.")
        self._logger.info(f"The maximum possible reward per molecule is: {self.max_reward:6.2f}")
        self._logger.info("Note: Maximum rewards only consider properties being optimized.")
        self._logger.info("=" * 80)

    def estimate_properties(self, mols):
        """Calculates the properties for a list of molecules

        Args:
            mols ([RDKit ROMol]): List of RDKit ROMol objects
        Returns:
            Dict: Dictionary of properties as keys and predictions (floats) as values
        """
        pred = {_prop: [] for _prop in self.properties}

        _mols = list(mols)
        # Store n_mols so estimate_rewards can validate; also returned in pred
        # for thread-safety when called from evaluate_batch with varying batch sizes
        self.n_mols = len(_mols)
        pred['__n_mols__'] = self.n_mols

        for _prop, _cls in self.properties.items():
            predictions = _cls.predict(_mols)
            self._logger.debug('_cls: %s | n=%d predictions', _cls, len(predictions))
            pred[_prop] = predictions
        return pred

    def estimate_rewards(self, predictions):
        """Calculates the rewards, given a dict of pre-calculated properties.

        Args:
            predictions (dict): Dictionary with properties as keys and lists of
                                predicted values as values.
        Returns:
            dict: Dict with property names as keys and rewards as values.
        """
        # Read n_mols from the predictions dict so batch calls with different
        # sizes don't race on self.n_mols instance state
        n_mols = predictions.get('__n_mols__', self.n_mols)
        rew = {}
        for _prop, cls in self.properties.items():
            if cls.optimize:
                _values = predictions[_prop]

                if len(_values) != n_mols:
                    msg = (f"ERROR: Something went wrong...\n"
                           f"Expecting {n_mols} values, but got only {len(_values)} "
                           f"for property {_prop}.")
                    quit(msg)

                rew[_prop] = cls.reward(_values)
                self._logger.debug('cls: %s | rew: %s', cls, rew[_prop])
            else:
                rew[_prop] = [0.0] * n_mols

        rew["TOTAL"] = self.total_reward(rew, n_mols)
        return rew

    def total_reward(self, rewards, n_mols: int | None = None):
        if n_mols is None:
            n_mols = self.n_mols
        total_rew = []
        for mol in range(n_mols):
            total_rew_mol = 0.0
            for _prop, cls in self.properties.items():
                if cls.optimize:
                    this_rew = rewards[_prop][mol] * cls.rew_coeff
                    self._logger.debug('cls: %s | rew_coeff: %s | rewards[_prop][mol]: %s | _prop: %s',
                                       cls, cls.rew_coeff, rewards[_prop][mol], _prop)
                    total_rew_mol += this_rew
            total_rew.append(total_rew_mol)
        return total_rew

    def check_and_adjust_thresholds(self, predictions):
        """Checks if the predictions are within the thresholds and adjusts them"""
        self.all_converged = True
        for _prop, cls in self.properties.items():
            if cls.optimize:
                _values = predictions[_prop]
                cls.check_and_adjust_property_threshold(_values)
                if not cls.converged:
                    self.all_converged = False
        return

    def smiles_reward_pipeline(self, smis, kwargs):
        """
        Sometimes the RL process needs to pass the molecules as SMILES and needs
        to get the reward. This function does that.

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