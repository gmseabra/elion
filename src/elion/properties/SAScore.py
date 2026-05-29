import logging
import numpy as np
import rdkit
from rdkit import Chem

from properties.Property import Property
from .SA_Score.sascore import SA_Scorer


class SAScore(Property):
    """
    Calculator class for SA_Score (synthetic accessibility).

    As implemented in RDKit SAS_scorer module, values are in the interval [1..10]:
         1 == GOOD (Easy to synthesize)
        10 == BAD  (Impossible to synthesize)
    """

    CITATION = (f" \"RDKit: Open-source cheminformatics version {rdkit.__version__} "
                "(www.rdkit.org)\"")

    def __init__(self, prop_name, logger: logging.Logger | None = None, **kwargs):
        super().__init__(prop_name, **kwargs)
        self._logger = logger or logging.getLogger(__name__)
        self.sascorer = SA_Scorer()

    def predict(self, mols, **kwargs):
        """
        Args:
            mols (rdkit.Chem.ROMol or list): molecule(s) to be evaluated
        Returns:
            list(float): SA scores
        """
        _mols = []
        _mols.extend(mols)
        return self.sascorer.predict(_mols)

    def reward(self, prop_values, **kwargs):
        """Given a property value, or list of values,
           returns this property rewards list(float).

        Args:
            prop_value (float or list(floats)): The calculated value(s) of the property
        Returns:
            list(float): This property rewards for each value passed in.
        """
        _prop_values, rewards = [], []
        _prop_values.extend(prop_values)

        for value in _prop_values:
            # SAScore: 1 == GOOD, 10 == BAD — negate and normalize
            rew = (-value) / 10
            rewards.append(rew)
        return rewards