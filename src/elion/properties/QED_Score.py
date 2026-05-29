import logging
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED

from properties.Property import Property


class QED_Score(Property):
    """
    Calculator class for QED score (drug likeness).

    As implemented in RDKit QED module, values are in the interval [0, 1]:
        0 == BAD  (all properties unfavourable)
        1 == GOOD (all properties favourable)
    """

    CITATION = (f" \"RDKit: Open-source cheminformatics version {rdkit.__version__} "
                "(www.rdkit.org)\"")

    def __init__(self, prop_name, logger: logging.Logger | None = None, **kwargs):
        super().__init__(prop_name, **kwargs)
        self._logger = logger or logging.getLogger(__name__)

    def predict(self, mols, **kwargs):
        """
        Args:
            mols: RDKit Mol or list of RDKit Mols
        Returns:
            list(float): Drug likeness scores
        """
        _mols, qed_scores = [], []
        _mols.extend(mols)

        for query_mol in _mols:
            score = -1.0
            try:
                score = QED.qed(query_mol)
            except Exception:
                # RDKit gives exception when the molecules are weird.
                # Here we just ignore them and pass a score of -1.
                pass
            qed_scores.append(score)
        return qed_scores

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
            # QED: 0 == BAD, 1 == GOOD — use value directly
            rewards.append(value)
        return rewards