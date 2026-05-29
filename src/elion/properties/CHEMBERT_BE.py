import logging
import numpy as np
from rdkit import Chem

from properties.Property import Property
from .CHEMBERT.chembert import chembert_model, SMILES_Dataset


class CHEMBERT_BE(Property):
    """
    Calculator class for CHEM-BERT binding energies.
    Estimates binding energies given a CHEM-BERT model and a SMILES file.
    """

    CITATION = ("  Kim, H., Lee, J., Ahn, S., & Lee, J. R. (2021).\n"
                "  \"A merged molecular representation learning for molecular \n"
                "  properties prediction with a web-based service.\" \n"
                "  Scientific Reports, 11(1), 11028.\n"
                "  https://doi.org/10.1038/s41598-021-90259-7")

    def __init__(self, prop_name, logger: logging.Logger | None = None, **kwargs):
        super().__init__(prop_name, **kwargs)
        self.model_file = kwargs['model_file']
        self._logger = logger or logging.getLogger(__name__)

        self._logger.info("Initializing CHEMBERT model...")
        self.model = chembert_model(self.model_file)
        self._logger.info("CHEMBERT model ready.")

    def predict(self, mols, **kwargs):
        """
        Args:
            mols (rdkit.Chem.ROMol or list): molecule(s) to be evaluated
        Returns:
            list(float): Predicted binding energies from the model
        """
        import warnings
        _mols = list(mols)
        smis = [Chem.MolToSmiles(mol) for mol in _mols]
        dataset = SMILES_Dataset(smis)
        # Suppress DataLoader worker count warning — we accept the default worker config
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*DataLoader will create.*worker.*')
            return self.model.predict(dataset)

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
            # CHEMBERT_BE learned from vina: the less the better
            rew = -value
            rewards.append(rew)
        return rewards