import numpy as np
# Chemistry
import rdkit
from rdkit import Chem

# Property is the abstract class from which all
# properties must inherit.
from properties.Property import Property

# Local 
from .SA_Score.sascore import SA_Scorer

class SAScore(Property):
    """
        Calculator class for SA_Score (synthetic acccessibility). 

        As implemented in RDKit SAS_scorer module, values are in the interval [1..10]:
             1 == GOOD (Easy to synthsize) 
            10 == BAD  (Impossible to synthesize)

        Here we use the SASCorer method developed by scientists at
        Novartis Institutes for BioMedical Research Inc.
        For details, see: http://www.doi.org/10.1186/1758-2946-1-8
    """

    CITATION = (f" \"RDKit: Open-source cheminformatics version {rdkit.__version__} "
                 "(www.rdkit.org)\"")


    def __init__(self, prop_name, **kwargs):
        # Initialize super
        super().__init__(prop_name, **kwargs)
        self.sascorer = SA_Scorer()


    def predict(self,
                mols,
                **kwargs):
        """
            Args:
                mols (rdkit.Chem.ROMol or list): molecule(s) to be evaluated

            Returns:
                list(float): Drug likeness scores
        """
        _mols = []
        _mols.extend(mols)
        sa_scores = self.sascorer.predict(_mols)
        return sa_scores

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

        sign = np.sign(self.thresh_step if self.optimize else self.threshold)
        unsigned_threshold = sign * self.threshold

        for value in _prop_values:
            # Use 0-1 as reward standard
            # SAScore
            # 1 == GOOD (Easy to synthsize)
            # 10 == BAD  (Impossible to synthesize)
            # add '-' and normalize SAScore
            rew = (- value) / 10
            rewards.append(rew)
        return rewards
    
