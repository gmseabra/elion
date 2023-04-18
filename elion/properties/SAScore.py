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
        Calculator class for SAScore (synthetic acccessibility). 

        As implemented in RDKit SAS_scorer module, values are in the interval [1..10]:
             1 == GOOD (Easy to synthsize) 
            10 == BAD  (Impossible to synthesize)

        Here we use the SASCorer method developed by scientists at
        Novartis Institutes for BioMedical Research Inc.
        For details, see: http://www.doi.org/10.1186/1758-2946-1-8
    """

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
        """Calculates the reward

        Args:
            prop_values (float/list): The values for the property

        Returns:
            list(float): The rewards
        """
        threshold = self.threshold

        _prop_values, rewards = [], []
        _prop_values.extend(prop_values)

        for value in _prop_values:
            rew = self.min_reward
            if value <= threshold:
                rew = self.max_reward
            rewards.append(rew)
        return rewards
    