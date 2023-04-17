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
              query_mol:rdkit.Chem.Mol,
              **kwargs) -> float:
        """
            Args:
                mol (rdkit.Chem.ROMol): molecule to be evaluated

            Returns:
                float: Drug likeness score
        """
        sa_score = 10.0
        if query_mol is not None:
            try:
                sa_score = self.sascorer.predict([query_mol])[0] 
            except:
                # In some cases,RDKit throws a weird exception and crashes.
                # Here, we just catch that and continue, leaving the value as the
                # maximum.
                pass
        return sa_score
    
    def reward(self, prop_value, **kwargs):
        """Calculates the reward

        Args:
            prop_value (float): The value for the property

        Returns:
            float: The reward
        """
        
        threshold = self.threshold
        reward = self.min_reward
        if prop_value <= threshold:
            reward = self.max_reward
    
        return reward



 