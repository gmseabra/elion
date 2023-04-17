# Chemistry
import rdkit
from rdkit import Chem
from rdkit.Chem import QED

# Property is the abstract class from which all
# properties must inherit.
from properties.Property import Property

class QEDScore(Property):
    """
        Calculator class for QED score (drug likeness). 

        Uses the method described in:
        _'Quantifying the chemical beauty of drugs'_
        Nature Chemistry volume 4, pages90–98(2012)
        https://doi.org/10.1038/nchem.1243
        
        As implemented in RDKit QED module, QED values are in the interval [0,1]:
            0 == BAD  (all properties unfavourable) 
            1 == GOOD (all properties favourable)
    """

    def predict(self, 
              query_mol:rdkit.Chem.Mol,
              **kwargs) -> float:
        """
            Args:
                mol (rdkit.Chem.ROMol): molecule to be evaluated

            Returns:
                float: Drug likeness score
        """
        qed_score = -1.0
        if query_mol is not None:
            try:
                qed_score = QED.qed(query_mol)
            except:
                # RDKit gives exception when the molecules are weird. 
                # Here we just ignore them and pass a score of -1.
                pass
        return qed_score
    
    def reward(self, prop_value, **kwargs):
        """Calculates the reward

        Args:
            prop_value (float): The value for the property

        Returns:
            float: The reward
        """
        
        threshold = self.threshold
        reward = self.min_reward
        if prop_value >= threshold:
            reward = self.max_reward
    
        return reward


 