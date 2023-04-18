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
                mols,
                **kwargs):
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
            if query_mol is not None:
                try:
                    score = QED.qed(query_mol)
                except:
                    # RDKit gives exception when the molecules are weird. 
                    # Here we just ignore them and pass a score of -1.
                    pass
                qed_scores.append(score)
        return qed_scores
    
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
            if value >= threshold:
                rew = self.max_reward
            rewards.append(rew)
        return rewards


 