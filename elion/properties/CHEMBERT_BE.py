# Chemistry
import rdkit
from rdkit import Chem

# Property is the abstract class from which all
# properties must inherit.
from properties.Property import Property

# Local
from .CHEMBERT.chembert import chembert_model, SMILES_Dataset

class CHEMBERT_BE(Property):
    """
        Calculator class for CHEM-BERT binding energies 
        Estimates binding energies given a CHEM-BERT model and a SMILES file.

        Note: This calculates the property of only ONE molecule, which is definately *not*
              the most efficient use of the CHEM-BERT implementation, but is one that
              fits our model. Later, we may come back to a more efficient implementation.
    """

    def __init__(self, prop_name, **kwargs):
        # Initialize super
        super().__init__(prop_name, **kwargs)
        self.model_file = kwargs['model_file']

        print("Initializing CHEMBERT model... ", end="")
        self.model = chembert_model(self.model_file)
        print("Done.")

    def predict(self, 
                query_mol:rdkit.Chem.Mol,
                **kwargs) -> float:
        """
            Args:
                mol (rdkit.Chem.ROMol): molecule to be evaluated

            Returns:
                float: Predicted binding energy from the model
        """
        score = 10.0
        if query_mol is not None:
            dataset = SMILES_Dataset([Chem.MolToSmiles(query_mol)])
            score = float(self.model.predict(dataset)[0])
        return score
    
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



 