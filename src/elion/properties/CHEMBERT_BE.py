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
                mols,
                **kwargs):
        """
            Args:
                mol (rdkit.Chem.ROMol or list): molecule to be evaluated

            Returns:
                float: Predicted binding energy from the model
        """

        _mols, chembert_scores = [], []
        _mols.extend(mols)

        smis = []
        for mol in _mols:
            smis.append(Chem.MolToSmiles(mol))
                   
        dataset = SMILES_Dataset(smis)
        chembert_scores = self.model.predict(dataset)

        return chembert_scores
