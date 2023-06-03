# Predicts the VinaScore for a given Molecule
from properties.Property import Property

class VinaScore(Property):
    """Estimation of Vina Scores
    """

    def predict(self,mols, **kwargs):
        _mols, vina_scores = [], []
        _mols.extend(mols)

        for mol in _mols:
            value = 30
            vina_scores.append(value)
        return vina_scores
