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
        return prop2

    def reward(self, prop_values, **kwargs):

        _prop_values, rewards = [], []
        _prop_values.extend(prop_values)

        for value in _prop_values:
            rew = 15
            rewards.append(rew)
        return rewards