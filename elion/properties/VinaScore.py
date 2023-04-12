# Predicts the VinaScore for a given Molecule
from properties.Property import Property

class VinaScore(Property):
    """Estimation of Vina Scores
    """

    def value(self,mol="CCCCC", **kwargs):
        return 30

    def reward(self, prop_value, **kwargs):
        return 15