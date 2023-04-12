# Predicts the VinaScore for a given Molecule
from properties.Property import Property

class VinaScore(Property):
    """Estimation of Vina Scores
    """

    def value(self):
        return 30

    def reward(self):
        return 15