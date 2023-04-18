from properties.Property import Property

class Prop2(Property):
    def predict(self,mols, **kwargs):
        _mols, prop2 = [], []
        _mols.extend(mols)

        for mol in _mols:
            value = 20
            prop2.append(value)
        return prop2

    def reward(self, prop_values, **kwargs):

        _prop_values, rewards = [], []
        _prop_values.extend(prop_values)

        for value in _prop_values:
            rew2 = 10
            rewards.append(rew2)
        return rewards