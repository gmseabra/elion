from properties.Property import Property

class Prop1(Property):
    def predict(self,mols, **kwargs):
        _mols, prop1 = [], []
        _mols.extend(mols)

        for mol in mols:
            value = 10
            prop1.append(value)
        return prop1

    def reward(self, prop_values, **kwargs):

        _prop_values, rewards = [], []
        _prop_values.extend(prop_values)

        for value in _prop_values:
            rew1 = 5
            rewards.append(rew1)
        return rewards