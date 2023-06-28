from properties.Property import Property

class Prop2(Property):
    def predict(self,mols, **kwargs):
        _mols, prop2 = [], []
        _mols.extend(mols)

        for mol in _mols:
            value = 20
            prop2.append(value)
        return prop2
