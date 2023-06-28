from properties.Property import Property

class Prop1(Property):
    def predict(self,mols, **kwargs):
        _mols, prop1 = [], []
        _mols.extend(mols)

        for mol in mols:
            value = 10
            prop1.append(value)
        return prop1