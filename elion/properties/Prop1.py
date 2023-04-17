from properties.Property import Property

class Prop1(Property):
    def predict(self,mol="CCCCC", **kwargs):
        return 10

    def reward(self, prop_value, **kwargs):
        return 5