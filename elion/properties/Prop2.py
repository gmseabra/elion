from properties.Property import Property

class Prop2(Property):
    def value(self,mol="CCCCC", **kwargs):
        return 20

    def reward(self, prop_value, **kwargs):
        return 10