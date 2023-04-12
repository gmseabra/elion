from properties.Property import Property

class Prop3(Property):
    def value(self,mol="CCCCC", **kwargs):
        return 30

    def reward(self, prop_value, **kwargs):
        return 15