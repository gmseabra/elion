# Abstract class for properties
import importlib

class Generator:
    """
    Molecular generator class. All this does is to instantiate a 
    generic generator passed in.

    The "generator" attribute in the returned object is whatever
    generator was chosen.
    """

    def __init__(self,
                 generator_properties):

        self.name = generator_properties['name']
        module = importlib.import_module(f'generators.{self.name}')
        module = getattr(module, self.name)
        self.generator = module(generator_properties)
        
