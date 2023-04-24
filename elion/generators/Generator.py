# Abstract class for properties
from abc import ABC, abstractmethod
import importlib

class Generator(ABC):
    """
    Abstract Class for molecular properties. All properties must
    inherit from this one.
    """

    def __init__(self,
                 generator_properties):

        self.name = generator_properties['name']
        self.batch_size = generator_properties['batch_size']

        module = importlib.import_module(f'generators.{self.name}')
        module = getattr(module, self.name)
        self.generator = module(generator_properties).generator

    def bomb_input(self, generator, msg):
        """For sending a message and quitting if there's an error in the input"""        
        message = f"\n*** ERROR when loading {generator}. ***\n{msg}"
        quit(message)
        
    def generate_mols(self):
        """
        Generates a sample of n_to_generate molecules using the provided generator.

        Args:
            generator (generator): Generator object to use. Must have a 'generate' function that
                                gets as argument a number and returns that number of molecules. 
            n_to_generate (int): Number of molecules to generate
        
        Returns:
            List of <n_to_generate> smiles strings.
        """
        generated = self.generator.generate(self.batch_size, verbose=1)
        return generated

    # # The methods below MUST be overridden by whatever generator implemented
    # @abstractmethod
    # def bias_generator(self, prop_values, **kwargs):
    #     """Bias the generator
    #     """
    #     pass