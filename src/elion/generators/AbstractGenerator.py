# Abstract class for properties
from abc import ABC, abstractmethod

class AbstractGenerator(ABC):
    """Abstract class for generators

    """
    
    def bomb_input(self, generator, msg):
        """For sending a message and quitting if there's an error in the input"""        
        message = f"\n*** ERROR when loading {generator}. ***\n{msg}"
        quit(message)
        
    @abstractmethod
    def generate_mols(self):
        """
        Generates a sample of molecules using the provided generator.
        """
        pass

    @abstractmethod
    def bias_generator(self, prop_values, **kwargs):
        """Biases the generator
        """
        pass 
