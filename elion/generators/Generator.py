# Abstract class for properties
from abc import ABC, abstractmethod

class Generator(ABC):
    """
    Abstract Class for molecular properties. All properties must
    inherit from this one.
    """

    def bomb_input(self, generator, msg):
        """For sending a message and quitting if there's an error in the input"""        
        message = f"\n*** ERROR when loading {generator}. ***\n{msg}"
        quit(message)

    # The methods below MUST be overridden by whatever generator implemented
    @abstractmethod
    def generate_mols(self, *args, **kwargs):
        """ Generate a number of molecules.
        """
        pass

    @abstractmethod
    def bias_generator(self, prop_values, **kwargs):
        """Bias the generator
        """
        pass