import torch
from abc import ABC, abstractmethod

class Initializer(ABC):
    """
        A template class to form different types of initializers
    """
    @abstractmethod
    def weights(self):
        """Returns initial weights tensor
        """
        pass

class Organizer(ABC):
    """
        A template class to specify different types of learning rules as organizers
        Attributes (Learning hyperparameters): 
            - penalty: when a negative potential occurs, reducing its impact on the cumulative result.
        ---------------------------------------------------------------------------
            - margin: A float value, need additional information.
            - threshold: A float value, need additional information.
    """
    def __init__(self, **kwargs):
        self.penalty = kwargs.get('penalty', 1.0)
        self.margin = kwargs.get('margin', 1.0)
        self.threshold = kwargs.get('threshold', 0.2)
        
    @abstractmethod
    def step(self):
        """Stores and updates potentials after each input
        """
        pass
    
    @abstractmethod
    def organize(self):
        """Transforms potentials into connection weights
        """
        pass
    
    @staticmethod
    @abstractmethod
    def _potential():
        """Calculates potential based on inputs and outputs
        """
        pass