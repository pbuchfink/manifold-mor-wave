'''General definition of a tensor field.'''
from abc import ABC, abstractmethod
from typing import Optional


class TensorField(ABC):
    def __init__(self, name: str, mu: Optional[dict] = None):
        self.name = name
        self.mu = None
        if mu:
            self.set_mu(mu)

    @abstractmethod
    def eval(self, x):
        '''Return tensor represented in coordinates.'''
        ...

    @abstractmethod
    def compute_derivative(self, x):
        '''Compute Jacobian of the tensor in coordinates.'''
        ...
    
    def is_time_dependent(self):
        '''Returns if vector field is time-dependent.'''
        return False

    def apply_derivative_y(self, x, y):
        '''Apply Jacobian of the tensor in coordinates to an element y in coordinates.'''
        return self.compute_derivative(x) @ y

    def set_mu(self, mu):
        self.mu = mu
