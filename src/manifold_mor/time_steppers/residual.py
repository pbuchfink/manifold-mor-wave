"""Basic class for a residual in the time stepper."""
from abc import ABC, abstractmethod


class TimeStepperResidual(ABC):
    def __init__(self, model=None):
        self.model = None
        if model is not None:
            self.register_model(model)

    def register_model(self, model):
        self.model = model

    def is_valid(self):
        return self.model is not None

    @abstractmethod
    def evaluate(self, x, x_old, dt):
        ...

    @abstractmethod
    def evaluate_derivative(self, x, x_old, dt):
        ...
    
    @abstractmethod
    def is_time_dependent(self):
        ...

    def apply_derivative_y(self, x, x_old, dt, y):
        self.evaluate_derivative(x, x_old, dt) @ y

    def initial_value(self, mu):
        return self.model.initial_value(mu)

    def set_mu(self, mu):
        self.model.set_mu(mu)

    def copy(self):
        return self.__class__()
