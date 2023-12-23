"""Special TimeStepper for vector fields."""
import numpy as np
from manifold_mor.fields.vector import VectorField
from manifold_mor.time_steppers.residual import TimeStepperResidual


class VectorFieldTimeStepperResidual(TimeStepperResidual):
    def __init__(self, model=None):
        self.vector_field = None
        super().__init__(model)

    def register_model(self, model):
        super().register_model(model)
        from manifold_mor.models.basic import Model
        assert isinstance(model, Model) \
            and hasattr(model, 'vector_field') \
            and isinstance(model.vector_field, VectorField)
        self.vector_field = model.vector_field
    
    def is_time_dependent(self):
        return self.vector_field.is_time_dependent()


class ImplicitEulerVectorFieldTimeStepperResidual(VectorFieldTimeStepperResidual):
    def evaluate(self, x, x_old, dt):
        return x - x_old - dt * self.vector_field.eval(x)

    def evaluate_derivative(self, x, x_old, dt):
        n = len(x)
        return np.eye(n) - dt * self.vector_field.compute_derivative(x)


class ExplicitEulerVectorFieldTimeStepperResidual(VectorFieldTimeStepperResidual):
    def evaluate(self, x, x_old, dt):
        return x - x_old - dt * self.vector_field.eval(x_old)

    def evaluate_derivative(self, x, x_old, dt):
        n = len(x)
        return np.eye(n)

class ImplicitMidpointVectorFieldTimeStepperResidual(VectorFieldTimeStepperResidual):
    def evaluate(self, x, x_old, dt):
        return x - x_old - dt * self.vector_field.eval((x + x_old) / 2)
    
    def evaluate_derivative(self, x, x_old, dt):
        n = len(x)
        return np.eye(n) - dt / 2 * self.vector_field.compute_derivative((x + x_old) / 2)
