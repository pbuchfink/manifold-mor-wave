"""Special TimeStepper for covector fields."""
import numpy as np
from manifold_mor.fields.covector import CovectorField
from manifold_mor.fields.metric import ConstantMetric, Metric
from manifold_mor.time_steppers.residual import TimeStepperResidual


class CovectorFieldTimeStepperResidual(TimeStepperResidual):
    def __init__(self, model=None):
        self.covector_field = None
        self.metric = None
        super().__init__(model)
    
    def register_model(self, model):
        super().register_model(model)
        from manifold_mor.models.basic import Model
        assert isinstance(model, Model) \
            and hasattr(model, 'covector_field') \
            and hasattr(model, 'metric') \
            and isinstance(model.covector_field, CovectorField) \
            and isinstance(model.metric, Metric)
        self.covector_field = model.covector_field
        self.metric = model.metric
    
    def is_time_dependent(self):
        return self.covector_field.is_time_dependent()


class ImplicitEulerCovectorFieldTimeStepperResidual(CovectorFieldTimeStepperResidual):
    def evaluate(self, x, x_old, dt):
        return self.metric.apply(x, None, x - x_old) - dt * self.covector_field.eval(x)

    def evaluate_derivative(self, x, x_old, dt):
        jac = self.metric.eval(x) - dt * self.covector_field.compute_derivative(x)
        if not isinstance(self.metric, ConstantMetric):
            jac += np.einsum('ijk,j', self.metric.compute_derivative(x), x)
        return jac
