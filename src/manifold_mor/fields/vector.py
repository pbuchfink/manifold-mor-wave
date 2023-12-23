'''General definition of a vector field.'''
from typing import Optional

from manifold_mor.fields.covector import CovectorField
from manifold_mor.fields.metric import ConstantMetric, Metric
from manifold_mor.fields.tensor import TensorField


class VectorField(TensorField):
    def __init__(self, name: str = 'vector_field', mu: Optional[dict] = None):
        super().__init__(name, mu=mu)


class MassInvertedVectorField(VectorField):
    '''This is a neat work arround to get a VectorField from a CovectorField by inverting the mass matrix.
    This class mainly exists to showcase that it is better to use the reduction directly for the
    CovectoeField.'''
    def __init__(self, covector: CovectorField, metric: Metric):
        assert isinstance(covector, CovectorField) and isinstance(metric, Metric)
        c_mu = covector.mu or dict()
        m_mu = metric.mu or dict()
        mu = {**c_mu, **m_mu}
        self.covector = covector
        self.metric = metric
        super().__init__(name='mass_inv_' + covector.name, mu=mu)

    def eval(self, x):
        return self.metric.apply_inverse(x, None, self.covector.eval(x))

    def compute_derivative(self, x):
        if not isinstance(self.metric, ConstantMetric):
            raise NotImplementedError()

        return self.metric.apply_inverse(x, None, self.covector.compute_derivative(x))

    def set_mu(self, mu):
        super().set_mu(mu)
        self.covector.set_mu(mu)
        self.metric.set_mu(mu)
