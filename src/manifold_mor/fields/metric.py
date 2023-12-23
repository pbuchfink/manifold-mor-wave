'''General definition of a metric tensor field and its reduced (pulled back) form.'''
from typing import TYPE_CHECKING, Optional

import numpy as np
from manifold_mor.fields.tensor import TensorField

if TYPE_CHECKING:
    from manifold_mor.trial_manifold.basic import TrialManifold


class Metric(TensorField):
    def __init__(self, name: str = 'metric', mu: Optional[dict] = None):
        super().__init__(name, mu=mu)
    
    def apply(self, x, v, w):
        '''Apply metric to the vectors v and w, both element of the tangent space at φ(x).'''
        if v is None:
            return self.eval(x) @ w
        elif not w is None:
            return v @ self.eval(x) @ w
        else:
            return v @ self.eval(x)

    def apply_inverse(self, x, v, w):
        '''Apply inverse metric to the vectors v and w, both element of the tangent space at φ(x).'''
        if v is None:
            return np.linalg.solve(self.eval(x), w)
        elif not w is None:
            return v @ np.linalg.solve(self.eval(x), w)
        else:
            return np.linalg.solve(self.eval(x), v).T


class ConstantMetric(Metric):
    def __init__(self, name='const_metric', mu=None):
        super().__init__(name=name, mu=mu)
        super().__setattr__('const_metric_tensor', None)

    def __setattr__(self, name, value):
        if name == 'const_metric_tensor':
            assert isinstance(value, np.ndarray)
            shape = value.shape
            assert len(shape) == 2 and shape[0] == shape[1]

        super().__setattr__(name, value)

    def eval(self, x):
        return self.const_metric_tensor

    def compute_derivative(self, x):
        n = self.const_metric_tensor.shape[0]
        return np.zeros((n, n, n))


class ReducedMetric(Metric):
    def __init__(self, metric: Metric, manifold: 'TrialManifold'):
        assert isinstance(metric, Metric)
        self.metric = metric
        self.manifold = manifold
        super().__init__(name='reduced_' + metric.name, mu=metric.mu)

    def eval(self, x):
        tan_map = self.manifold.tangent_map(x)
        return tan_map.T @ self.metric.eval(self.manifold.map(x)) @ tan_map

    def compute_derivative(self, x):
        if isinstance(self.metric, ConstantMetric):
            n = self.manifold.dim
            return np.zeros((n, n, n))
        else:
            derv_metric = self.metric.compute_derivative(self.manifold.map(x))
            tan_map = self.manifold.tangent_map(x)
            return np.einsum('ikm,ij,kl,mn', derv_metric, tan_map, tan_map, tan_map)

    def set_mu(self, mu):
        self.metric.set_mu(mu)
