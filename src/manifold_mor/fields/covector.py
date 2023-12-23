'''General definition of a covector field and its reduced (pulled back) form.'''
from typing import TYPE_CHECKING, Optional

from manifold_mor.fields.tensor import TensorField

if TYPE_CHECKING:
    from manifold_mor.trial_manifold.basic import TrialManifold


class CovectorField(TensorField):
    def __init__(self, name: str = 'covector_field', mu: Optional[dict] = None):
        super().__init__(name, mu=mu)

    def apply(self, x, v):
        return self.eval(x) @ v


class ReducedCovectorField(CovectorField):
    def __init__(self, covector_field: CovectorField, manifold: 'TrialManifold'):
        assert isinstance(covector_field, CovectorField)
        self.covector_field = covector_field
        self.manifold = manifold
        super().__init__(name='reduced_' + covector_field.name, mu=covector_field.mu)

    def eval(self, x):
        return self.manifold.tangent_map(x).T @ self.covector_field.eval(self.manifold.map(x))

    def compute_derivative(self, x):
        # neglects derivatives of the tangent map (Quasi-Newton)
        tan_map = self.manifold.tangent_map(x)
        return tan_map.T @ self.covector_field.compute_derivative(self.manifold.map(x)) @ tan_map

    def set_mu(self, mu):
        self.covector_field.set_mu(mu)
