'''Different possibilities for reduced (i.e. pulled-back) vector fields.'''
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from manifold_mor.fields.vector import VectorField
from scipy.linalg import lstsq

if TYPE_CHECKING:
    from manifold_mor.trial_manifold.basic import TrialManifold


class VectorFieldProjector(ABC):
    def __init__(self, manifold: "TrialManifold") -> None:
        self.manifold = manifold

    @abstractmethod
    def project_to_reduced_manifold(self, x: np.ndarray, v: np.ndarray):
        ...


class ReducedVectorField(VectorField):
    def __init__(self, vector_field: VectorField, manifold: "TrialManifold", projector: VectorFieldProjector):
        assert isinstance(vector_field, VectorField)
        self.vector_field = vector_field
        self.manifold = manifold
        self.projector = projector
        super().__init__('reduced_' + vector_field.name, mu=vector_field.mu)

    def set_mu(self, mu):
        super().set_mu(mu)
        self.vector_field.set_mu(mu)

    def reconstruct(self, x, reduced_vector=None):
        if reduced_vector is None:
            reduced_vector = self.eval(x)
        return self.manifold.tangent_map(x) @ reduced_vector

    def eval(self, x):
        return self.projector.project_to_reduced_manifold(
            x, self.vector_field.eval(self.manifold.map(x))
        )
    
    def compute_derivative(self, x):
        # neglects derivatives of the tangent map (Quasi-Newton)
        tan_map = self.manifold.tangent_map(x)
        x_full = self.manifold.map(x)
        try:
            vector_field_derivative_tan_map = self.vector_field.compute_derivative(x_full) @ tan_map
        except NotImplementedError:
            vector_field_derivative_tan_map = np.zeros((self.manifold.ambient_dim, self.manifold.dim))
            for i_tan_vec, tan_vec in enumerate(tan_map.T):
                vector_field_derivative_tan_map[:, i_tan_vec] = self.vector_field.apply_derivative_y(x_full, tan_vec)
        return self.projector.project_to_reduced_manifold(x, vector_field_derivative_tan_map)
    
    def project_tangent(self, x_array: np.ndarray, v_array: np.ndarray):
        if x_array.ndim == 1:
            assert v_array.ndim == 1
            x_array = x_array[np.newaxis, ...]
            v_array = v_array[np.newaxis, ...]
            _unpack = True
        else:
            _unpack = False

        proj_v = np.empty_like(v_array)
        for x, (i_t, v) in zip(x_array, enumerate(v_array)):
            xr = self.manifold.inv_map(x)
            proj_v[i_t] = self.reconstruct(
                xr, self.projector.project_to_reduced_manifold(xr, v)
            )
        
        if _unpack:
            assert proj_v.shape[0] == 1
            proj_v = proj_v[0]
        return proj_v


class MoorePenroseVectorFieldProjector(VectorFieldProjector):
    def project_to_reduced_manifold(self, x: np.ndarray, v: np.ndarray):
        assert x.ndim == 1
        tan_map = self.manifold.tangent_map(x)
        v_red, _, _, _ = lstsq(tan_map, v)
        return v_red


class WeightedMoorePenroseVectorFieldProjector(VectorFieldProjector):
    def __init__(self, vector_field, manifold):
        assert hasattr(vector_field, 'apply_weighting_matrix') and callable(vector_field.apply_weighting_matrix)
        self._cached_tan_map_weighted_product = (None,) * 2
        super().__init__(vector_field, manifold)

    def _save_tan_map_weighted_product(self, xr, tan_map_weighted_product):
        self._cached_tan_map_weighted_product = (xr, tan_map_weighted_product)

    def _load_tan_map_weighted_product(self, xr):
        if np.all(self._cached_tan_map_weighted_product[0] == xr):
            return self._cached_tan_map_weighted_product[1]
        else:
            return None

    def _get_tan_map_weighted_product(self, xr):
        tan_map_weighted_product = self._load_tan_map_weighted_product(xr)
        if tan_map_weighted_product is None:
            tan_map = self.manifold.tangent_map(xr)
            tan_map_weighted_product = np.zeros((tan_map.shape[1],)*2)
            for i_tan, tan in enumerate(tan_map.T):
                tan_map_weighted_product[:, i_tan] = tan_map.T @ self.vector_field.apply_weighting_matrix(tan)
            self._save_tan_map_weighted_product(xr, tan_map_weighted_product)
        return tan_map_weighted_product

    def project_to_reduced_manifold(self, x: np.ndarray, v: np.ndarray):
        # multiply with weighting matrix
        if v.ndim == 2:
            weighted_v = np.zeros_like(v)
            for i_v, v_vec in enumerate(v):
                weighted_v[:, i_v] = self.vector_field.apply_weighting_matrix(v_vec)
        elif v.dim == 2:
            weighted_v = self.vector_field.apply_weighting_matrix(v)

        # multiply with inverse weighted product
        tan_map = self.manifold.tangent_map(x)
        tan_map_weighted_prod = self._get_tan_map_weighted_product(x)
        return np.linalg.solve(tan_map_weighted_prod, tan_map.T @ weighted_v)


class EncoderVectorFieldProjector(VectorFieldProjector):
    def project_to_reduced_manifold(self, x: np.ndarray, v: np.ndarray):
        assert x.ndim == 1
        x_full = self.manifold.map(x)
        if v.ndim == 2 and x.ndim == 1 and v.shape[0] == x_full.shape[0]:
            #different conventions of sorting vectors
            return self.manifold.encoder_jvp(x_full, v.T).T
        else:
            return self.manifold.encoder_jvp(x_full, v)


class LinearVectorFieldProjector(VectorFieldProjector):
    def __init__(self, manifold: "TrialManifold") -> None:
        super().__init__(manifold)
        self.tan_map = self.manifold.tangent_map(np.zeros(self.manifold.dim))


class LinearOrthogonalVectorFieldProjector(LinearVectorFieldProjector):
    def __init__(self, manifold: "TrialManifold") -> None:
        super().__init__(manifold)
        assert np.allclose(self.tan_map.T @ self.tan_map, np.eye(self.manifold.dim))

    def project_to_reduced_manifold(self, x: np.ndarray, v: np.ndarray):
        return self.tan_map.T @ v


class LinearSymplecticVectorFieldProjector(LinearVectorFieldProjector):
    def __init__(self, manifold: "TrialManifold") -> None:
        super().__init__(manifold)
        assert np.allclose(
            -self.apply_canonical_J(self.tan_map.T @ self.apply_canonical_J(self.tan_map)),
            np.eye(self.manifold.dim)
        )

    def apply_canonical_J(self, arr: np.ndarray):
        two_n = arr.shape[0]
        assert two_n % 2 == 0
        return np.vstack([arr[two_n//2:, ...], -arr[:two_n//2, ...]])

    def project_to_reduced_manifold(self, x: np.ndarray, v: np.ndarray):
        return -self.apply_canonical_J(self.tan_map.T @ self.apply_canonical_J(v))
