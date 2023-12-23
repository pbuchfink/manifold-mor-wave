"""Symplectic linear charts.
These charts are used for (classical) Symplectic MOR.
No offline--online decomposition is yet performed for linear charts.
"""
import numpy as np
from manifold_mor.chart.linear import LinearTrainableChart


class CanonicallySymplecticLinearTrainableChart(LinearTrainableChart):
    def inv_map(self, x):
        # symplectic inverse (with canonical J2N and J2n) of self._basis times x
        N = x.shape[-1] // 2
        assert x.shape[-1] == 2*N
        res = np.hstack([x[..., N:], -x[..., :N]]) @ self._basis
        n = res.shape[-1] // 2
        return np.hstack([-res[..., n:], res[..., :n]])

class PsdCotangentLiftChart(CanonicallySymplecticLinearTrainableChart):
    def __init__(self, is_trained=False):
        super().__init__(is_trained)
        self.energy_contained = []

    def train(self, snapshots, training_params, logger=None, model=None):
        if training_params is None:
            training_params = dict()

        U, s = snapshots.get_psd_cotangent_lift_data()

        if 'red_dim' in training_params.keys():
            assert training_params['red_dim'] % 2 == 0, 'red_dim has to be even'
            dim = training_params['red_dim'] // 2
        elif 'energy_contained' in training_params.keys():
            dim = np.where(np.cumsum(s) / np.sum(s) >= training_params['energy_contained'])[0][0]
        else:
            return ValueError('Training_params must include red_dim or energy_contained.')

        U_dim = U[:, :dim].copy()

        self._basis = np.vstack([
            np.hstack([U_dim, np.zeros_like(U_dim)]),
            np.hstack([np.zeros_like(U_dim), U_dim]),
        ])        
        self.energy_contained = np.sum(s[:dim+1]) / np.sum(s)

        self.is_trained = True
        self.ambient_dim = self._basis.shape[0]
        self.dim = self._basis.shape[1]

        return {'energy_contained': self.energy_contained}


class PsdComplexSvdChart(CanonicallySymplecticLinearTrainableChart):
    def __init__(self, is_trained=False):
        super().__init__(is_trained)
        self.energy_contained = []

    def train(self, snapshots, training_params, logger=None, model=None):
        if training_params is None:
            training_params = dict()

        # combined snapshot matrix

        U, s = snapshots.get_psd_complex_svd_data()

        if 'red_dim' in training_params.keys():
            assert training_params['red_dim'] % 2 == 0, 'red_dim has to be even'
            dim = training_params['red_dim'] // 2
        elif 'energy_contained' in training_params.keys():
            dim = np.where(np.cumsum(s) / np.sum(s) >= training_params['energy_contained'])[0][0]
        else:
            return ValueError('Training_params must include red_dim or energy_contained.')

        U_dim = U[:, :dim].copy()

        self._basis = np.vstack([
            np.hstack([U_dim.real, U_dim.imag]),
            np.hstack([-U_dim.imag, U_dim.real]),
        ])        
        self.energy_contained = np.sum(s[:dim+1]) / np.sum(s)

        self.is_trained = True
        self.ambient_dim = self._basis.shape[0]
        self.dim = self._basis.shape[1]

        return {'energy_contained': self.energy_contained}
