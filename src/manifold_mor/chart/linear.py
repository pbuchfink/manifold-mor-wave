"""General linear chart and PCA (or POD).
This chart (Pca) is the go-to-solution in classical MOR.
No offline--online decomposition is yet performed for linear charts.
"""
import numpy as np
from manifold_mor.chart.basic import TrainableChart
from manifold_mor.fields.vector_reduced import (LinearVectorFieldProjector,
                                                VectorFieldProjector)


class LinearTrainableChart(TrainableChart):
    def __init__(self, is_trained=False):
        self._basis = np.array([])
        super().__init__(is_trained=is_trained)

    def map(self, xr):
        return xr @ self._basis.T

    def inv_map(self, x):
        return x @ self._basis

    def tangent_map(self, xr):
        return self._basis

    def is_valid_projector(self, projector: 'VectorFieldProjector'):
        return isinstance(projector, LinearVectorFieldProjector)

    def hessian(self, xr):
        return np.zeros(self.ambient_dim, self.dim, self.dim)

    def number_of_parameters_in_decoder(self):
        return self.dim * self.ambient_dim

    def number_of_parameters_in_total(self):
        return self.dim * self.ambient_dim


class PcaChart(LinearTrainableChart):
    def __init__(self, is_trained=False):
        super().__init__(is_trained)
        self.energy_contained = []

    def train(self, snapshots, training_params, logger=None, model=None):
        if training_params is None:
            training_params = dict()
        # perform PCA
        U, s = snapshots.get_svd_data()

        if 'red_dim' in training_params.keys():
            dim = training_params['red_dim']
        elif 'energy_contained' in training_params.keys():
            dim = np.where(np.cumsum(s) / np.sum(s) >= training_params['energy_contained'])[0][0]
        else:
            return ValueError('Training_params must include red_dim or energy_contained.')

        self._basis = U[:, :dim].copy()
        self.energy_contained = np.sum(s[:dim+1]) / np.sum(s)

        self.is_trained = True
        self.ambient_dim = U.shape[0]
        self.dim = dim

        return {'energy_contained': self.energy_contained}
