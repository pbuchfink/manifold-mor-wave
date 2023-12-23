"""Identity chart.
This is used for DOFs that are not reduced in combination with a ComponentwiseManifold.
Such DOFs are e.g. boundary DOFs.
"""
import numpy as np
from manifold_mor.chart.basic import Chart
from manifold_mor.fields.vector_reduced import LinearOrthogonalVectorFieldProjector, VectorFieldProjector
    

class IdentityChart(Chart):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ambient_dim = dim

    def map(self, xr):
        return xr 

    def inv_map(self, x):
        return x

    def tangent_map(self, xr):
        return np.eye(self.dim)

    def encoder_tangent_map(self, x):
        return np.eye(self.dim)

    def is_valid_projector(self, projector: 'VectorFieldProjector'):
        return isinstance(projector, LinearOrthogonalVectorFieldProjector)

    def hessian(self, xr):
        return np.zeros(self.ambient_dim, self.dim, self.dim)
