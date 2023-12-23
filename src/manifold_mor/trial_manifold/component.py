"""A class for a manifold composed of different manifolds.
It is used to use different components with different charts.
E.g. to not reduce boundary nodes."""
from typing import TYPE_CHECKING
import numpy as np
from manifold_mor.mor.snapshots import Snapshots, SubsampledSnapshots
from manifold_mor.trial_manifold.basic import (TrainableTrialManifold,
                                               TrialManifold)
from scipy.linalg import block_diag

if TYPE_CHECKING:
    from manifold_mor.fields.vector import VectorField
    from manifold_mor.fields.vector_reduced import VectorFieldProjector


class ComponentwiseManifold(TrialManifold):
    def __init__(self, manifolds, component_indices): #TODO: call constructor of TrialManifold
        assert isinstance(component_indices, (tuple, list, np.ndarray)) \
            and all(isinstance(idx, (tuple, list, np.ndarray)) for idx in component_indices)
        component_indices = tuple(np.array(idx) for idx in component_indices)
        assert isinstance(manifolds, (tuple, list))
        assert len(component_indices) == len(manifolds)
        merged_component_indices = np.hstack(component_indices)
        assert merged_component_indices.size == np.unique(merged_component_indices).size, 'there are duplicate indices'
        # assert all names are unique such that manifold name can be used as readable identifier
        manifold_names = set(mnf.name for mnf in manifolds)
        assert len(manifold_names) == len(manifolds), 'manifold names have to be unique'

        self.n_comp = len(component_indices)
        self.component_indices = component_indices
        self.inv_component_indices = np.argsort(merged_component_indices)
        self.manifolds = manifolds
        self.dim_stops = None
        self.dims = tuple(mnf.dim for mnf in manifolds)
        self.ambient_dims = tuple(mnf.ambient_dim for mnf in manifolds)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'dims' and all(not v is None for v in value):
            self.dim = np.sum(self.dims)
            dims_cumsum = np.hstack(((0,), np.cumsum(self.dims)))
            self.dim_stops = tuple(zip(dims_cumsum[:-1], dims_cumsum[1:]))
        if name == 'ambient_dims' and all(not v is None for v in value):
            self.ambient_dim = np.sum(self.ambient_dims)

    def map(self, xr):
        assert xr.shape[-1] == self.dim
        x = np.hstack(tuple(mnf.map(xr[..., ds[0]:ds[1]]) for mnf, ds in zip(self.manifolds, self.dim_stops)))
        return x[..., self.inv_component_indices]

    def inv_map(self, x):
        assert x.shape[-1] == self.ambient_dim
        return np.hstack(tuple(mnf.inv_map(x[..., idcs]) for mnf, idcs in zip(self.manifolds, self.component_indices)))

    def tangent_map(self, xr):
        assert xr.shape[-1] == self.dim
        tangent_map = block_diag(*(mnf.tangent_map(self.split_xr(xr, i_mnf)) for i_mnf, mnf in enumerate(self.manifolds)))
        return tangent_map[self.inv_component_indices]

    def encoder_tangent_map(self, x):
        assert x.shape[-1] == self.ambient_dim
        encoder_tangent_map = block_diag(*(mnf.encoder_tangent_map(x[idcs]) for mnf, idcs in zip(self.manifolds, self.component_indices)))
        return encoder_tangent_map[:, self.inv_component_indices]

    def is_valid(self):
        return all(mnf.is_valid() for mnf in self.manifolds) \
            and all(not dim is None for dim in self.dims) \
            and all(not ambient_dim is None for ambient_dim in self.ambient_dims) \
            and self.dim == sum(self.dims) \
            and self.ambient_dim == sum(self.ambient_dims) \
            and all(len(idx) == ambient_dim for idx, ambient_dim in zip(self.component_indices, self.ambient_dims))
    
    def split_xr(self, xr, i):
        assert xr.shape[-1] == self.dim
        ds = self.dim_stops[i]
        return xr[ds[0]:ds[1]]

    def pullback_vector_field(self, vector_field: 'VectorField', projector: 'VectorFieldProjector'):
        raise NotImplementedError()


class ComponentwiseTrainableManifold(ComponentwiseManifold, TrainableTrialManifold):
    def __init__(self, manifolds, component_indices, idx_trainable, model_name, name):
        super(ComponentwiseManifold, self).__init__(model_name=model_name, name=name)
        super().__init__(manifolds=manifolds, component_indices=component_indices)
        self.idx_trainable = idx_trainable
        self.trainable_manifold = self.manifolds[idx_trainable]
        assert isinstance(self.trainable_manifold, TrainableTrialManifold)
    
    def get_subsampled_snapshots(self, snapshots: Snapshots, i_manifold: int = None) -> SubsampledSnapshots:
        i_manifold = i_manifold or self.idx_trainable
        return snapshots.get_subsampled_snapshots(self.component_indices[i_manifold], str(i_manifold))

    def train(self, snapshots: Snapshots, training_params: dict, logger=None, model=None, **kwargs):
        assert snapshots.get_dim() == self.trainable_manifold.ambient_dim
        summary = self.trainable_manifold.train(
            snapshots,
            training_params,
            logger=logger,
            model=model,
            **kwargs
        )
        self.dims = tuple(mnf.dim for mnf in self.manifolds)
        self.ambient_dims = tuple(mnf.ambient_dim for mnf in self.manifolds)

        self.loggable_parameters.update(summary)
        return summary
