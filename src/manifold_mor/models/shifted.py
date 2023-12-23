'''A superclass of model which performs the coordinate transformation from z to x
in the sense of Buchfink, Glas, Haasdonk: Symplectic MOR on manifolds.'''
from typing import Union

import numpy as np
import torch

from manifold_mor.fields.symplectic import HamiltonianVectorField
from manifold_mor.fields.vector import VectorField
from manifold_mor.models.basic import Model
from manifold_mor.mor.snapshots import Snapshots


class ShiftedModel(Model):
    def __init__(self, model: Model):
        """Shift the model to initial value zero.

        Args:
            model (Model): the model to shift (aka the unshifted model).
        """
        self.model = model
        assert hasattr(self.model, 'vector_field')
        if isinstance(self.model.vector_field, HamiltonianVectorField):
            self.vector_field = ShiftedHamiltonianVectorField(self)
        else:
            self.vector_field = ShiftedVectorField(self)
        super().__init__('shifted_' + model.name, model.time_stepper.copy())
        self.time_stepper.residual.register_model(self)
        self._cached_initial_values = dict()
        self.shift_numpy = None
        self.shift_torch = None

    def get_shift(self, mu):
        mu_idx = mu['mu_idx']
        try:
            return self._cached_initial_values[mu_idx]
        except KeyError:
            initial_value = self.model.initial_value(mu)
            self._cached_initial_values[mu_idx] = initial_value
            return initial_value

    def initial_value(self, mu):
        return np.zeros(self.get_dim())

    def shift_x(self, x: Union[np.ndarray, torch.Tensor]):
        if isinstance(x, np.ndarray):
            return x + self.shift_numpy
        elif isinstance(x, torch.Tensor):
            return x + self.shift_torch

    def check_mu(self, mu):
        return self.model.check_mu(mu)

    def get_dim(self):
        return self.model.get_dim()

    def set_mu(self, mu):
        super().set_mu(mu)
        shift = self.get_shift(mu)
        if isinstance(shift, torch.Tensor):
            self.shift_torch = shift
            self.shift_numpy = shift.numpy()
        elif isinstance(shift, np.ndarray):
            self.shift_numpy = shift
            self.shift_torch = torch.tensor(self.shift_numpy, dtype=torch.get_default_dtype())
        else:
            raise NotImplementedError('Unkown type for shift: {}'.format(type(shift)))
        self.model.set_mu(mu)

    def visualize(self, filename, sol_x, sol_t, **kwargs):
        self.model.visualize(filename, sol_x, sol_t, **kwargs)

    def unshift_snapshots(self, snapshots: Snapshots):
        new_snap_mat = np.empty_like(snapshots.matrix)
        for i_mu, mu in enumerate(snapshots.get_mus()):
            new_snap_mat[i_mu] = snapshots.matrix[i_mu] + self.get_shift(mu)
        return Snapshots(
            snapshots.model_name,
            'unshifted_' + snapshots.name,
            snapshots.get_mus(),
            *snapshots.get_temporal_data(),
            snapshots.sol_t,
            new_snap_mat,
            hook_fcns_results=snapshots.hook_fcns_results,
        )


class ShiftedVectorField(VectorField):
    def __init__(self, shifted_model, mu=None):
        assert isinstance(shifted_model, ShiftedModel)
        self.model = shifted_model.model
        self.shifted_model = shifted_model
        self.vector_field = self.model.vector_field
        self._cached_initial_values = {}
        super().__init__('shifted_' + self.vector_field.name, mu=mu)

    def eval(self, x):
        return self.vector_field.eval(self.shifted_model.shift_x(x))

    def compute_derivative(self, x):
        return self.vector_field.compute_derivative(self.shifted_model.shift_x(x))

    def apply_derivative_y(self, x, y):
        return self.vector_field.apply_derivative_y(self.shifted_model.shift_x(x), y)

    def _inv_M_dt_Df_y(self, x, dt, y):
        if hasattr(self.vector_field, '_inv_M_dt_Df_y'):
            return self.vector_field._inv_M_dt_Df_y(self.shifted_model.shift_x(x), dt, y)
        else:
            raise NotImplementedError()

    def set_mu(self, mu):
        super().set_mu(mu)
        self.vector_field.set_mu(mu)


# TODO: duplicate code. could be resolved by introducing charts on high-dim mnf.
class ShiftedHamiltonianVectorField(HamiltonianVectorField):
    def __init__(self, shifted_model, mu=None):
        assert isinstance(shifted_model, ShiftedModel)
        assert isinstance(shifted_model.model.vector_field, HamiltonianVectorField)
        self.model = shifted_model.model
        self.shifted_model = shifted_model
        self.vector_field = self.model.vector_field
        self.initial_value = None
        # cache initial values to avoid computing them ofted
        # (inesp. during training which uses eval_in_torch)
        self._cached_initial_values = dict()
        # TODO: nicer solution to get inner indices
        N = shifted_model.get_dim() // 2
        self.idx_inner = list(range(1, N-1)) + list(range(N+1, 2*N-1))
        super().__init__(
            self.vector_field.ham_sys,
            self.vector_field.sys_dim,
            self.model.time_stepper,
            'shifted_' + self.vector_field.name,
            mu=mu,
            hamiltonian_in_torch=self.vector_field.hamiltonian_in_torch,
            hamiltonian_vf_in_torch=self.vector_field.hamiltonian_vf_in_torch,
        )
    
    def eval(self, x):
        return self.vector_field.eval(self.shifted_model.shift_x(x))

    def eval_in_torch(self, tensor_in, mu):  #TODO: will soon be replaced
        initial_value = self.compute_initial_value_inner_in_torch(mu, dtype=tensor_in.dtype, device=tensor_in.device)
        return self.vector_field.eval_in_torch(tensor_in + initial_value, mu)

    def compute_derivative(self, x):
        return self.vector_field.compute_derivative(self.shifted_model.shift_x(x))

    def apply_derivative_y(self, x, y):
        return self.vector_field.apply_derivative_y(self.shifted_model.shift_x(x), y)

    def set_mu(self, mu):
        super().set_mu(mu)
        self.vector_field.set_mu(mu)
        self.initial_value = self.model.initial_value(mu)

    def Ham(self, x):
        return self.vector_field.Ham(self.shifted_model.shift_x(x))

    def gradHam(self, x):
        return self.vector_field.gradHam(self.shifted_model.shift_x(x))

    def hessianHam_y(self, x, y):
        return self.vector_field.hessianHam_y(self.shifted_model.shift_x(x), y)

    def J_y(self, x, y):
        return self.vector_field.J_y(self.shifted_model.shift_x(x), y)

    def _inv_M_dt_Df_y(self, x, dt, y):
        return self.vector_field._inv_M_dt_Df_y(self.shifted_model.shift_x(x), dt, y)

    def eval_hamiltonian_in_torch(self, tensor_in, mu, with_boundary=True): #TODO: will soon be replaced
        # yet only used for differences (where initial_value cancels)
        if with_boundary:
            initial_value = self.compute_initial_value_inner_in_torch(mu, dtype=tensor_in.dtype, device=tensor_in.device)
            return super().eval_hamiltonian_in_torch(tensor_in + initial_value, mu, with_boundary)
        else:
            return super().eval_hamiltonian_in_torch(tensor_in, mu, with_boundary)

    def eval_grad_Ham_in_torch(self, tensor_in, mu): #TODO: will soon be replaced
        initial_value = self.compute_initial_value_inner_in_torch(mu, dtype=tensor_in.dtype, device=tensor_in.device)
        return super().eval_grad_Ham_in_torch(tensor_in + initial_value, mu)

    def compute_initial_value_inner_in_torch(self, mu, dtype, device): #TODO: will soon be replaced
        try:
            idx = mu['mu_idx']
        except KeyError:
            raise RuntimeError('mu requires a fixed index "mu_idx"')
        initial_value = self._cached_initial_values.get(idx, None)
        if initial_value is None:
            self.model.set_mu(mu)
            initial_value = torch.tensor(
                self.model.initial_value(mu)[self.idx_inner],
                dtype=dtype,
                device=device
            )
            self._cached_initial_values[idx] = initial_value
        return initial_value
    
