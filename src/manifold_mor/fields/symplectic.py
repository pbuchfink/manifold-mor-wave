'''General definition of a Hamiltonian vector field and
a special reduced version yet exclusive for the wave example.'''
from typing import TYPE_CHECKING
import numpy as np
from manifold_mor.fields.vector import VectorField
from manifold_mor.fields.vector_reduced import ReducedVectorField, VectorFieldProjector
from hamiltonian_models.base import QuadraticHamiltonianSystem, Vectorized

if TYPE_CHECKING:
    from manifold_mor.trial_manifold.basic import TrialManifold

class HamiltonianVectorField(VectorField):
    '''A vector field to compute the vector-field-pullback of a Hamiltonian system.
    Implements methods to use integrators of the Hamiltonian models framework.'''
    def __init__(self, ham_sys, sys_dim, time_stepper, name='hamiltonian_vector_field', mu=None, \
        hamiltonian_in_torch=None, hamiltonian_vf_in_torch=None):
        super().__init__(name, mu=mu)
        self.ham_sys = ham_sys
        self.sys_dim = sys_dim
        self.time_stepper = time_stepper
        self.hamiltonian_in_torch = hamiltonian_in_torch
        self.hamiltonian_vf_in_torch = hamiltonian_vf_in_torch
    
    def eval(self, x):
        return self.to_numpy(self.ham_sys.dxdt(self.from_numpy(x), self.mu))

    def compute_derivative(self, x):
        raise NotImplementedError('Use apply_derivative_y')

    def apply_derivative_y(self, x, y):
        return self.to_numpy(self.ham_sys.J_y(
            self.from_numpy(x),
            self.mu,
            self.ham_sys.hessianHam_y(
                self.from_numpy(x),
                self.mu,
                self.from_numpy(y)
            )
        ))

    def Ham(self, x):
        return self.ham_sys.Ham(self.from_numpy(x), self.mu)

    def gradHam(self, x):
        return self.to_numpy(self.ham_sys.gradHam(self.from_numpy(x), self.mu))

    def hessianHam_y(self, x: np.ndarray, y: np.ndarray):
        if isinstance(self.ham_sys, Vectorized):
            if isinstance(self.ham_sys, QuadraticHamiltonianSystem):
                forward_x = [] # not required for quadratic Hamiltonian systems
            else:
                forward_x = self.from_numpy(x)
            # process multiplication with hessian batched for all elements in y (if y is a list of vectors)
            return self.to_numpy(self.ham_sys.hessianHam_y(forward_x, self.mu, self.from_numpy(y)))
        else:
            forward_x = self.from_numpy(x)
            truncate_first_dim = False
            if y.ndim == 1:
                truncate_first_dim = True
                y = y.reshape((len(y), 1))
                result = np.empty_like(y)
                for i_y, y_vec in enumerate(y):
                    result[i_y, :] = self.to_numpy(self.ham_sys.hessianHam_y(forward_x, self.mu, self.from_numpy(y_vec)))
            if truncate_first_dim:
                return result[0, ...]
            else:
                return result.T

    def J_y(self, x, y):
        return self.to_numpy(self.ham_sys.J_y(self.from_numpy(x), self.mu, self.from_numpy(y)))

    def apply_weighting_matrix(self, y):
        return self.to_numpy(self.ham_sys.H_product_y(self.mu, self.from_numpy(y)))

    def _inv_M_dt_Df_y(self, x, dt, y):
        return self.to_numpy(self.ham_sys._inv_M_dt_Df_y(self.from_numpy(x), self.mu, dt, self.from_numpy(y)))

    def from_numpy(self, obj: np.ndarray):
        if obj.ndim > 1:
            assert obj.shape[-1] == self.sys_dim
        return self.ham_sys.phase_space.from_numpy(obj)

    def to_numpy(self, obj):
        return obj.to_numpy()

    def set_mu(self, mu):
        # avoid calling preassemble multiple times for same mu
        if not mu == self.mu:
            super().set_mu(mu)
            if hasattr(self.ham_sys, 'preassemble'):
                self.ham_sys.preassemble(mu, self.time_stepper.dt)

    def update_mu(self, mu, update):
        if hasattr(self.ham_sys, 'update_mu'):
            return self.ham_sys.update_mu(mu, update)
        else:
            return mu
    
    def eval_in_torch(self, tensor_in, mu):
        return self.hamiltonian_vf_in_torch(tensor_in, mu)

    def eval_hamiltonian_in_torch(self, tensor_in, mu, with_boundary=True):
        return self.hamiltonian_in_torch(tensor_in, mu, with_boundary)
    
    def eval_grad_Ham_in_torch(self, tensor_in, mu):
        return self.hamiltonian_vf_in_torch.grad_Ham(tensor_in, mu)

    def to(self, device):
        self.hamiltonian_vf_in_torch.to(device)
        self.hamiltonian_in_torch.to(device)


class ReducedHamiltonianVectorField(ReducedVectorField):
    '''assumes that self.vector_field.J_y is based on a tensor with inv(J) = -J'''
    def __init__(self, vector_field, manifold):
        assert isinstance(vector_field, HamiltonianVectorField)
        assert manifold.dim % 2 == 0
        projector = SymplecticVectorFieldProjector(manifold, vector_field, self)
        super().__init__(vector_field, manifold, projector)

    def J_y(self, x, y):
        return np.linalg.solve(self.inv_J_tensor(x), y)

    def inv_J_y(self, x, y):
        tan_map = self.manifold.tangent_map(x)
        return -tan_map.T @ self.vector_field.J_y(self.manifold.map(x), tan_map @ y)

    def inv_J_tensor(self, x: np.ndarray) -> np.ndarray:
        tensor = np.zeros([self.manifold.dim]*2)
        for i_col in range(self.manifold.dim):
            v = np.zeros(self.manifold.dim)
            v[i_col] = 1.
            tensor[:, i_col] = self.inv_J_y(x, v)
        return tensor

    def eval(self, x):
        red_tan = self.J_y(x, self.gradHam(x))
        # # compare with evaluation via projector
        # super_eval = super().eval(x)
        # assert np.allclose(
        #     red_tan,
        #     super_eval
        # )
        return red_tan

    def compute_derivative(self, x):
        # neglects derivatives of the tangent map (Quasi-Newton)
        tan_map = self.manifold.tangent_map(x)
        red_jac = tan_map.T @ self.vector_field.hessianHam_y(self.manifold.map(x), tan_map.T).T
        return self.J_y(x, red_jac)

    def Ham(self, x):
        return self.vector_field.Ham(self.manifold.map(x))

    def gradHam(self, x):
        tan_map = self.manifold.tangent_map(x)
        return tan_map.T @ self.vector_field.gradHam(self.manifold.map(x))

    def update_mu(self, mu, update):
        return self.vector_field.update_mu(mu, update)


class CanonicalReducedHamiltonianVectorField(ReducedHamiltonianVectorField):
    def __init__(self, vector_field, manifold):
        assert hasattr(manifold, 'J_y') and callable(manifold.J_y)
        assert hasattr(manifold, 'J_y') and callable(manifold.J_y)
        super().__init__(vector_field, manifold)

    def J_y(self, x, y):
        return self.manifold.J_y(x, y)
    
    def inv_J_y(self, x, y):
        return self.manifold.inv_J_y(x, y)


class SymplecticVectorFieldProjector(VectorFieldProjector):
    def __init__(
        self,
        manifold: "TrialManifold",
        vector_field: "VectorField",
        red_vector_field: ReducedHamiltonianVectorField,
    ) -> None:
        self.vector_field = vector_field
        self.red_vector_field = red_vector_field
        super().__init__(manifold)

    def project_to_reduced_manifold(self, x: np.ndarray, v: np.ndarray):
        tan_map = self.manifold.tangent_map(x)
        # minus from J_tensor and J_y cancels
        return self.red_vector_field.J_y(x, -tan_map.T @ self.vector_field.J_y(self.manifold.map(x), v))
