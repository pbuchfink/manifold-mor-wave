'''Interface classes to interface with models from hamiltonian_models.'''

from warnings import warn
import os

import numpy as np
import torch
from manifold_mor.fields.symplectic import HamiltonianVectorField
from manifold_mor.models.basic import Model
from manifold_mor.time_steppers.hamiltonian import HamiltonianModelTimeStepper

from hamiltonian_models.base import HamiltonianSystem
from hamiltonian_models.integrators.base import TimeDataList
# from hamiltonian_models.linear_elasticity import LinearElasticBeamProblem
from hamiltonian_models.wave import (FixedEndsLinearWaveProblem, SineGordonProblem,
                         TravellingBumpLinearWaveProblem,
                         TravellingFrontLinearWaveProblem)


class HamiltonianModel(Model):
    def __init__(self, ham_sys, time_stepper, sys_dim, \
        hamiltonian_in_torch=None, hamiltonian_vf_in_torch=None):

        assert isinstance(ham_sys, HamiltonianSystem)
        assert isinstance(time_stepper, HamiltonianModelTimeStepper)
        self.vector_field = HamiltonianVectorField(
            ham_sys,
            sys_dim,
            time_stepper,
            hamiltonian_in_torch=hamiltonian_in_torch,
            hamiltonian_vf_in_torch=hamiltonian_vf_in_torch,
        )
        self.sys_dim = sys_dim
        self.ham_sys = ham_sys
        super().__init__('hamiltonian_' + ham_sys.__class__.__name__, time_stepper)
        self.time_stepper.residual.register_model(self)

        # sanity check
        if self.vector_field.is_time_dependent():
            warn('For time-dependent HamiltonianVectorField, check that set_mu is not too expensive.')

    def visualize(self, filename, sol_x, sol_t, subsample=None):
        folder = os.path.dirname(filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        # subsample if required
        n_t = len(sol_t)
        if subsample is None:
            subsample = n_t
        n_vis_t = min(subsample, n_t)
        idx_in_t = np.linspace(0, n_t, n_vis_t, dtype=int, endpoint=False)
        
        self.ham_sys.visualize(filename, 
            TimeDataList(
                td_t=sol_t[idx_in_t],
                td_data=[self.ham_sys.phase_space.from_numpy(x) for x in sol_x[idx_in_t]]
            )
        )

    def initial_value(self, mu):
        return self.vector_field.to_numpy(self.ham_sys.initial_value(mu))

    def check_mu(self, mu):
        return self.ham_sys.check_parameters(mu)

    def get_dim(self):
        return self.sys_dim

    def set_mu(self, mu):
        self.vector_field.set_mu(mu)

    def apply_weighting_matrix(self, y):
        return self.vector_field.apply_weighting_matrix(y)


class LinearWaveModel(HamiltonianModel):
    def __init__(self, ham_sys, time_stepper, sys_dim, diri_left=0., diri_right=0.):
        hamiltonian_in_torch = LinearWaveHamiltonianInTorch(
            ham_sys.dx,
            ham_sys.n_x-2, #minus 2 (only inner points)
            diri_left=diri_left,
            diri_right=diri_right
        )
        super().__init__(
            ham_sys, time_stepper, sys_dim,
            hamiltonian_in_torch=hamiltonian_in_torch,
            hamiltonian_vf_in_torch=LinearWaveHamiltonianVectorFieldInTorch(hamiltonian_in_torch),
        )


class FixedEndsLinearWaveModel(LinearWaveModel):
    def __init__(self, integrator_name, dt, l=1., n_x=514):
        sys_dim = 2 * n_x # factor 2: (q,p)
        time_stepper = HamiltonianModelTimeStepper.from_integrator_name(integrator_name, dt)
        time_stepper.non_linear_solver.rel_tol = -1 #disable abortion upon rel_err
        super().__init__(FixedEndsLinearWaveProblem(l, n_x), time_stepper, sys_dim)


class TravellingBumpModel(LinearWaveModel):
    def __init__(self, integrator_name, dt, l=1., n_x=514):
        sys_dim = 2 * n_x # factor 2: (q,p)
        time_stepper = HamiltonianModelTimeStepper.from_integrator_name(integrator_name, dt)
        time_stepper.non_linear_solver.rel_tol = -1 #disable abortion upon rel_err
        super().__init__(TravellingBumpLinearWaveProblem(l, n_x), time_stepper, sys_dim)
    


class TravellingFrontModel(LinearWaveModel):
    def __init__(self, integrator_name, dt, l=1., n_x=514):
        sys_dim = 2 * n_x # factor 2: (q,p)
        time_stepper = HamiltonianModelTimeStepper.from_integrator_name(integrator_name, dt)
        time_stepper.non_linear_solver.rel_tol = -1 #disable abortion upon rel_err
        super().__init__(TravellingFrontLinearWaveProblem(l, n_x), time_stepper, sys_dim, diri_left=1.)


class LinearWaveHamiltonianInTorch(torch.nn.Module):
    '''Hamiltonian for linear wave equation in pyTorch.'''
    def __init__(self, dx, n_x, diri_left=0., diri_right=0.):
        super().__init__()
        self.conv_diff_zero_rb = torch.nn.Conv1d(1, 1, 3, 1, 1, bias=None)
        self.conv_diff_zero_rb.weight.requires_grad = False
        self.conv_diff_zero_rb.weight.data = 1/dx * torch.tensor([1, -2, 1])[None, None, :]

        if all(np.isclose([diri_left, diri_right], 0.)):
            self.zero_pad = True
        else:
            self.zero_pad = False
            self.conv_diff_w_bdry = torch.nn.Conv1d(1, 1, 3, 1, 0, bias=None)
            self.conv_diff_w_bdry.weight = self.conv_diff_zero_rb.weight
            self.diri_left = diri_left
            self.diri_right = diri_right
        self.dx = dx

    def compute_conv_diff(self, in_q, with_boundary):
        if with_boundary and not self.zero_pad:
            in_q = torch.nn.functional.pad(in_q, (0, 1), 'constant', self.diri_right)
            in_q = torch.nn.functional.pad(in_q, (1, 0), 'constant', self.diri_left)
            return self.conv_diff_w_bdry(in_q.unsqueeze(1)).squeeze(1)
        else:
            return self.conv_diff_zero_rb(in_q.unsqueeze(1)).squeeze(1)

    def H_product_y(self, in_q, in_p, mu, with_boundary):
        out_q = -mu['c']**2 * self.compute_conv_diff(in_q, with_boundary)
        out_p = in_p / self.dx
        return out_q, out_p

    def forward(self, tensor_in, mu, with_boundary=True):
        in_q, in_p = torch.split(tensor_in, tensor_in.shape[-1]//2, -1)
        H_op_q, H_op_p = self.H_product_y(in_q, in_p, mu, with_boundary)
        # kinetic part of the Hamiltonian
        ham_kin = self.dx/2 * torch.sum(H_op_p * in_p, dim=-1)
        # potential part of the Hamiltonian
        ham_pot = self.dx/2 * torch.sum(H_op_q * in_q, dim=-1)
        return ham_kin + ham_pot, ham_pot, ham_kin

class LinearWaveHamiltonianVectorFieldInTorch(torch.nn.Module):
    '''Hamiltonian vector field for linear wave equation in pyTorch.'''
    def __init__(self, hamiltonian_in_torch):
        super().__init__()
        self.hamiltonian_in_torch = hamiltonian_in_torch
    
    def grad_Ham(self, tensor_in, mu):
        in_q, in_p = torch.split(tensor_in, tensor_in.shape[-1]//2, -1)
        H_op_q, H_op_p = self.hamiltonian_in_torch.H_product_y(in_q, in_p, mu, True)
        return torch.cat([H_op_q, H_op_p], -1)
    
    def forward(self, tensor_in, mu):
        in_q, in_p = torch.split(tensor_in, tensor_in.shape[-1]//2, -1)
        H_op_q, H_op_p = self.hamiltonian_in_torch.H_product_y(in_q, in_p, mu, True)
        return torch.cat([H_op_p, -H_op_q], -1)

class SineGordonModel(HamiltonianModel):
    LOSS_QUADRATIC = 'quadratic'
    LOSS_HAMILTONIAN = 'Hamiltonian'
    LOSS_TYPES = (LOSS_QUADRATIC, LOSS_HAMILTONIAN)
    def __init__(self, integrator_name, dt, loss_type=LOSS_QUADRATIC, n_x=514):
        assert loss_type in self.LOSS_TYPES
        sys_dim = 2 * n_x # factor 2: (q,p)
        time_stepper = HamiltonianModelTimeStepper.from_integrator_name(integrator_name, dt)
        time_stepper.non_linear_solver.rel_tol = -1 #disable abortion upon rel_err
        ham_sys = SineGordonProblem(n_x)
        hamiltonian_in_torch = SineGordonHamiltonianInTorch(ham_sys.dx, loss_type)
        super().__init__(
            ham_sys, time_stepper, sys_dim,
            hamiltonian_in_torch=hamiltonian_in_torch,
            hamiltonian_vf_in_torch=SineGordonHamiltonianVectorFieldInTorch(hamiltonian_in_torch)
        )


class SineGordonHamiltonianInTorch(torch.nn.Module):
    '''Hamiltonian for Sine-Gordon equation in pyTorch.'''
    def __init__(self, dx, loss_type):
        super().__init__()
        self.conv_diff_zero_rb = torch.nn.Conv1d(1, 1, 3, 1, 1, bias=None)
        self.conv_diff_zero_rb.weight.requires_grad = False
        self.conv_diff_zero_rb.weight.data = 1/dx * torch.tensor([1, -2, 1])[None, None, :]

        self.conv_diff_w_bdry = torch.nn.Conv1d(1, 1, 3, 1, 0, bias=None)
        self.conv_diff_w_bdry.weight = self.conv_diff_zero_rb.weight
        self.diri_left = 1.1519552880819725e-05 #value copied from initial value
        self.diri_right = 2*np.pi
        self.loss_type = loss_type
        self.dx = dx

    def compute_conv_diff(self, in_q, with_boundary):
        if with_boundary:
            in_q = torch.nn.functional.pad(in_q, (0, 1), 'constant', self.diri_right)
            in_q = torch.nn.functional.pad(in_q, (1, 0), 'constant', self.diri_left)
            return self.conv_diff_w_bdry(in_q.unsqueeze(1)).squeeze(1)
        else:
            return self.conv_diff_zero_rb(in_q.unsqueeze(1)).squeeze(1)

    def H_product_y(self, in_q, in_p, mu, with_boundary):
        out_q = -self.compute_conv_diff(in_q, with_boundary)
        out_p = in_p / self.dx
        return out_q, out_p

    def forward(self, tensor_in, mu, with_boundary=True):
        in_q, in_p = torch.split(tensor_in, tensor_in.shape[-1]//2, -1)
        H_op_q, H_op_p = self.H_product_y(in_q, in_p, mu, with_boundary)
        # kinetic part of the Hamiltonian
        ham_kin = self.dx/2 * torch.sum(H_op_p * in_p, dim=-1)
        # potential part of the Hamiltonian
        ham_pot = self.dx/2 * torch.sum(H_op_q * in_q, dim=-1)
        # add non-quadratic part if desired
        if self.loss_type == SineGordonModel.LOSS_HAMILTONIAN:
            ham_pot += self.dx**2 * torch.sum(1 - torch.cos(in_q[:, 1:-1]), dim=-1)
        return ham_kin + ham_pot, ham_pot, ham_kin

class SineGordonHamiltonianVectorFieldInTorch(torch.nn.Module):
    '''Hamiltonian vector field for linear wave equation in pyTorch.'''
    def __init__(self, hamiltonian_in_torch):
        super().__init__()
        self.hamiltonian_in_torch = hamiltonian_in_torch
    
    def grad_Ham(self, tensor_in, mu):
        in_q, in_p = torch.split(tensor_in, tensor_in.shape[-1]//2, -1)
        H_op_q, H_op_p = self.hamiltonian_in_torch.H_product_y(in_q, in_p, mu, True)
        # non quadratic part
        non_quad = self.hamiltonian_in_torch.dx * torch.sin(in_q)
        return torch.cat([H_op_q + non_quad, H_op_p], -1)

    def forward(self, tensor_in, mu):
        in_q, in_p = torch.split(tensor_in, tensor_in.shape[-1]//2, -1)
        H_op_q, H_op_p = self.hamiltonian_in_torch.H_product_y(in_q, in_p, mu, True)
        # non quadratic part
        non_quad = self.hamiltonian_in_torch.dx * torch.sin(in_q)
        return torch.cat([H_op_p, -H_op_q - non_quad], -1)
