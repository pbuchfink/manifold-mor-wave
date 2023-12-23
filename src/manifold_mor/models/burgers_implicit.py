'''Burgers model which can also be used with implicit time integration.
The fluxes only fall into one if-case due to the positiviry of the solution.
This is the Burgers model used in Lee and Carlberg 2020.'''

import os
from os.path import splitext

import numpy as np
import torch
from manifold_mor.fields.vector import VectorField
from manifold_mor.models.basic import Model
from manifold_mor.time_steppers.basic import TimeStepper
from manifold_mor.time_steppers.solvers import NewtonNonlinearSolver
from manifold_mor.time_steppers.vector import \
    ImplicitEulerVectorFieldTimeStepperResidual
from pyevtk.hl import gridToVTK
from pyevtk.vtk import VtkGroup
from scipy.sparse import diags


class BurgersModel(Model):
    '''Parametric Burgers equation solved with Godunov flux and backward-Euler scheme. Riemann implicit.'''
    def __init__(self, grid_l=0., grid_r=100., n_points=256, source_coeff=2e-2):
        self.grid_l, self.grid_r, self.n_points = grid_l, grid_r, n_points
        self.source_coeff = source_coeff

        self.dx = (grid_r - grid_l) / (n_points - 1)
        self.grid = np.linspace(self.grid_l, self.grid_r, self.n_points)
        assert np.isclose(self.dx, np.diff(self.grid[0:2]))

        self.vector_field = BurgersVectorField(self, source_coeff)
        time_stepper = TimeStepper(
            ImplicitEulerVectorFieldTimeStepperResidual(self),
            NewtonNonlinearSolver(rel_tol=-1, abs_tol=1e-6, max_iter=10)
        )
        super().__init__(name='burgers', time_stepper=time_stepper)

    def check_mu(self, mu):
        return 'u_bc_left' in mu.keys() and 4.25 <= mu['u_bc_left'] <= 5.5 \
            and 'source_exponent' in mu.keys() and 0.015 <= mu['source_exponent'] <= 0.03

    def get_dim(self):
        return self.n_points
            
    def initial_value(self, mu):
        return np.ones((self.n_points,))

    def visualize(self, filename, sol_x, sol_t):
        '''
            Output results of solve() to pvd file (as input for paraview)
            for displacemnet and momentum.
        '''
        # generate folders for visualization output
        folder = os.path.dirname(filename)
        if not os.path.exists(folder):
            os.makedirs(folder)

        split_filename = splitext(filename)
        n = int(np.ceil(np.log10(len(sol_t))))
        # plot displacement and momentum
        z = np.array([0])
        file_path = split_filename[0]
        g = VtkGroup(file_path)
        for i_t, (t, x) in enumerate(zip(sol_t, sol_x)):
            file_name_i = gridToVTK(file_path + ('{0:0' + str(n) + 'd}').format(i_t), self.grid, z, z, pointData={"u" : x})
            g.addFile(filepath=file_name_i, sim_time=t)
        g.save()


class BurgersVectorField(VectorField):
    def __init__(self, model, source_coeff, mu={'u_bc_left': 5, 'source_exponent': 0.02}):
        super().__init__(mu=mu)
        self.model = model
        self.source_coeff = source_coeff

    def eval(self, x):
        # numerical flux
        x_sq = x**2
        f = -.5 * x_sq
        f[1:] += .5 * x_sq[:-1]
        f *= 1/self.model.dx
        # source term
        #TODO: why no cell averages?
        #TODO: offset by 1?
        f += self.source_coeff * np.exp(self.mu['source_exponent'] * self.model.grid)
        # boundary condition
        f[0] += 1/(2*self.model.dx) * self.mu['u_bc_left']**2
        return f

    def compute_derivative(self, x):
        # numerical flux
        #TODO
        jacobian = diags(-x) + diags(x[:-1], -1)
        jacobian *= 1/self.model.dx
        return jacobian.toarray() #TODO sparse implementation

    #TODO: for efficiency it might be better to use apply functions
    def apply_derivative(self, x, y):
        modified_y = False
        if len(y.shape) == 1:
            y = y[np.newaxis, :]
            modified_y = True

        # numerical flux
        res = - y * x / self.model.dx
        res[:, 1:] -= res[:, :-1]

        if modified_y:
            res = res[0]
        return res

    #TODO: for efficiency it might be better to use apply functions
    #Problem with inverse: probably different inverses for different integrators required
    def apply_inverse_derivative(self, x, y):
        modified_y = False
        if len(y.shape) == 1:
            y = y[np.newaxis, :]
            modified_y = True

        res = - self.model.dx * np.cumsum(y, -1) * (1/x)

        if modified_y:
            res = res[0]
        return res


class TorchBurgersModel(Model):
    def __init__(self, grid_l=0., grid_r=100., n_points=256, source_coeff=2e-2):
        self.grid_l, self.grid_r, self.n_points = grid_l, grid_r, n_points
        self.source_coeff = source_coeff

        self.dx = (grid_r - grid_l) / (n_points - 1)
        self.grid = torch.linspace(self.grid_l, self.grid_r, self.n_points)

        self.vector_field = TorchBurgersVectorField(self)
        # the time stepper is a dummy to make all functionalities work
        # in the future time_stepper = None might be an option
        time_stepper = TimeStepper(
            ImplicitEulerVectorFieldTimeStepperResidual(self),
            NewtonNonlinearSolver(rel_tol=-1, abs_tol=1e-6, max_iter=10)
        )
        super().__init__('torch_burgers', time_stepper)

    def check_mu(self, mu):
        return 'u_bc_left' in mu.keys() and 4.25 <= mu['u_bc_left'] <= 5.5 \
            and 'source_exponent' in mu.keys() and 0.015 <= mu['source_exponent'] <= 0.03

    def get_dim(self):
        return self.n_points
            
    def initial_value(self, mu):
        return torch.ones((self.n_points,))
    
    def solve(self, t_0, t_end, dt, mu, logger=None, hook_fcns=None):
        raise NotImplementedError('Use BurgersModel instead.')


class TorchBurgersVectorField(VectorField):
    def __init__(self, model: TorchBurgersModel):
        super().__init__()
        self.model = model

    def eval(self, x):
        # numerical flux
        x_sq = x**2
        f = -.5 * x_sq
        f[..., 1:] += .5 * x_sq[..., :-1]
        f *= 1/self.model.dx
        # source term
        #TODO: why no cell averages?
        #TODO: offset by 1?
        f += self.model.source_coeff * np.exp(self.mu['source_exponent'] * self.model.grid)
        # boundary condition
        f[..., 0] += 1/(2*self.model.dx) * self.mu['u_bc_left']**2
        return f

    def compute_derivative(self, x):
        raise NotImplementedError()
