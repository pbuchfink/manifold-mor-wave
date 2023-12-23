"""Special TimeStepper for Hamiltonian vector fields.
It interfaces to the integrators from hamiltonian_models."""
import numpy as np
from hamiltonian_models.base import HamiltonianSystem
from hamiltonian_models.dyn_sys import DynamicalSystem
from hamiltonian_models.integrators.implicit_midpoint import \
    ImplicitMidpointIntegrator
from hamiltonian_models.integrators.stormer_verlet import \
    SeparableStormerVerletIntegrator
from hamiltonian_models.wave import TravellingWaveClosedFormulaIntegrator
from manifold_mor.models.shifted import ShiftedHamiltonianVectorField
from manifold_mor.time_steppers.basic import TimeStepper
from manifold_mor.time_steppers.solvers import NewtonNonlinearSolver
from manifold_mor.time_steppers.vector import VectorFieldTimeStepperResidual


class HamiltonianModelTimeStepper(TimeStepper):
    INTEGRATOR_STOERMER_VERLET = 'stoermer_verlet'
    INTEGRATOR_IMPL_MIDPOINT = 'implicit_midpoint'
    INTEGRATOR_CLOSED_FORMULA_TRAVELLING_WAVE = 'closed_formula_travelling_wave'

    def __init__(self, time_stepper_residual, non_linear_solver=None):
        assert isinstance(time_stepper_residual, SymplecticTimeStepperResidual)
        super().__init__(
            time_stepper_residual,
            non_linear_solver or NewtonNonlinearSolver()
        )
        self.dt = time_stepper_residual.symplectic_integrator._dt

    @classmethod
    def from_integrator_name(cls, integrator_name, dt):
        if integrator_name == cls.INTEGRATOR_STOERMER_VERLET:
            residual = SeparableStormerVerletTimeStepperResidual(dt)
        elif integrator_name == cls.INTEGRATOR_IMPL_MIDPOINT:
            residual = ImplicitMidpointTimeStepperResidual(dt)
        elif integrator_name == cls.INTEGRATOR_CLOSED_FORMULA_TRAVELLING_WAVE:
            return ClosedFormulaHamiltonianModelTimeStepper(
                ClosedFormulaTimeStepperResidual(TravellingWaveClosedFormulaIntegrator(dt))
            )
        else:
            raise NotImplementedError('Unknown integrator_name: {}'.format(integrator_name))
        return cls(residual)
    
    def solve(self, t_0, t_end, dt, mu, hook_fcns=None, callbacks=None):
        assert np.isclose(dt, self.dt)
        return super().solve(t_0, t_end, dt, mu, hook_fcns=hook_fcns, callbacks=callbacks)


class SymplecticTimeStepperResidual(VectorFieldTimeStepperResidual):
    def __init__(self, symplectic_integrator, model=None):
        self.symplectic_integrator = symplectic_integrator
        self.dyn_sys = None
        super().__init__(model)

    def is_valid(self):
        return super().is_valid() \
            and self.dyn_sys is not None
    
    def is_time_dependent(self):
        return self.vector_field.is_time_dependent()


class ImplicitMidpointTimeStepperResidual(SymplecticTimeStepperResidual):
    def __init__(self, dt, model=None):
        super().__init__(symplectic_integrator=ImplicitMidpointIntegrator(dt), model=model)

    def register_model(self, model):
        super().register_model(model)
        self.dyn_sys = InterfaceDynamicalSystem(model.vector_field)
        self.vector_field = model.vector_field

    def evaluate(self, x, x_old, dt):
        return self.symplectic_integrator.residual(self.dyn_sys, x, x_old, self.vector_field.mu)

    def evaluate_derivative(self, x, x_old, dt):
        raise NotImplementedError('Use apply_inv_derivative instead')

    def apply_derivative_y(self, x, x_old, dt, y):
        return self.symplectic_integrator.apply_jacobian(self.dyn_sys, x, x_old, self.vector_field.mu, y)

    def apply_inv_derivative(self, x, res, x_old, dt):
        return self.symplectic_integrator.apply_inv_jacobian(self.dyn_sys, x, x_old, self.vector_field.mu, res)

    def copy(self):
        return ImplicitMidpointTimeStepperResidual(self.symplectic_integrator._dt)


class SeparableStormerVerletTimeStepperResidual(SymplecticTimeStepperResidual):
    def __init__(self, dt, model=None):
        super().__init__(symplectic_integrator=SeparableStormerVerletIntegrator(dt), model=model)

    def register_model(self, model):
        super().register_model(model)
        if hasattr(model.vector_field, 'ham_sys'):
            self.dyn_sys = model.vector_field.ham_sys
        else:
            self.dyn_sys = InterfaceHamiltonianSystem(model.vector_field)
        self.vector_field = model.vector_field

    def evaluate(self, x, x_old, dt):
        return self.symplectic_integrator.residual(
            self.dyn_sys,
            self.dyn_sys.phase_space.from_numpy(x),
            self.dyn_sys.phase_space.from_numpy(x_old),
            self.vector_field.mu
        ).to_numpy()

    def evaluate_derivative(self, x, x_old, dt):
        raise NotImplementedError('Use apply_inv_derivative instead')

    def apply_derivative_y(self, x, x_old, dt, y):
        return self.symplectic_integrator.apply_jacobian(
            self.dyn_sys,
            self.dyn_sys.phase_space.from_numpy(x),
            self.dyn_sys.phase_space.from_numpy(x_old),
            self.vector_field.mu,
            self.dyn_sys.phase_space.from_numpy(y)
        ).to_numpy()

    def apply_inv_derivative(self, x, res, x_old, dt):
        return self.symplectic_integrator.apply_inv_jacobian(
            self.dyn_sys,
            self.dyn_sys.phase_space.from_numpy(x),
            self.dyn_sys.phase_space.from_numpy(x_old),
            self.vector_field.mu,
            self.dyn_sys.phase_space.from_numpy(res)
        ).to_numpy()

    def copy(self):
        return SeparableStormerVerletTimeStepperResidual(self.symplectic_integrator._dt)

class InterfaceDynamicalSystem(DynamicalSystem):
    def __init__(self, vector_field):
        #TODO is_linear flag in manifold_mor framework (which avoids 1 computation of residual in integrator)
        super().__init__(is_linear=False)
        self.vector_field = vector_field

    # TODO: more general
    def _mass_y(self, mu, y):
        self.assert_same_mu(mu)
        return y
    
    def dxdt(self, x, mu):
        self.assert_same_mu(mu)
        return self.vector_field.eval(x)

    def apply_dxdt_jacobian(self, x, mu, y):
        self.assert_same_mu(mu)
        return self.vector_field.apply_derivative_y(x, y)

    def _inv_M_dt_Df_y(self, x, mu, dt, y):
        self.assert_same_mu(mu)
        if hasattr(self.vector_field, '_inv_M_dt_Df_y'):
            res = self.vector_field._inv_M_dt_Df_y(x, dt, y)
        else:
            jac = self.vector_field.compute_derivative(x)
            res = np.linalg.solve(np.eye(jac.shape[0]) - dt/2 * jac, y)
        return res

    def assert_same_mu(self, mu):
        vf_mu = self.vector_field.mu
        assert all(vf_mu[k] == v for k, v in mu.items())

    def initial_value(self, mu):
        raise NotImplementedError('Initial value is handled on the manifold mor side.')

    def solve(self, t_0, t_end, integrator, mu):
        raise NotImplementedError('Solve is handled on the manifold mor side.')

class InterfaceHamiltonianSystem(HamiltonianSystem):
    def __init__(self, vector_field):
        #TODO is_linear flag in manifold_mor framework (which avoids 1 computation of residual in integrator)
        super().__init__(phase_space=vector_field.phase_space, is_linear=False)
        self.vector_field = vector_field

    def Ham(self, x, mu):
        self.assert_same_mu(mu)
        return self.vector_field.Ham(x.to_numpy())

    def gradHam(self, x, mu):
        self.assert_same_mu(mu)
        return self.phase_space.from_numpy(
            self.vector_field.gradHam(x.to_numpy())
        )

    def J_y(self, x, mu, y):
        self.assert_same_mu(mu)
        return self.phase_space.from_numpy(
            self.vector_field.J_y(x.to_numpy(), y.to_numpy())
        )

    def assert_same_mu(self, mu):
        vf_mu = self.vector_field.mu
        assert all(vf_mu[k] == v for k, v in mu.items())

    def initial_value(self, mu):
        raise NotImplementedError('Initial value is handled on the manifold mor side.')

    def solve(self, t_0, t_end, integrator, mu):
        raise NotImplementedError('Solve is handled on the manifold mor side.')

    def update_mu(self, mu, update):
        return self.vector_field.update_mu(mu, update)


class ClosedFormulaHamiltonianModelTimeStepper(HamiltonianModelTimeStepper):
    '''A time-stepper that directly returns the solution form a closed formula.'''
    def solve(self, t_0, t_end, dt, mu, hook_fcns=None, callbacks=None):
        closed_formula_integrator = self.residual.symplectic_integrator
        if not (hook_fcns is None and callbacks is None):
            raise NotImplementedError()
        assert np.isclose(dt, closed_formula_integrator._dt)

        vector_field = self.residual.vector_field
        ham_sys = vector_field.ham_sys
        td_x, _ = ham_sys.solve(t_0, t_end, closed_formula_integrator, mu)

        x = np.array([datum.to_numpy() for datum in td_x._data])
        t = np.array(td_x._t)

        if isinstance(vector_field, ShiftedHamiltonianVectorField):
            x -= x[0]

        return x, t


class ClosedFormulaTimeStepperResidual(SymplecticTimeStepperResidual):
    def evaluate(self, x, x_old, dt):
        raise NotImplementedError()

    def evaluate_derivative(self, x, x_old, dt):
        raise NotImplementedError('Use apply_inv_derivative instead')

    def apply_derivative_y(self, x, x_old, dt, y):
        raise NotImplementedError()

    def apply_inv_derivative(self, x, res, x_old, dt):
        raise NotImplementedError()

    def copy(self):
        return ClosedFormulaTimeStepperResidual(self.symplectic_integrator)

