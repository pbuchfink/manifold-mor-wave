'''A reduced model that is responsible for the reduction of the incomming field.'''
import numpy as np
from manifold_mor.fields.symplectic import (
    CanonicalReducedHamiltonianVectorField, HamiltonianVectorField,
    ReducedHamiltonianVectorField)
from manifold_mor.fields.vector_reduced import VectorFieldProjector
from manifold_mor.models.basic import Model
from manifold_mor.time_steppers.basic import (HookFunction, TimeStepper,
                                              TimeStepperError)
from manifold_mor.time_steppers.covector import \
    CovectorFieldTimeStepperResidual
from manifold_mor.time_steppers.hamiltonian import HamiltonianModelTimeStepper
from manifold_mor.time_steppers.residual import TimeStepperResidual
from manifold_mor.time_steppers.vector import VectorFieldTimeStepperResidual
from manifold_mor.trial_manifold.basic import TrialManifold


class ReducedModel(Model):
    def __init__(
        self,
        manifold: TrialManifold,
        model: Model,
        time_stepper: TimeStepper,
    ) -> None:
        """Reduce a full-order |Model| (FOM) to a |TrialManifold| with a given |TimeStepper|.

        Args:
            manifold (TrialManifold): the trial manifold to reduce the FOM on.
            model (Model): the FOM.
            time_stepper (TimeStepper): the |TimeStepper| of the ROM.
        """
        assert isinstance(manifold, TrialManifold) and manifold.is_valid(), \
            'manifold has to be a valid TrialManifold'
        assert isinstance(model, Model)
        super().__init__('reduced_' + model.name, time_stepper)

        self.manifold = manifold
        self.model = model
        self.time_stepper.residual.register_model(self)

    def solve(self, t_0, t_end, dt, mu, logger=None, hook_fcns=[]):
        assert isinstance(hook_fcns, (list, tuple))
        if isinstance(hook_fcns, tuple):
            hook_fcns = list(hook_fcns)
        hook_fcns = hook_fcns.copy() # avoid manipulating original list
        reconstruct_hook = ReconstructionHookFunction(self.manifold)
        hook_fcns.append(reconstruct_hook)
        try:
            red_sol_x, red_sol_t = super().solve(t_0, t_end, dt, mu, logger=logger, hook_fcns=hook_fcns)
        except TimeStepperError as e:
            # add solution to error
            e.red_sol_x = reconstruct_hook.results()
            e.red_sol_t = reconstruct_hook.sol_t
            raise e
        return reconstruct_hook.results(), red_sol_x, red_sol_t

    def initial_value(self, mu):
        return self.manifold.inv_map(self.model.initial_value(mu))

    def check_mu(self, mu):
        return self.model.check_mu(mu)

    def get_dim(self):
        return self.manifold.dim


class PulledbackReducedModel(ReducedModel):
    def __init__(
        self,
        manifold: TrialManifold,
        model: Model,
    ) -> None:
        """Manifold MOR presented in Lee and Carlberg 2020.

        Args:
            manifold (TrialManifold): the reduced manifold to restrict the model to.
            model (Model): the model

        Raises:
            ValueError: if given model cannot be reduced, e.g. has no vector field or covector field.
        """
        self.pullback(manifold, model)
        super().__init__(manifold, model, model.time_stepper.copy())
    
    def pullback(self, manifold: TrialManifold, model: Model):
        if hasattr(model, 'vector_field') \
            and model.time_stepper.is_for_vector_field():

            raise RuntimeError('Use PulledbackVectorFieldReducedModel instead.')
        elif hasattr(model, 'covector_field') \
            and hasattr(model, 'metric') \
            and model.time_stepper.is_for_covector_field():

            self.covector_field = manifold.pullback(model.covector_field)
            self.metric = manifold.pullback(model.metric)
        else:
            raise ValueError('Given model cannot be reduced.')


class PulledbackVectorFieldReducedModel(PulledbackReducedModel):
    def __init__(self, manifold: TrialManifold, model: Model, projector: VectorFieldProjector) -> None:
        self.projector = projector
        super().__init__(manifold, model)

    def pullback(self, manifold: TrialManifold, model: Model):
        assert hasattr(model, 'vector_field') \
            and model.time_stepper.is_for_vector_field()
        self.vector_field = manifold.pullback_vector_field(model.vector_field, self.projector)


class SymplecticPulledbackReducedModel(ReducedModel):
    def __init__(
        self,
        manifold: TrialManifold,
        model: Model,
        use_canonical: bool = False,
    ) -> None:
        """Symplectic Manifold MOR presented in Buchfink, Glas and Haasdonk 2021.

        Args:
            manifold (TrialManifold): the reduced manifold to restrict the model to.
            model (Model): the model
            use_canonical (bool, optional): flag, wehter to use 
        """
        assert hasattr(model, 'vector_field') \
            and isinstance(model.time_stepper, HamiltonianModelTimeStepper) \
            and isinstance(model.vector_field, HamiltonianVectorField)

        if use_canonical:
            self.vector_field = CanonicalReducedHamiltonianVectorField(model.vector_field, manifold)
        else:
            self.vector_field = ReducedHamiltonianVectorField(model.vector_field, manifold)

        super().__init__(manifold, model, model.time_stepper.copy())


class LspgReducedTimStepperResidual(TimeStepperResidual):
    def __init__(self, manifold, time_stepper_residual):
        super().__init__()
        self.time_stepper_residual = time_stepper_residual
        self.manifold = manifold
        self._cached_full_x_old = (None,)*2
        self._cached_full_residual_data = (None,)*5

    def evaluate(self, x, x_old, dt):
        full_residual, phi = self._load_or_compute_full_residual_data(x, x_old, dt)
        return phi.T @ full_residual

    def evaluate_derivative(self, x, x_old, dt):
        _, phi = self._load_or_compute_full_residual_data(x, x_old, dt)
        # Quasi-Newton approximation of the Jacobian (neglect derivatives (a) of full_derivative and (b) of tan_map)
        return phi.T @ phi

    def _load_or_compute_full_x_old(self, x_old):
        if all(self._cached_full_x_old[0] == x_old):
            full_x_old = self._cached_full_x_old[1]
        else:
            full_x_old = self.manifold.map(x_old)
            self._cached_full_x_old = (x_old, full_x_old)
        return full_x_old

    def _load_or_compute_full_residual_data(self, x, x_old, dt):
        cached_x, cached_x_old, cached_dt = self._cached_full_residual_data[:3]
        if all(cached_x == x) and all(cached_x_old == x_old) and cached_dt == dt:
            return self._cached_full_residual_data[3:]
        else:
            full_x = self.manifold.map(x)
            full_x_old = self._load_or_compute_full_x_old(x_old)
            full_residual = self.time_stepper_residual.evaluate(full_x, full_x_old, dt)
            try:
                full_derivative = self.time_stepper_residual.evaluate_derivative(full_x, full_x_old, dt)
                # phi is the derivative of the full residual concatinated with 
                phi = full_derivative @ self.manifold.tangent_map(x)
            except NotImplementedError:
                tan_map = self.manifold.tangent_map(x)
                phi = np.zeros_like(tan_map)
                for i_tan_vec, tan_vec in enumerate(tan_map.T):
                    phi[:, i_tan_vec] = self.time_stepper_residual.apply_derivative_y(full_x, full_x_old, dt, tan_vec)
            self._cached_full_x_old = (x_old, full_x_old)
            self._cached_full_residual_data = (x, x_old, dt, full_residual, phi)
                
        return full_residual, phi

    def set_mu(self, mu):
        super().set_mu(mu)
        self.time_stepper_residual.set_mu(mu)
    
    def is_time_dependent(self):
        return self.time_stepper_residual.is_time_dependent()


class WeightedLspgReducedTimStepperResidual(LspgReducedTimStepperResidual):
    def __init__(self, manifold, time_stepper_residual):
        self.apply_weighting_matrix = None
        super().__init__(manifold, time_stepper_residual)

    def register_model(self, model):
        assert hasattr(model.model, 'apply_weighting_matrix') and callable(model.model.apply_weighting_matrix)
        self.apply_weighting_matrix = model.model.apply_weighting_matrix
        super().register_model(model)

    def evaluate(self, x, x_old, dt):
        full_residual, phi = self._load_or_compute_full_residual_data(x, x_old, dt)
        return phi.T @ self.apply_weighting_matrix(full_residual)

    def evaluate_derivative(self, x, x_old, dt):
        _, phi = self._load_or_compute_full_residual_data(x, x_old, dt)
        # Quasi-Newton approximation of the Jacobian (neglect derivatives (a) of full_derivative and (b) of tan_map)
        phi_weighted_prod = np.zeros((phi.shape[1],)*2)
        for i_phi_col, phi_col in enumerate(phi.T):
            phi_weighted_prod[:, i_phi_col] = phi.T @ self.apply_weighting_matrix(phi_col)
        return phi_weighted_prod


class LspgTimeStepper(TimeStepper):
    def is_for_vector_field(self):
        return isinstance(self.residual.time_stepper_residual, VectorFieldTimeStepperResidual)

    def is_for_covector_field(self):
        return isinstance(self.residual.time_stepper_residual, CovectorFieldTimeStepperResidual)


class LspgReducedModel(ReducedModel):
    def __init__(
        self,
        manifold: TrialManifold,
        model: Model,
        weighted: bool = False,
    ) -> None:
        """Least-Squares Petrov-Galerkin Manifold MOR presented in Lee and Carlberg 2020.

        Args:
            manifold (TrialManifold): the reduced manifold to restrict the model to.
            model (Model): the model
            weighted (bool, optional): whether to use |WeightedLspgReducedTimStepperResidual|.
                Defaults to False.

        Raises:
            ValueError: if given model cannot be reduced.
        """
        if hasattr(model, 'vector_field') \
            and model.time_stepper.is_for_vector_field():

            self.vector_field = model.vector_field
        elif hasattr(model, 'covector_field') \
            and hasattr(model, 'metric') \
            and model.time_stepper.is_for_covector_field():

            self.covector_field = model.covector_field
            self.metric = model.metric
        else:
            raise ValueError('Given model cannot be reduced.')

        non_linear_solver = model.time_stepper.non_linear_solver
        time_stepper_residual = model.time_stepper.residual.copy()
        time_stepper_residual.register_model(self)
        if weighted:
            lspg_residual = WeightedLspgReducedTimStepperResidual(manifold, time_stepper_residual)
        else:
            lspg_residual = LspgReducedTimStepperResidual(manifold, time_stepper_residual)
        time_stepper = LspgTimeStepper(lspg_residual, non_linear_solver)

        super().__init__(manifold, model, time_stepper)

    def is_for_vector_field(self):
        return self.model.is_for_vector_field()

    def is_for_covector_field(self):
        return self.model.is_for_covector_field()


class ReconstructionHookFunction(HookFunction):
    def __init__(self, manifold):
        self.manifold = manifold
        self.x_full = None
        self.sol_t = None

    def initialize(self, x_0: np.ndarray, t_0: float, dt: float, n_t: int, mu: dict):
        x_0_full = self.manifold.map(x_0)
        self.x_full = np.empty((n_t, len(x_0_full)))
        self.sol_t = np.empty((n_t,))

    def eval(self, x: np.ndarray, t: float, i_t: int):
        self.x_full[i_t, :] = self.manifold.map(x)
        self.sol_t[i_t] = t
    
    def finalize(self, all_x: np.ndarray, all_t: np.ndarray, failed: bool):
        if failed:
            n_t = len(all_t)
            self.x_full = self.x_full[:n_t]
            self.sol_t = self.sol_t[:n_t]

    def results(self):
        return self.x_full
