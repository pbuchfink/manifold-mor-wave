"""Basic class for time steppers (i.e. integrators).
The formulation is based on a residual formulation which allows to straight-forwardly apply Newton's method to solve non-linear systems.
"""
import copy
from abc import ABC, abstractmethod
from manifold_mor.context import get_current_context

import numpy as np
from manifold_mor.time_steppers.covector import \
    CovectorFieldTimeStepperResidual
from manifold_mor.time_steppers.residual import TimeStepperResidual
from manifold_mor.time_steppers.solvers import (NewtonNonlinearSolver,
                                                NonlinearSolver)
from manifold_mor.time_steppers.vector import VectorFieldTimeStepperResidual


class TimeStepperError(ValueError):
    def __init__(self, time_step, *args):
        super(TimeStepperError, self).__init__(*args)
        self.time_step = time_step


class TimeStepper(object):
    def __init__(
        self,
        time_stepper_residual: TimeStepperResidual,
        non_linear_solver: NonlinearSolver = NewtonNonlinearSolver()
    ):
        assert isinstance(time_stepper_residual, TimeStepperResidual)
        assert isinstance(non_linear_solver, NonlinearSolver)

        self.residual = time_stepper_residual
        self.non_linear_solver = non_linear_solver
        self.print_progress_frequency = 0.1 #report every 10%

    def solve(self, t_0, t_end, dt, mu, hook_fcns=None, callbacks=None):
        assert self.residual.is_valid()

        # deepcopy mu to avoid altering given mu
        mu = copy.deepcopy(mu)

        # add time step to mu
        if self.residual.is_time_dependent():
            mu['_t'] = t_0

        self.residual.set_mu(mu)

        t = np.arange(start=t_0, stop=t_end, step=dt)
        n_t = len(t)
        x_0 = self.residual.initial_value(mu)
        if isinstance(x_0, np.ndarray):
            # saving in an numpy array is essential for reduced simulations with SympNets
            # saving in a list causes some kind of memory leak
            x = np.empty((n_t, x_0.size))
        else:
            x = [None] * n_t
        x[0] = x_0

        hook_fcns = self._to_hook_fcn_list(hook_fcns)
        for hook_fcn in hook_fcns:
            hook_fcn.initialize(x[0], t_0, dt, n_t, mu)

        # decide whether to use apply_inv_derivative or evaluate_derivative
        if hasattr(self.residual, 'apply_inv_derivative'):
            apply_inv_jacobian = self.residual.apply_inv_derivative
        else:
            apply_inv_jacobian = lambda y, b, **kwargs: np.linalg.solve(self.residual.evaluate_derivative(y, **kwargs), b)
        
        i_t = 0
        try:
            for hook_fcn in hook_fcns:
                hook_fcn.eval(x[0], t[0], 0)

            for i_t in range(0, n_t-1):
                if self.residual.is_time_dependent():
                    mu['_t'] = t[i_t]
                    self.residual.set_mu(mu)
                verbose = self.print_progress(i_t, n_t)
                x[i_t+1], res, n_iter = self.non_linear_solver.solve(
                    self.residual.evaluate,
                    apply_inv_jacobian,
                    x[i_t],
                    x_old=x[i_t],
                    dt=dt,
                    verbose=verbose
                )

                for hook_fcn in hook_fcns:
                    hook_fcn.eval(x[i_t+1], t[i_t+1], i_t+1)

                # TODO: unite callbacks and hook functions?
                # TODO: print_progress as callback?
                if callbacks:
                    callback_kwargs = {'residual': res, 'solver_iterations': n_iter, 'time_step': i_t}
                    for callback in callbacks:
                        callback(**callback_kwargs)

            for hook_fcn in hook_fcns:
                hook_fcn.finalize(x, t, failed=False)

        except Exception as e:
            if get_current_context().options['raise_errors_during_timestepping']:
                # reraise exact error
                raise e

            failed_message = str(e)

            # truncate solution
            t = t[:i_t]
            x = x[:i_t]

            # finalize hook functions with failed
            for hook_fcn in hook_fcns:
                hook_fcn.finalize(x, t, failed=True)
            raise TimeStepperError(
                i_t,
                'Error in Time step {}: {}'.format(i_t, failed_message)
            ) from e

        self.print_progress(n_t, n_t)

        return x, t

    def print_progress(self, current, total, iter_info=''):
        '''Prints progress and given iter_info in intervals specified by self.print_progress_frequency.
        It should hold
            0 <= last <= current <= total
        Parameters
        ----------
        current
            current iteration
        total
            total iterations
        iter_info
            additional information to be printed.'''
        idx_last = int(((current-1)/total)/self.print_progress_frequency)
        idx_current = int((current/total)/self.print_progress_frequency)
        if idx_last != idx_current:
            print(str(self.__class__.__name__) + ': %3d%%' % (current/total*100) + iter_info)
            return True
        return False

    def _to_hook_fcn_list(self, hook_fcns):
        if hook_fcns is None:
            return tuple()
        if isinstance(hook_fcns, HookFunction):
            return (hook_fcns,)
        if isinstance(hook_fcns, (tuple, list)) \
            and all(isinstance(hook_fcn, HookFunction) for hook_fcn in hook_fcns):
            return hook_fcns
        raise ValueError('hook_fcns has to be None, HookFunction or tuple/list of HookFunction.')

    def is_for_vector_field(self):
        return isinstance(self.residual, VectorFieldTimeStepperResidual)

    def is_for_covector_field(self):
        return isinstance(self.residual, CovectorFieldTimeStepperResidual)

    def copy(self):
        residual = self.residual.copy()
        return type(self)(residual, self.non_linear_solver)

class HookFunction(ABC):
    @abstractmethod
    def initialize(self, x_0: np.ndarray, t_0: float, dt: float, n_t: int, mu: dict):
        ...

    @abstractmethod
    def eval(self, x: np.ndarray, t: float, i_t: int):
        ...

    @abstractmethod
    def results(self):
        ...

    def finalize(self, all_x: np.ndarray, all_t: np.ndarray, failed: bool = False):
        pass
