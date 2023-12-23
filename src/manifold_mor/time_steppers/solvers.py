"""Newton's method to solve systems of nonlinear equations."""
from abc import ABC, abstractmethod

import numpy as np

class SolverError(Exception):
    ...


class NonlinearSolver(ABC):
    @abstractmethod
    def solve(self, function, jacobian, verbose=False, **kwargs):
        pass


class NewtonNonlinearSolver(NonlinearSolver):
    def __init__(self, abs_tol=1e-8, rel_tol=1e-6, max_iter=1e2, minimal_iniital_norm_fun_eval=1e-6, norm=np.linalg.norm):
        super().__init__()
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self.minimal_iniital_norm_fun_eval = minimal_iniital_norm_fun_eval
        self.norm = norm

    def solve(self, function, apply_inv_jacobian, x, verbose=False, **kwargs):
        fun_eval = function(x, **kwargs)
        norm_fun_eval = self.norm(fun_eval)
        initial_norm_eval = max(norm_fun_eval, self.minimal_iniital_norm_fun_eval)
        it = 0
        while norm_fun_eval > self.abs_tol \
            and norm_fun_eval / initial_norm_eval > self.rel_tol \
            and it < self.max_iter:
            
            x = x - apply_inv_jacobian(x, fun_eval, **kwargs)
            fun_eval = function(x, **kwargs)
            norm_fun_eval = self.norm(fun_eval)
            it += 1

        if norm_fun_eval <= self.abs_tol:
            message = 'Absolute tolerance ({}) met after {} iterations.'.format(self.abs_tol, it)
        elif norm_fun_eval / initial_norm_eval <= self.rel_tol:
            message = 'Relative tolerance ({}) met after {} iterations.'.format(self.rel_tol, it)
        elif np.isinf(norm_fun_eval):
            raise SolverError('Norm of residual is inf after {} iterations.'.format(it))
        else:
            raise SolverError('Maximal number of iterations ({}) exceeded with function norm {}.'.format(self.max_iter, norm_fun_eval))
        if verbose:
            print(message)

        return x, norm_fun_eval, it
