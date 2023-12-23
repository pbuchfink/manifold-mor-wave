"""Modules to linearize a time stepper residual
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla

if TYPE_CHECKING:
    from manifold_mor.time_steppers.residual import TimeStepperResidual


class Linearization(ABC):
    def __init__(self, residual: 'TimeStepperResidual') -> None:
        self.residual = residual
    
    @abstractmethod
    def apply_derivative(
        self,
        x: np.ndarray,
        x_old: np.ndarray,
        t_old: float,
        dt: float,
        y: np.ndarray,
    ) -> np.ndarray:
        """Apply linearization.

            res = Df|_{x} * y

        Args:
            x (np.ndarray): point where linearization is computed
            x_old (np.ndarray): solution at previous time step
            t_old (float): previous time step
            dt (float): time step
            y (np.ndarray): vector to apply derivative to

        Returns:
            np.ndarray: result vector res
        """

    @abstractmethod
    def apply_inv_derivative(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_old: np.ndarray,
        t_old: float,
        dt: float,
    ) -> np.ndarray:
        """Apply inverse linearization.

            Df|_{x} * res = y

        Args:
            x (np.ndarray): point where linearization is computed
            x_old (np.ndarray): solution at previous time step
            t_old (float): previous time step
            dt (float): time step
            y (np.ndarray): vector to apply derivative to

        Returns:
            np.ndarray: result vector res
        """
        ...


class DefaultLinearization(Linearization):
    """Linearization which uses eval derivative of the residual.
    """
    def apply_derivative(
        self,
        x: np.ndarray,
        x_old: np.ndarray,
        t_old: float,
        dt: float,
        y: np.ndarray,
    ) -> np.ndarray:
        jac = self.residual.evaluate_derivative(x, x_old, t_old, dt)
        return jac @ x

    def apply_inv_derivative(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_old: np.ndarray,
        t_old: float,
        dt: float,
    ) -> np.ndarray:
        jac = self.residual.evaluate_derivative(x, x_old, t_old, dt)
        if sparse.issparse(jac):
            return spla.spsolve(jac, y)
        else:
            return la.solve(jac, y)