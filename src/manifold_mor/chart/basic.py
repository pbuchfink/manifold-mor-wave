"""All classes for Charts.
"""
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from manifold_mor.models.basic import Model
from manifold_mor.mor.snapshots import Snapshots
from manifold_mor.utils.logging import Logger
from ray.tune.result import DONE, SHOULD_CHECKPOINT

if TYPE_CHECKING:
    from manifold_mor.fields.vector_reduced import VectorFieldProjector


class Chart(ABC):
    def __init__(self):
        self.manifold = None
        self._cached_x = (None,) * 2
        self._cached_jac = (None,) * 2
        # is set during the setup of the arichitecture or training
        self.ambient_dim = None
        self.dim = None

    def register_manifold(self, manifold):
        self.manifold = manifold

    @abstractmethod
    def map(self, xr):
        ...

    @abstractmethod
    def inv_map(self, x):
        ...

    def project_x(self, x):
        return self.map(self.inv_map(x))

    @abstractmethod
    def tangent_map(self, xr):
        ...
    
    def is_valid_projector(self, projector: 'VectorFieldProjector'):
        # accept all projectors
        return True

    def __str__(self):
        return self.__class__.__name__

    def _save_cached_x(self, xr, x):
        self._cached_x = (xr, x)

    def _load_cached_x(self, xr):
        if np.all(self._cached_x[0] == xr) and self._cached_x[0].shape == xr.shape:
            return self._cached_x[1]
        else:
            return None

    def _save_cached_jac(self, xr, jac):
        self._cached_jac = (xr, jac)

    def _load_cached_jac(self, xr):
        if np.all(self._cached_jac[0] == xr):
            return self._cached_jac[1]
        else:
            return None

    # def jacobian(self, xr):
    #     dim = self.manifold.dim
    #     jac = np.zeros((self.manifold.ambient_dim, dim))
    #     unit_vec_i = np.zeros(self.manifold.ambient_dim)
    #     for i_row in range(self.manifold.ambient_dim):
    #         unit_vec_i[i_row-1] = 0.
    #         unit_vec_i[i_row] = 1.
    #         _, v = self.apply_transposed_jacobian(xr, unit_vec_i)
    #         jac[i_row, :] = v

    #     return jac

    def hessian(self, xr):
        '''Computes the derivative of the Jacoian at xr.
        Returns a 3rd-order tensor.'''
        raise NotImplementedError()

    def is_valid(self):
        return self.manifold is not None \
            and self.ambient_dim is not None \
            and self.manifold.ambient_dim == self.ambient_dim \
            and self.dim is not None \
            and self.manifold.dim == self.dim


class TorchEvaluatable(ABC):
    """A mixin for |Chart|s that are evaluatable in torch."""
    @abstractmethod
    def map_torch(self, xr: torch.Tensor):
        ...

    @abstractmethod
    def inv_map_torch(self, x: torch.Tensor):
        ...

    def project_x_torch(self, x: torch.Tensor):
        return self.map_torch(self.inv_map_torch(x))


class TrainableChart(Chart):
    def __init__(self, is_trained=False):
        super().__init__()
        self.is_trained = is_trained

    @abstractmethod
    def train(
        self,
        snapshots: Snapshots,
        training_params: dict,
        logger: Logger = None,
        model: Model = None
    ) -> dict:
        ...

    @abstractmethod
    def number_of_parameters_in_decoder(self):
        ...

    @abstractmethod
    def number_of_parameters_in_total(self):
        ...

    def is_valid(self):
        return super().is_valid() and self.is_trained


class IterativelyTrainableCallback:
    def train_setup(self, essentials: dict, logger: Logger = None):
        ...

    def train_step(self, essentials: dict, summary: dict, logger: Logger = None):
        ...
    
    def train_finished(self, essentials: dict, summary: dict, logger: Logger = None):
        ...


class ReportIterativelyTrainingCallback(IterativelyTrainableCallback):
    def train_step(self, essentials: dict, summary: dict, logger: Logger = None):
        logger.report_training_callback(**summary, summary=summary)
    
    def train_finished(self, essentials: dict, summary: dict, logger: Logger = None):
        logger.report_training_callback(**summary, summary=summary)


class IterativelyTrainableChart(TrainableChart):
    # summary keys chosen to match ray analogons
    SUMMARY_KEY_SHOULD_STOP = DONE
    SUMMARY_KEY_SHOULD_CHECKPOINT = SHOULD_CHECKPOINT
    def __init__(self, is_trained: bool, verbose: bool = True):
        super().__init__(is_trained=is_trained)
        self.verbose = verbose
        self._epoch = None

    def train(
        self,
        snapshots: Snapshots,
        training_params: dict,
        logger: Optional[Logger] = None,
        model: Optional[Model] = None,
        callbacks: Optional[List[IterativelyTrainableCallback]] = None,
        essentials: Optional[dict] = None,
    ) -> dict:
        if callbacks is None:
            callbacks = []
        if essentials is None:
            essentials = {}
        # avoid that modifying these has consequences in parent methods
        callbacks = callbacks.copy()
        essentials = essentials.copy()

        essentials = self.train_setup(snapshots, training_params, logger, model, callbacks, essentials)

        for callback in callbacks:
            callback.train_setup(essentials, logger)

        epoch = training_params.get('initial_iter', 0)
        while epoch < training_params['n_epochs']:
            epoch += 1 #in analogy to tune (increment epoch right after step before creating checkpoints)
            self._epoch = epoch
            summary = {'epoch': epoch}
            self.train_step(essentials, summary, logger)

            for callback in callbacks:
                callback.train_step(essentials, summary, logger)
        
            # print information
            if self.verbose:
                status = 'Epoch: {}\n'.format(epoch)
                for key, val in summary.items():
                    status += '\t{}: {}\n'.format(key, val)
                print(status)
            
            if summary.get(IterativelyTrainableChart.SUMMARY_KEY_SHOULD_STOP, False):
                break
        
        self.train_finished(essentials, summary, logger)
        
        for callback in callbacks:
            callback.train_finished(essentials, summary, logger)
        
        return summary

    def get_epoch(self) -> int:
        return self._epoch

    @abstractmethod
    def train_setup(
        self,
        snapshots: Snapshots,
        training_params: dict,
        logger: Logger,
        model: Model,
        callbacks: List[IterativelyTrainableCallback],
        essentials: dict,
    ) -> dict:
        ...

    @abstractmethod
    def train_step(
        self,
        essentials: dict,
        summary: dict,
        logger: Logger = None,
    ):
        ...
    
    @abstractmethod
    def train_finished(
        self,
        essentials: dict,
        summary: dict,
        logger: Logger = None,
    ):
        ...
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        essentials: dict,
        summary: dict,
        checkpoint_name_prefix: str = None,
    ):
        raise NotImplementedError()
    
    def load_checkpoint(
        self,
        checkpoint_dir: str,
        essentials: dict = None,
        summary: dict = None,
        checkpoint_name_prefix: str = None,
    ):
        raise NotImplementedError()


class CheckpointIterativelyTrainingCallback(IterativelyTrainableCallback):
    def __init__(
        self,
        chart: IterativelyTrainableChart,
        checkpoint_dir: str,
        threshold_epoch_checkpoint: int = 0,
        checkpoint_freq: int = None,
        checkpoint_at_end: bool = False,
        load_checkpoint_dir: str = None,
    ) -> None:
        self.chart = chart
        self.checkpoint_dir = checkpoint_dir
        self.threshold_epoch_checkpoint = threshold_epoch_checkpoint
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_at_end = checkpoint_at_end
        self.load_checkpoint_dir = load_checkpoint_dir

    @classmethod
    def from_config(cls, chart, config):
        cls(
            chart,
            checkpoint_dir=config.get('checkpoint_dir'),
            threshold_epoch_checkpoint=config.get('threshold_epoch_checkpoint', 0),
            checkpoint_freq=config.get('checkpoint_freq', None),
            checkpoint_at_end=config.get('checkpoint_at_end', False),
        )

    def save_checkpoint(self, essentials: dict, summary: dict):
        checkpoint_dir = os.path.join(self.checkpoint_dir, "checkpoint_{}".format(summary['epoch']))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.chart.save_checkpoint(checkpoint_dir, essentials, summary)

    def train_setup(self, essentials: dict, logger: Logger = None):
        '''Load checkpoint dir if requried.'''
        if not self.load_checkpoint_dir is None:
            self.chart.load_checkpoint(self.load_checkpoint_dir, essentials=essentials)

    def train_step(self, essentials: dict, summary: dict, logger: Logger = None):
        # do not checkpoint, if threshold is set
        if summary['epoch'] < self.threshold_epoch_checkpoint:
            return 
        if (self.checkpoint_freq is not None and summary['epoch'] % self.checkpoint_freq == 0) \
            or summary.get(IterativelyTrainableChart.SUMMARY_KEY_SHOULD_CHECKPOINT, False):

            self.save_checkpoint(essentials, summary)
    
    def train_finished(self, essentials: dict, summary: dict, logger: Logger = None):
        if self.checkpoint_at_end:
            self.save_checkpoint(essentials, summary)
