import os
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable

import torch
from manifold_mor.chart.auto_encoder import AutoEncoderChart
from manifold_mor.chart.basic import IterativelyTrainableCallback
from manifold_mor.chart.torch_losses import DictLoss
from manifold_mor.chart.torch_modules import TorchAutoEncoder
from manifold_mor.experiments.basic import ManifoldMorExperiment
from manifold_mor.experiments.trial_manifold import HasReductionExperiment, IntermediateTrialManifoldExperiment
from manifold_mor.utils.logging import Logger
from structured_nn.utils.loss import Loss

if TYPE_CHECKING:
    from manifold_mor.trial_manifold.basic import TrainableTrialManifold


class FrequentIterativelyTrainableCallback(IterativelyTrainableCallback):
    def __init__(self, frequency: int) -> None:
        super().__init__()
        self.frequency = frequency

    @abstractmethod
    def frequent_call(self, essentials: dict, summary: dict, logger: Logger = None):
        ...

    def train_step(self, essentials: dict, summary: dict, logger: Logger = None):
        if summary['epoch'] % self.frequency == 0:
            self.frequent_call(essentials, summary, logger)


class MorAndProjectionFrequentIterativelyTrainableCallback(FrequentIterativelyTrainableCallback):
    """Run |ModelReductionExperiment|."""
    def __init__(
        self,
        frequency: int,
        manifold: 'TrainableTrialManifold',
        torch_ae: TorchAutoEncoder,
        reductions: Iterable[str],
        logger_mor: Logger,
        parent_experiment: ManifoldMorExperiment,
    ) -> None:
        assert isinstance(parent_experiment, HasReductionExperiment)
        super().__init__(frequency)
        self.manifold = manifold
        self.torch_ae = torch_ae
        self.reductions = reductions
        self.logger_mor = logger_mor
        self.parent_experiment = parent_experiment

    def frequent_call(self, essentials: dict, summary: dict, logger: Logger = None):
        torch_ae = self.torch_ae
        # ensure eval mode
        if torch_ae.training:
            warnings.warn('Set torch_ae to non-training')
            torch_ae.eval()

        # work arround to get a TrialManifoldExperiment that hosts the current manifold
        manifold_experiment = IntermediateTrialManifoldExperiment(
            self.parent_experiment,
            self.manifold,
        )

        summary['eval_manifold_epoch'] = self.manifold.get_epoch()

        for reduction in self.reductions:
            mor_exp = self.parent_experiment.get_reduction_experiment(
                reduction, essentials, summary, manifold_experiment=manifold_experiment
            )
            mor_exp['epoch'] = summary['epoch']
            mor_exp.multi_log('mor', self.logger_mor)
            mor_exp.run()
            self.parent_experiment.link_experiment(mor_exp)


class AdditionalLossesFrequentIterativelyTrainableCallback(FrequentIterativelyTrainableCallback):
    def __init__(
        self,
        frequency: int,
        torch_ae: TorchAutoEncoder,
        eval_loss_func: Loss,
        chart: AutoEncoderChart,
    ) -> None:
        assert isinstance(eval_loss_func, DictLoss)
        super().__init__(frequency)
        self.torch_ae = torch_ae
        self.eval_loss_func = eval_loss_func
        self.chart = chart
        self.eval_loss_func_required_keys = eval_loss_func.required_keys()

    def frequent_call(self, essentials: dict, summary: dict, logger: Logger = None):
        # compute losses
        torch_ae = self.torch_ae
        val_loader = essentials['val_loader']
        if torch_ae.training:
            warnings.warn('Set torch_ae to non-training')
            torch_ae.eval()

        with torch.no_grad():
            for val_batch in val_loader:
                ae_val_batch = self.chart._compute_ae_batch(
                    val_batch,
                    self.eval_loss_func_required_keys,
                    essentials.get('mus', None),
                    essentials.get('torch_model', None),
                )
                self.eval_loss_func(ae_val_batch, val_batch)
            self.eval_loss_func.avg_loss_over_batches()
        
        summary.update(self.eval_loss_func.summary_dict())
