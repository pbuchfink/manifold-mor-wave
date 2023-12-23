from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable

from manifold_mor.chart.basic import IterativelyTrainableCallback, IterativelyTrainableChart
from manifold_mor.chart.torch_modules import TorchAutoEncoder
from manifold_mor.experiments.basic import ManifoldMorExperiment
from manifold_mor.experiments.caching import (cached_result,
                                              exists_cached_result)
from manifold_mor.experiments.model import ModelExperiment
from manifold_mor.mor.snapshots import Snapshots
from manifold_mor.trial_manifold.basic import (SingleChartTrainableTrialManifold, TrainableTrialManifold,
                                               TrialManifold)
from manifold_mor.trial_manifold.component import ComponentwiseTrainableManifold
from structured_nn.utils.loss import Loss

if TYPE_CHECKING:
    from manifold_mor.experiments.model_reduction import \
        ModelReductionExperiment


class TrialManifoldExperiment(ManifoldMorExperiment):
    '''A |ManifoldMorExperiment| specifying an experiment with a |TrialManifold|.
    This is based on a |ModelExperiment| which specifies the corresponding full-order model.
    '''
    def __init__(self, **parameters) -> None:
        super().__init__(**parameters)
        self._cached_snapshots = None

    # implemented as property, since often used
    @property
    def model_experiment(self) -> ModelExperiment:
        return self._parameters['model_experiment']
    
    def get_experiment_type(self) -> str:
        return 'trial_manifold'
    
    def _get_required_param_keys(self) -> set:
        required_keys = super()._get_required_param_keys()
        required_keys.update(['model_experiment'])
        return required_keys

    @abstractmethod
    def get_untrained_manifold(self) -> TrialManifold:
        """Returns the (untrained) manifold.
        Untrained is in braces since this does not apply for |TrialManifold|s that are NOT
        |TrainableTrialManifold|s.

        Returns:
            TrialManifold: the (untrained) manifold.
        """
        ...
    
    def get_callbacks(self, manifold: TrialManifold) -> Iterable[IterativelyTrainableCallback]:
        """Get all |IterativelyTrainableCallback| to evaluate in training.

        Returns:
            Iterable[IterativelyTrainableCallback]: callbacks to evaluate in training
        """
        return None
    
    def get_essentials(self, manifold: TrialManifold) -> dict:
        """Get essentials forwarded to the training function. E.g. training and validation loss can
        be implemented here.

        Args:
            manifold (TrialManifold): The manifold to be trained.

        Returns:
            dict: the essentials forwarded to the train method.
        """
        return dict()
    
    def get_loss_evaluation(self, torch_ae: TorchAutoEncoder) -> Loss:
        return None

    def get_training_snapshots(self, use_rhs_if_available: bool = False) -> Snapshots:
        assert self.model_experiment['mu_scenario'] == ModelExperiment.MU_TRAINING
        if self._cached_snapshots is None:
            snapshots = self.model_experiment.get_snapshots()
            snapshots.make_dataset(use_rhs_if_available=use_rhs_if_available)
            self._cached_snapshots = snapshots
        return self._cached_snapshots

    @cached_result
    def _compute_manifold(self) -> TrialManifold:
        """Returns manifold. This may include training, if the underlying manifold is a
        |TrainableTrialManifold|.

        Returns:
            TrialManifold: the (trained) manifold.
        """
        manifold = self.get_untrained_manifold()
        if isinstance(manifold, TrainableTrialManifold):
            if isinstance(manifold, SingleChartTrainableTrialManifold):
                chart = manifold.chart
            elif isinstance(manifold, ComponentwiseTrainableManifold):
                assert isinstance(manifold.trainable_manifold, SingleChartTrainableTrialManifold)
                chart = manifold.trainable_manifold.chart
            else:
                raise NotImplementedError()
            if isinstance(chart, IterativelyTrainableChart):
                kwargs = dict(
                    callbacks = self.get_callbacks(manifold),
                    essentials = self.get_essentials(manifold),
                )
            else:
                kwargs = {}
            manifold.train(
                self.get_training_snapshots(),
                self._parameters,
                self.get_logger('train'),
                self.model_experiment.get_model(),
                **kwargs,
            )
        return manifold

    def get_manifold(self) -> TrialManifold:
        if not self.cached_results_exist():
            raise RuntimeError('Run experiment before using get_manifold.')
        return self._compute_manifold()

    def cached_results_exist(self, **technical_params) -> bool:
        return exists_cached_result(self, self._compute_manifold)

    def _prepare(self, **technical_params) -> None:
        super()._prepare(**technical_params)
        self._init_logger('train')
        assert self.model_experiment['mu_scenario'] == ModelExperiment.MU_TRAINING, \
            'model_experiment should contain training data'

    def _execute(self, **technical_params) -> None:
        # get (trained) manifold
        self._compute_manifold()


class HasReductionExperiment(ABC):
    @abstractmethod
    def get_reduction_experiment(
        self,
        reduction: str,
        essentials: dict,
        summary: dict,
        **kwargs
    ) -> 'ModelReductionExperiment':
        ...


class IntermediateTrialManifoldExperiment(TrialManifoldExperiment):
    """A class for intermediate manifolds. This is used in intermediate MOR experiments during
    training.
    """
    def __init__(
        self,
        parent_experiment: TrialManifoldExperiment,
        parent_manifold: TrainableTrialManifold = None,
    ) -> None:
        super().__init__(
            parent_manifold=parent_manifold,
            parent_experiment_name=parent_experiment.get_experiment_name(),
            model_experiment=parent_experiment.model_experiment,
        )
        # copy modified attributes
        for key in parent_experiment.modified_parameter_keys:
            self[key] = parent_experiment[key]
        if parent_manifold is not None:
            self['epoch'] = parent_manifold.get_epoch()

    @classmethod
    def from_cache(cls, parent_experiment: TrialManifoldExperiment, epoch: int):
        obj = cls(parent_experiment, None)
        obj['epoch'] = epoch
        assert exists_cached_result(obj, obj.get_manifold)
        return obj

    @property
    def parent_manifold(self):
        parent_manifold = self['parent_manifold']
        assert parent_manifold.get_epoch() == self.epoch, \
            'manifold was not copied and advanced in training. The experiment is no longer valid.'
        return self['parent_manifold']

    @property
    def epoch(self):
        return self['epoch']

    @cached_result
    def get_manifold(self) -> 'TrialManifold':
        return self.parent_manifold

    def get_experiment_name(self) -> str:
        return self['parent_experiment_name']

    def get_untrained_manifold(self) -> 'TrialManifold':
        raise NotImplementedError()
