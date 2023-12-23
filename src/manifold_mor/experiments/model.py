import os
from abc import abstractmethod
from typing import List, Tuple

from manifold_mor.experiments.basic import ManifoldMorExperiment
from manifold_mor.experiments.caching import (cached_result,
                                              exists_cached_result)
from manifold_mor.models.basic import Model
from manifold_mor.mor.snapshots import SnapshotGenerator, Snapshots


class ModelExperiment(ManifoldMorExperiment):
    '''A |ManifoldMorExperiment| specifying a model.
    This approach allows to store / cache snapshots of a model
    (since the parameters are fixed).'''
    MU_TRAINING = 'training'
    MU_TEST_REPRODUCTION = 'training' # share same snapshots with training
    MU_TEST_GENERALIZATION = 'generalization'

    def __init__(
        self,
        mu_scenario: str,
        compute_rhs: bool = True, #parameter with default value
        **parameters
    ) -> None:
        parameters['compute_rhs'] = compute_rhs
        super().__init__(**parameters)
        # automatically add to modified parameters
        self['mu_scenario'] = mu_scenario
    
    def get_experiment_type(self) -> str:
        return 'fom'

    def _get_required_param_keys(self) -> set:
        required_keys = super()._get_required_param_keys()
        required_keys.update(['t_0', 'dt', 't_end', 'shifted'])
        return required_keys

    def get_temporal_data(self) -> Tuple[float, float, float]:
        return self['t_0'], self['t_end'], self['dt']

    @abstractmethod
    def get_model(self) -> Model:
        ...

    @abstractmethod
    def get_mus(self) -> List[dict]:
        '''Get list of parameters. Should depend on self['mu_scenario'].'''
        ...
    
    #TODO: use potential hook functions in snapshot generation
    def get_hook_fcns(self):
        return []

    @cached_result
    def _compute_snapshots(self) -> Snapshots:
        '''Get snapshots for the specific |ModelExperiment|.
        If callbacks are required, overwrite this method.
        '''
        #TODO: SnapshotGenerator architecture might be removed since experiments now allow for caching results
        return SnapshotGenerator.generate_from_model_experiment(self)

    def get_snapshots(self) -> Snapshots:
        if not self.cached_results_exist():
            raise RuntimeError('Run experiment before using get_snapshots.')
        return self._compute_snapshots()
    
    def cached_results_exist(self, **technical_params) -> bool:
        return exists_cached_result(self, self._compute_snapshots)
    
    def _prepare(self, **technical_params) -> None:
        super()._prepare(**technical_params)
        self._init_logger('fom')

    def _execute(self, **technical_params) -> None:
        self._compute_snapshots()


class HasTorchModel():
    @abstractmethod
    def get_torch_model(self) -> Model:
        ...
