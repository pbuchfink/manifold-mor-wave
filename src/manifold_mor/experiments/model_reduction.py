
from abc import abstractmethod
from typing import List

from manifold_mor.experiments.basic import ManifoldMorExperiment
from manifold_mor.experiments.caching import (cached_result,
                                              exists_cached_result)
from manifold_mor.experiments.model import ModelExperiment
from manifold_mor.experiments.trial_manifold import TrialManifoldExperiment
from manifold_mor.mor.mor_pipeline import MorPipeline, RomGenerator
from manifold_mor.time_steppers.basic import HookFunction
from manifold_mor.trial_manifold.basic import TrialManifold


class ModelReductionExperiment(ManifoldMorExperiment):
    REDUCTION_LSPG = 'lspg'
    REDUCTION_PULLBACK = 'pullback'
    REDUCTION_LSPG_WEIGHTED = 'lspg_weighted'
    REDUCTION_PULLBACK_WEIGHTED = 'pullback_weighted'
    REDUCTION_SYMPL_CANONICAL = 'sympl_canonical'
    REDUCTION_SYMPL_PULLBACK = 'sympl_pullback'
    def __init__(
        self,
        model_experiment: ModelExperiment,
        reduction: str,
        **parameters
    ) -> None:
        super().__init__(
            model_experiment=model_experiment,
            reduction=reduction,
            **parameters
        )
        self.modified_parameter_keys.add('reduction')

    # implemented as property, since often used
    @property
    def model_experiment(self) -> ModelExperiment:
        return self._parameters['model_experiment']

    def _get_required_param_keys(self) -> set:
        required_keys = super()._get_required_param_keys()
        required_keys.update(['model_experiment', 'reduction'])
        return required_keys

    def get_experiment_type(self) -> str:
        return 'mor'

    @abstractmethod
    def get_manifold(self) -> TrialManifold:
        ...

    def get_rom_generator(self):
        return RomGenerator()

    def get_hook_fcns(self) -> List[HookFunction]:
        """Returns list of |HookFunction|s considered in this experiment.
        Default: Hook functions from model_experiment (i.e. FOM) are inherited.

        Returns:
            List[HookFunction]: list of |HookFunction|s to be considered in this experiment.
        """
        return self.model_experiment.get_hook_fcns()

    @cached_result
    def _run_mor_pipeline(self, mor_pipeline: MorPipeline):
        manifold = mor_pipeline.manifold
        config = {
            'manifold_name': manifold.name,
            'red_dim': manifold.dim,
        }
        for key in self.modified_parameter_keys:
            config[key] = self[key]
        #TODO: MorPipeline might be replaced by ModelReductionExperiment?
        mor_pipeline.run(
            run_name = '_'.join([manifold.name, self['reduction']]), config = config
        )
        return mor_pipeline.results()

    def cached_results_exist(self, **technical_params) -> bool:
        return exists_cached_result(self, self._run_mor_pipeline)

    def get_mor_pipeline_results(self):
        if not self.cached_results_exist():
            raise RuntimeError('Run experiment before using get_mor_pipeline_results.')
        return self._run_mor_pipeline()

    def get_mor_pipeline_modules(self):
        return []

    def _prepare(self, **technical_params) -> None:
        super()._prepare(**technical_params)
        self._init_logger('mor')
    
    def _execute(self, **technical_params) -> None:
        verbose = technical_params.get('verbose', False)

        mor_pipeline_params = dict(
            model = self.model_experiment.get_model(),
            snapshots = self.model_experiment.get_snapshots(),
            manifold = self.get_manifold(),
            rom_generator = self.get_rom_generator(),
            logger = self.get_logger('mor'),
            modules = self.get_mor_pipeline_modules(),
            hook_fcns = self.get_hook_fcns(),
            verbose = verbose,
        )
        mor_pipeline = MorPipeline(**mor_pipeline_params)

        self._run_mor_pipeline(mor_pipeline)


class TrialManifoldExperimentBased():
    @property
    def manifold_experiment(self) -> TrialManifoldExperiment:
        return self['manifold_experiment']
    
    def get_manifold(self) -> TrialManifold:
        return self.manifold_experiment.get_manifold()
