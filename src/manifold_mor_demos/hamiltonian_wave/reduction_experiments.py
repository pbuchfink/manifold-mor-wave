
from manifold_mor_demos.hamiltonian_wave.manifold_experiments import \
    BumpManifoldExperiment
from manifold_mor_demos.hamiltonian_wave.model_experiments import BumpModelExperiment
from manifold_mor.experiments.model_reduction import (
    ModelReductionExperiment, TrialManifoldExperimentBased)
from manifold_mor.mor.metrics import RelEuclideanMorErrorMetric
from manifold_mor.mor.mor_pipeline import RomGenerator
from manifold_mor.mor.mor_pipeline_ingredients import (
    HamiltonianModule, ProjectionMorPipelineModule, ProjectionMorPipelineModuleFomTimeSampler, SymplecticityExperimentModule)


class BumpModelReductionExperiment(TrialManifoldExperimentBased,ModelReductionExperiment):
    def __init__(
        self,
        manifold_experiment: BumpManifoldExperiment,
        reduction: str,
        debug: bool = False,
    ) -> None:
        model_experiment = BumpModelExperiment(
            mu_scenario=BumpModelExperiment.MU_TEST_GENERALIZATION,
            shifted=manifold_experiment.model_experiment['shifted'],
            debug=debug,
        )
        super().__init__(
            model_experiment=model_experiment,
            num_subsample=250,
            time_sampler_mode='fom',
            sqrt_in_metric=True,
            reduction=reduction,
        )
        self['manifold_experiment'] = manifold_experiment
        self['reduction'] = reduction
        if debug:
            self.modified_parameter_keys.add('model_experiment')

    def get_experiment_name(self) -> str:
        return 'bump_generalization'

    def get_mor_pipeline_modules(self):
        return get_mor_pipeline_modules(self)

def get_mor_pipeline_modules(red_exp: ModelReductionExperiment):
    N = red_exp.model_experiment.get_model().get_dim() // 2
    mor_err_metrics=[
        RelEuclideanMorErrorMetric(sqrt=red_exp['sqrt_in_metric']),
        RelEuclideanMorErrorMetric(
            indices=list(range(N)), sqrt=red_exp['sqrt_in_metric'], name='q_2norm'),
        RelEuclideanMorErrorMetric(
            indices=list(range(N, 2*N)), sqrt=red_exp['sqrt_in_metric'], name='p_2norm'),
    ]
    if red_exp['time_sampler_mode'] == 'rom':
        proj_module = ProjectionMorPipelineModule.init_with_mor_time_sampler(
            red_exp['num_subsample'],
            mor_err_metrics=mor_err_metrics
        )
    elif red_exp['time_sampler_mode'] == 'fom':
        proj_module = ProjectionMorPipelineModule(
            ProjectionMorPipelineModuleFomTimeSampler(red_exp['num_subsample']),
            mor_err_metrics=mor_err_metrics,
            lock_time_sampler_to_first_run=False,
        )
    else:
        raise NotImplementedError('Unknown time_sampler_mode: {}'.format(red_exp['time_sampler_mode']))

    ham_module = HamiltonianModule()
    sympl_module = SymplecticityExperimentModule()

    return [proj_module, ham_module, sympl_module]