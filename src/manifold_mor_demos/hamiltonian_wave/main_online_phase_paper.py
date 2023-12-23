

from argparse import ArgumentParser
from typing import List

import ray
from manifold_mor.chart.auto_encoder import AutoEncoderChart
from manifold_mor_demos.hamiltonian_wave.manifold_experiments import (
    RED_DIMS, BumpA0AutoEncoderManifoldExperiment, BumpA1AutoEncoderManifoldExperiment, BumpAutoEncoderBasedManifoldExperiment, BumpLinearSubspaceExperiment)
from manifold_mor_demos.hamiltonian_wave.model_experiments import \
    BumpModelExperiment
from manifold_mor.context import ManifoldMorContext, get_current_context
from manifold_mor.experiments.basic import ManifoldMorExperiment
from manifold_mor.mor.mor_pipeline import RomGenerator


class BumpOnlineExperiment(ManifoldMorExperiment):
    # DCAs A_{0,s}, A_{0} and A_{1} from the paper
    MANIFOLD_TYPES = ['ae_0_s', 'ae_0', 'ae_1', 'pca', 'psd_cotan']
    def __init__(self, debug: bool = False) -> None:
        reductions_symplectic = [RomGenerator.REDUCTION_SYMPL_CANONICAL]
        reductions_non_symplectic = [RomGenerator.REDUCTION_PULLBACK, RomGenerator.REDUCTION_LSPG]
        super().__init__(
            debug=debug,
            reductions={
                'ae_0_s': reductions_symplectic,
                'ae_0': reductions_non_symplectic,
                'ae_1': reductions_non_symplectic,
                'pca': [RomGenerator.REDUCTION_LINEAR_SUBSPACE, RomGenerator.REDUCTION_LSPG],
                'psd_cotan': reductions_symplectic,
            },
        )
        if debug:
            self.modified_parameter_keys.add('debug')

    def get_experiment_name(self) -> str:
        return 'bump_online'

    def get_experiment_type(self) -> str:
        return None

    def _prepare(self, **technical_params) -> None:
        self.seed_rng()  # seed experiment to get same rng_states every time

        super()._prepare(**technical_params)
        self.init_loggers(['fom', 'mor'])

    def _execute_mor(
        self,
        manifold_experiment: 'BumpAutoEncoderBasedManifoldExperiment',
        model_experiments: List[BumpModelExperiment],
        reductions: List[str],
    ):
        if not manifold_experiment.cached_results_exist():
            raise RuntimeError('Trained manifold cannot be found: {}.'.format(manifold_experiment))

        for reduction in reductions:
            reduction_experiment = manifold_experiment.get_reduction_experiment(
                reduction, None, None
            )
            if self['debug']:
                reduction_experiment['model_experiment'] = model_experiments[BumpModelExperiment.MU_TEST_GENERALIZATION]
            if not reduction_experiment.cached_results_exist():
                reduction_experiment.multi_log('mor', self.get_logger('mor'))
                reduction_experiment.run()
                self.link_experiment(reduction_experiment)

    def get_manifold_experiment(self, red_dim: int, manifold_type: str):
        assert manifold_type in self.MANIFOLD_TYPES
        if manifold_type == 'ae_0':
            return BumpA0AutoEncoderManifoldExperiment(1., red_dim, debug=self['debug'])
        elif manifold_type == 'ae_0_s':
            return BumpA0AutoEncoderManifoldExperiment(
                0.9,
                red_dim,
                shift=AutoEncoderChart.SHIFT_INITIAL_VALUE,
                debug=self['debug']
            )
        elif manifold_type == 'ae_1':
            return BumpA1AutoEncoderManifoldExperiment(red_dim, debug=self['debug'])
        elif manifold_type in ['pca', 'psd_cotan']:
            return BumpLinearSubspaceExperiment(red_dim, manifold_type, debug=self['debug'])
        else:
            raise NotImplementedError('Unknown manifold_type: {}'.format(manifold_type))

    def _execute(self, **technical_params) -> None:
        # run (and cache) FOM
        model_experiments = dict()
        for mu_scenario in [
            BumpModelExperiment.MU_TEST_GENERALIZATION,
            BumpModelExperiment.MU_TRAINING,
        ]:
            model_experiment = BumpModelExperiment(mu_scenario, debug=self['debug'])
            if not model_experiment.cached_results_exist():
                model_experiment.multi_log('fom', self.get_logger('fom'))
                model_experiment.run()
                self.link_experiment(model_experiment)
            model_experiments[mu_scenario] = model_experiment

        red_dims = RED_DIMS
        if self['debug']:
            red_dims = red_dims[:4]
        context = get_current_context()
        is_parallel_train_mor = context.is_parallel(context.TASK_MOR)
        if is_parallel_train_mor:
            obj_ids = []

        for manifold_type in self.MANIFOLD_TYPES:
            for red_dim in red_dims:
                # run (and cache) training of trial manifold
                manifold_experiment = self.get_manifold_experiment(red_dim, manifold_type)
                train_kwargs = dict(
                    manifold_experiment=manifold_experiment,
                    model_experiments=model_experiments,
                    reductions=self['reductions'][manifold_type]
                )
                if is_parallel_train_mor:
                    @ray.remote(**context.get_resources_remote(context.TASK_MOR))
                    def parallel_execute(**kwargs):
                        context.reinit()
                        return self._execute_mor(**kwargs)

                    obj_ids.append(parallel_execute.remote(**train_kwargs))
                else:
                    self._execute_mor(**train_kwargs)
        
        # wait for parallel runs to finish
        if is_parallel_train_mor:
            ray.get(obj_ids)


if __name__ == "__main__":
    # initialize context
    context = ManifoldMorContext()
    # get arguments
    parser = ArgumentParser(
        usage='run mor experiments of linear wave equation'
    )
    # debug flag
    parser.add_argument("--debug", action="store_true", help="whether to use debug mode",
        default=False)
    # parallel
    parser.add_argument("--ncpu", default=0, help="number of cpus to use in parallel execusion",
        type=int)
    parser.add_argument("--cpus", default=1, help="number of cpus per task",
        type=int)
    # requires to set up cluster before,
    #   see https://docs.ray.io/en/latest/cluster/cloud.html#cluster-private-setup
    parser.add_argument("--cluster", action="store_true", help="sets address to auto",
        default=False)
    args = parser.parse_args()

    # parallel execusion
    if args.ncpu > 0 or args.cluster:
        if not ray.is_initialized():
            if args.cluster:
                address = 'auto'
                num_cpus = None
            else:
                address = None
                num_cpus = args.ncpu
            ray.init(include_dashboard=False, num_cpus=num_cpus,  address=address) #helpful to debug: local_mode=True
        context.specify_parallel_task(context.TASK_MOR, cpu=args.cpus)

    # run experiment
    BumpOnlineExperiment(debug=args.debug).run()
