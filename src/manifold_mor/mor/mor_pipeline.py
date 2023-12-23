'''A general framework to perform MOR experiments.'''
import time
import traceback
from typing import List

import numpy as np
from manifold_mor.fields.vector_reduced import (
    EncoderVectorFieldProjector, LinearOrthogonalVectorFieldProjector,
    LinearSymplecticVectorFieldProjector, MoorePenroseVectorFieldProjector,
    WeightedMoorePenroseVectorFieldProjector)
from manifold_mor.models.basic import Model
from manifold_mor.mor.reduced_model import (LspgReducedModel,
                                            PulledbackReducedModel,
                                            PulledbackVectorFieldReducedModel,
                                            ReducedModel,
                                            SymplecticPulledbackReducedModel)
from manifold_mor.mor.snapshots import Snapshots
from manifold_mor.time_steppers.basic import HookFunction, TimeStepperError
from manifold_mor.trial_manifold.basic import TrialManifold
from manifold_mor.utils.logging import Logger


class RomGenerator():
    REDUCTION_LINEAR_SUBSPACE = 'linear_subspace'
    REDUCTION_SYMPLECTIC_LINEAR_SUBSPACE = 'sympl_linear_subspace'
    REDUCTION_LSPG = 'lspg'
    REDUCTION_PULLBACK = 'pullback'
    REDUCTION_LSPG_WEIGHTED = 'lspg_weighted'
    REDUCTION_PULLBACK_WEIGHTED = 'pullback_weighted'
    REDUCTION_PULLBACK_ENCODER = 'pullback_encoder'
    REDUCTION_SYMPL_CANONICAL = 'sympl_canonical'
    REDUCTION_SYMPL_PULLBACK = 'sympl_pullback'
    REDUCTIONS = [REDUCTION_LSPG, REDUCTION_PULLBACK, REDUCTION_LSPG_WEIGHTED, REDUCTION_PULLBACK_WEIGHTED,
                  REDUCTION_SYMPL_CANONICAL, REDUCTION_SYMPL_PULLBACK]
    def get_rom(self, config, manifold, model):
        # reduced model
        if config['reduction'] == self.REDUCTION_LINEAR_SUBSPACE:
            if (hasattr(model, 'vector_field')
                and model.time_stepper.is_for_vector_field()
            ):
                #TODO: separate rom class that enables online-efficiency
                return PulledbackVectorFieldReducedModel(
                    manifold,
                    model,
                    LinearOrthogonalVectorFieldProjector(manifold),
                )
            else:
                raise NotImplementedError()
        elif config['reduction'] == self.REDUCTION_SYMPLECTIC_LINEAR_SUBSPACE:
            if (hasattr(model, 'vector_field')
                and model.time_stepper.is_for_vector_field()
            ):
                #TODO: symplectic form via manifold
                raise NotImplementedError('This would require symplectic form in projector')
                #TODO: separate rom class that enables online-efficiency
                return PulledbackVectorFieldReducedModel(
                    manifold,
                    model,
                    LinearSymplecticVectorFieldProjector(manifold),
                )
            else:
                raise NotImplementedError()
        if config['reduction'] == self.REDUCTION_PULLBACK:
            if (hasattr(model, 'vector_field')
                and model.time_stepper.is_for_vector_field()
            ):
                return PulledbackVectorFieldReducedModel(
                    manifold,
                    model,
                    MoorePenroseVectorFieldProjector(manifold),
                )
            else:
                return PulledbackReducedModel(manifold, model)
        elif config['reduction'] == self.REDUCTION_LSPG:
            return LspgReducedModel(manifold, model)
        elif config['reduction'] == self.REDUCTION_PULLBACK_WEIGHTED:
            return PulledbackVectorFieldReducedModel(
                manifold,
                model,
                WeightedMoorePenroseVectorFieldProjector(model.vector_field, manifold)
            )
        elif config['reduction'] == self.REDUCTION_LSPG_WEIGHTED:
            return LspgReducedModel(manifold, model, weighted=True)
        elif config['reduction'] == self.REDUCTION_PULLBACK_ENCODER:
            return PulledbackVectorFieldReducedModel(
                manifold,
                model,
                EncoderVectorFieldProjector(manifold)
            )
        elif config['reduction'] == self.REDUCTION_SYMPL_PULLBACK:
            return SymplecticPulledbackReducedModel(manifold, model)
        elif config['reduction'] == self.REDUCTION_SYMPL_CANONICAL:
            return SymplecticPulledbackReducedModel(manifold, model, use_canonical=True)
        else:
            raise NotImplementedError('Unkown reduction: {}'.format(config['reduction']))


class MorPipelineModule():
    NAME = None
    def __init__(self) -> None:
        super().__init__()
        self._hook_fcns = None
    
    def get_hook_fcns(self):
        return self._hook_fcns

    def initialize(self, run_name: str, experiment: 'MorPipeline', rom: ReducedModel):
        ...
    
    def eval(
        self,
        experiment: 'MorPipeline',
        rom: ReducedModel,
        mu: dict,
        red_sol_x: np.ndarray,
        red_sol_t: np.ndarray,
    ) -> None:
        return False
    
    def finalize(self, experiment: 'MorPipeline', rom: ReducedModel):
        ...

    def results(self):
        return None
    
    def __str__(self) -> str:
        return self.NAME
   

class MorPipeline:
    def __init__(
        self,
        model: Model,
        snapshots: Snapshots,
        manifold: TrialManifold,
        rom_generator: RomGenerator,
        logger: Logger,
        modules: List[MorPipelineModule] = None,
        hook_fcns: List[HookFunction] = None,
        verbose: bool = True,
    ) -> None:

        self.model = model
        self.snapshots = snapshots
        self.manifold = manifold
        self.rom_generator = rom_generator
        self.logger = logger
        self.modules = modules or []
        self.hook_fcns = hook_fcns or []
        self.verbose = verbose

    def run(self, run_name: str, config: dict):
        rom = self.rom_generator.get_rom(config, self.manifold, self.model)

        for module in self.modules:
            module.initialize(run_name, self, rom)
        
        # gather hook functions
        # the list is copied to avoid interferences with list entries from previous runs
        hook_fcns = self.hook_fcns.copy()
        for module in self.modules:
            hook_fcns += module.get_hook_fcns() or []

        # run all parameters
        for mu in self.snapshots.mus:
            experiment_name_mu = '{}_mu_{}'.format(run_name, mu['mu_idx'])
            self.logger.start_run(experiment_name_mu)
            self.logger.log_params(config)
            self.logger.log_params({
                self.logger.PARAM_MANIFOLD: str(self.manifold),
            })
            if hasattr(self.manifold, 'loggable_parameters'):
                self.logger.log_params({'manifold_' + k: v for k, v in self.manifold.loggable_parameters.items()})

            # set model mu
            self.model.set_mu(mu)

            # compute reduced solution
            rt_mor = time.time()
            status = self.logger.STATUS_FINISHED
            try:
                red_sol_x, _, red_sol_t = rom.solve(*self.snapshots.get_temporal_data(), mu, logger=self.logger, hook_fcns=hook_fcns)
            except TimeStepperError as err:
                print('Aborted one parameter in {} due to: {}'.format(experiment_name_mu, str(err)))
                red_sol_x = err.red_sol_x
                red_sol_t = err.red_sol_t
                self.logger.add_note('{}\n{}'.format(
                    str(err),
                    traceback.format_exc()
                ))
                status = self.logger.STATUS_FAILED
            self.logger.log_metric('rt_mor', time.time() - rt_mor)

            # evaluate modules
            for module in self.modules:
                rt_eval = time.time()
                log_rt = module.eval(self, rom, mu, red_sol_x, red_sol_t)
                if log_rt:
                    rt_eval = time.time() - rt_eval
                    self.logger.log_metric('rt_{}'.format(module), rt_eval)
                
            self.logger.end_run(status)
        
        # finalize modules
        for module in self.modules:
            module.finalize(self, rom)

    def results(self):
        results = dict()
        for m in self.modules:
            res = m.results()
            if res is not None:
                results[str(m)] = res
        return results
