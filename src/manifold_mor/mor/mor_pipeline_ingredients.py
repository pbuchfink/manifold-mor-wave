'''
Collection of useful |MorPipelineModule|s and |HookFunctions|
'''
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from manifold_mor.context import get_current_context
from manifold_mor.mor.metrics import MorErrorMetric
from manifold_mor.mor.mor_pipeline import MorPipeline, MorPipelineModule
from manifold_mor.mor.reduced_model import ReducedModel, SymplecticPulledbackReducedModel
from manifold_mor.time_steppers.basic import HookFunction
from manifold_mor.trial_manifold.basic import TrialManifold
from manifold_mor.utils.logging import Logger
from matplotlib import pyplot as plt


##
## MODULES
##
class ProjectionMorPipelineModuleTimeSampler(ABC):
    '''Used to handle projections on different temporal grids.
    This is required if one of the ROMs fails and aborts early.'''
    @abstractmethod
    def sample_time(self, fom_sol_t: np.ndarray, red_sol_t: np.ndarray, i_mu: int) -> Tuple[np.ndarray, np.ndarray]:
        '''Samples time for the projection. Returns time steps and corresponding indices.'''
    
    def _find_relevant_time_steps(self, ref_sol_t: np.ndarray, red_sol_t: np.ndarray):
        # short cut to check if ref_sol_t equals red_sol_t
        if len(ref_sol_t) == len(red_sol_t) and np.isclose(sum((ref_sol_t - red_sol_t)**2), 0):
            return ref_sol_t.copy(), np.arange(len(ref_sol_t))
        # otherwise select relevant time steps
        else:
            idxs = []
            sol_t = []
            for t in ref_sol_t:
                idx = np.where(np.isclose(red_sol_t, t))[0]
                if len(idx) > 0:
                    idxs.append(idx[0])
                    sol_t.append(t)
            return np.array(sol_t), np.array(idxs, dtype=int)


class ProjectionMorPipelineModuleMorTimeSampler(ProjectionMorPipelineModuleTimeSampler):
    '''Use time dictated by the rom.'''
    def __init__(self, num_subsample: int = None) -> None:
        super().__init__()
        assert num_subsample is None or num_subsample > 0
        self.num_subsample = num_subsample
        
    def sample_time(self, fom_sol_t: np.ndarray, red_sol_t: np.ndarray, i_mu: int) -> Tuple[np.ndarray, np.ndarray]:
        '''Samples time for the projection.'''
        n_t = len(red_sol_t)
        if not self.num_subsample is None and n_t > self.num_subsample:
            idxs = np.linspace(0, n_t-1, self.num_subsample).astype(int)
        else:
            idxs = np.arange(n_t)
        return red_sol_t[idxs], idxs


class ProjectionMorPipelineModuleFomTimeSampler(ProjectionMorPipelineModuleTimeSampler):
    '''Use time dictated by the rom.'''
    def __init__(self, num_subsample: int = None) -> None:
        super().__init__()
        assert num_subsample is None or num_subsample > 0
        self.num_subsample = num_subsample
        
    def sample_time(self, fom_sol_t: np.ndarray, red_sol_t: np.ndarray, i_mu: int) -> Tuple[np.ndarray, np.ndarray]:
        '''Samples time for the projection.'''
        n_t = len(fom_sol_t)
        if not self.num_subsample is None and n_t > self.num_subsample:
            fixed_sol_t = fom_sol_t[np.linspace(0, n_t-1, self.num_subsample).astype(int)]
        else:
            fixed_sol_t = fom_sol_t
        return self._find_relevant_time_steps(fixed_sol_t, red_sol_t)


class ProjectionMorPipelineModuleFixedTimeSampler(ProjectionMorPipelineModuleTimeSampler):
    '''Use time dictated by the user (e.g. the temporal grid of a previous rom run).'''
    def __init__(self, time_steps: Dict[int, np.ndarray]) -> None:
        super().__init__()
        self.time_steps = time_steps
    
    def sample_time(self, fom_sol_t: np.ndarray, red_sol_t: np.ndarray, i_mu: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.time_steps, dict):
            fixed_sol_t = self.time_steps[i_mu]
        elif isinstance(self.time_steps, np.ndarray) and self.time_steps.ndim == 1:
            fixed_sol_t = self.time_steps
        else:
            raise NotImplementedError('Unknown type for self.time_steps: {}'.format(self.time_steps))
        
        return self._find_relevant_time_steps(fixed_sol_t, red_sol_t)


class ProjectionMorPipelineModule(MorPipelineModule):
    '''Module to compute and store projectes and MOR resutls of states and vector fields.
    If norms are provided, it also computes the errros of the respective states and vector fields.
    The results of this moduled are designed to be visualsed with the ProjectionMorPlot.'''
    NAME = 'projection'
    def __init__(
        self,
        time_sampler: ProjectionMorPipelineModuleTimeSampler,
        lock_time_sampler_to_first_run: bool = True,
        save_rom: bool = True,
        save_vf: bool = True,
        save_vf_unprojected: bool = True,
        mor_err_metrics: List[MorErrorMetric] = None,
        compute_proj_err_for_all: bool = False,
        log_arrays: bool = False,
    ) -> None:
        super().__init__()
        self.time_sampler = time_sampler
        self.lock_time_sampler_to_first_run = lock_time_sampler_to_first_run
        self.save_rom = save_rom
        self.save_vf = save_vf
        self.save_vf_unprojected = save_vf_unprojected
        self.mor_err_metrics = mor_err_metrics or []
        self.results_dict = dict()
        self.collect_vector_hook = None
        self._current_run_name = None
        self.compute_proj_err_for_all = compute_proj_err_for_all
        self.log_arrays = log_arrays
    
    @classmethod
    def init_with_mor_time_sampler(
        cls,
        num_subsample: int,
        lock_time_sampler_to_first_run: bool = True,
        save_rom: bool = True,
        save_vf: bool = True,
        save_vf_unprojected: bool = True,
        mor_err_metrics: List[MorErrorMetric] = None,
        compute_proj_err_for_all: bool = False,
    ):
        return cls(
            ProjectionMorPipelineModuleMorTimeSampler(num_subsample),
            lock_time_sampler_to_first_run,
            save_rom,
            save_vf,
            save_vf_unprojected,
            mor_err_metrics,
            compute_proj_err_for_all,
        )

    def initialize(self, run_name: str, mor_pipeline: MorPipeline, rom: ReducedModel):
        self._current_run_name = run_name
        # hasattr excludes LSPG
        if self.save_vf and self.save_rom and hasattr(rom.vector_field, 'reconstruct'):
            self.collect_vector_hook = CollectVectorFieldHookFunction(rom)
            self._hook_fcns = [self.collect_vector_hook]
        else:
            self.collect_vector_hook = None
            self._hook_fcns = None
        # containers for results
        self.results_dict[self._current_run_name] = dict()
        for i_mu in range(len(mor_pipeline.snapshots.mus)):
            self.results_dict[self._current_run_name][i_mu] = dict()

    def eval(
        self,
        mor_pipeline: MorPipeline,
        rom: ReducedModel,
        mu: dict,
        red_sol_x: np.ndarray,
        red_sol_t: np.ndarray,
    ) -> None:
        # list of loggable keys
        keys_loggable = []
        # get time steps and snapshots dictated by the time sampler
        i_mu = mor_pipeline.snapshots.get_mu_index(mu)
        fom_sol_t = mor_pipeline.snapshots.sol_t
        sol_t, idxs = self.time_sampler.sample_time(fom_sol_t, red_sol_t, i_mu)
        snapshots_mu = mor_pipeline.snapshots.get(mu)
        if not self.compute_proj_err_for_all:
            snapshots_mu = snapshots_mu[idxs]
        if self.save_rom:
            red_sol_x = red_sol_x[idxs]
        
        # sanity check
        if not all(np.isclose(fom_sol_t[idxs], sol_t)) or not all(np.isclose(red_sol_t[idxs], sol_t)):
            raise RuntimeError('Time steps do not match.')

        # save time
        self.results_dict[self._current_run_name][i_mu]['t'] = sol_t
        keys_loggable += ['t']
        self.results_dict[self._current_run_name][i_mu]['indices'] = idxs

        # states
        if self.save_rom:
            self.results_dict[self._current_run_name][i_mu]['x_rom'] = red_sol_x
        proj_states = mor_pipeline.manifold.project_state(snapshots_mu)
        self.results_dict[self._current_run_name][i_mu]['x_proj'] = proj_states

        # compute vf quantities
        vf_unproj = None
        vf_mu = None
        if self.save_vf or self.save_vf_unprojected:
            #TODO: batched processing of vector fields
            vf_unproj = np.empty_like(snapshots_mu)
            for i_x, x in enumerate(snapshots_mu):
                vf_unproj[i_x] = mor_pipeline.model.vector_field.eval(x)
            if len(self.mor_err_metrics) > 0:
                vf_mu = mor_pipeline.snapshots.get_rhs(mu)
                if not self.compute_proj_err_for_all:
                    vf_mu = vf_mu[idxs]

        # vector field (vf)
        if not self.collect_vector_hook is None:
            if self.save_rom:
                assert not self.collect_vector_hook is None
                self.results_dict[self._current_run_name][i_mu]['vf_rom'] = self.collect_vector_hook.results()['collected_vector_fields'][idxs]
            self.results_dict[self._current_run_name][i_mu]['vf_proj'] = rom.vector_field.project_tangent(snapshots_mu, vf_unproj)

        # vector field unprojected (vf unproj) 
        if self.save_vf_unprojected:
            self.results_dict[self._current_run_name][i_mu]['vf_unproj'] = vf_unproj
            #TODO: batched processing of vector fields
            if self.save_rom:
                vf_rom_unproj = np.empty_like(red_sol_x)
                for i_x, c_red_sol_x in enumerate(red_sol_x):
                    vf_rom_unproj[i_x] = mor_pipeline.model.vector_field.eval(c_red_sol_x)
                self.results_dict[self._current_run_name][i_mu]['vf_rom_unproj'] = vf_rom_unproj

        # log arrays as artifact
        context = get_current_context()
        logger = mor_pipeline.logger
        loggable_arrays = {}
        if not logger is None and self.log_arrays:
            for key, array in self.results_dict[self._current_run_name][i_mu].items():
                if key.startswith('t') or key.startswith('indices'):
                        continue
                
                loggable_arrays[key] = array

            tmp_array_path = os.path.join(context.get_cache_path(), ProjectionMorPipelineModule.get_artifact_file_name())
            np.savez(tmp_array_path, **loggable_arrays)
            try:
                mor_pipeline.logger.log_artifact(
                    tmp_array_path,
                    delete=True
                )
            except NotImplementedError:
                os.remove(tmp_array_path)

        # compute errors
        if len(self.mor_err_metrics) > 0:
            new_entries = dict()
            for mor_err_metric in self.mor_err_metrics:
                for key, array in self.results_dict[self._current_run_name][i_mu].items():
                    if key.startswith('x'):
                        target = snapshots_mu
                    elif key.startswith('vf'):
                        target = vf_mu
                    elif key.startswith('t') or key.startswith('indices'):
                        continue
                    else:
                        raise NotImplementedError('Unknown reference value for key "{}"'.format(key))
                    
                    # restrict to reference solution for idxs if rom-errors are computed
                    if len(target) > len(array) and 'rom' in key and self.compute_proj_err_for_all:
                        target = target[idxs]

                    errs = mor_err_metric.eval(array, target)
                    for errs_key, errs_val in errs.items():
                        dict_key = 'err_{}_{}'.format(key, errs_key)
                        new_entries[dict_key] = errs_val
                        if len(errs_val) > 1:
                            if hasattr(mor_err_metric, 'sqrt') and mor_err_metric.sqrt:
                                new_entries['total_{}'.format(dict_key)] = np.sqrt(np.sum(errs_val**2))
                            else:
                                sum_v = np.sum(errs_val)
                                new_entries['total_{}'.format(dict_key)] = sum_v
                                new_entries['total_sqrt_{}'.format(dict_key)] = np.sqrt(sum_v)
            keys_loggable += new_entries.keys()
            self.results_dict[self._current_run_name][i_mu].update(new_entries)

        # log errors
        if not logger is None:
            for key in keys_loggable:
                val = self.results_dict[self._current_run_name][i_mu][key]
                # append module name for better overview and to avoid conflicts between different modules
                key_logger = '{}_{}'.format(self, key)
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    for i_val, c_val in enumerate(val):
                        logger.log_metric(key_logger, c_val, i_val)
                else:
                    try:
                        logger.log_metric(key_logger, val)
                    except ValueError:
                        pass

        return True
    
    @staticmethod
    def get_artifact_file_name():
        return '{}.npz'.format(ProjectionMorPipelineModule.NAME + '_' + uuid.uuid4().hex[:4])

    def finalize(self, more_pipeline: MorPipeline, rom: ReducedModel):
        # if required, time temporal grid for subsequent computations is locked to the grid of the first run
        # this becomes relevant if any runs fail in between. Then the temporal grids might differ
        if self.lock_time_sampler_to_first_run:
            proj_results = self.results_dict[self._current_run_name]
            self.time_sampler = ProjectionMorPipelineModuleFixedTimeSampler(
                {i_mu: v_mu['t'] for i_mu, v_mu in proj_results.items()}
            )
            self.lock_time_sampler_to_first_run = False

    def results(self):
        return self.results_dict


class HamiltonianModule(MorPipelineModule):
    NAME = 'hamiltonian'
    LOG_KEY_HAMILTONIAN = 'hamiltonian'
    def eval(self, experiment: 'MorPipeline', rom: ReducedModel, mu: dict, red_sol_x: np.ndarray, red_sol_t: np.ndarray) -> None:
        ham = experiment.model.vector_field.Ham(red_sol_x)
        for i_t, c_ham in enumerate(ham):
            experiment.logger.log_metric(self.LOG_KEY_HAMILTONIAN, c_ham, i_t)
        if not ham is None and len(ham) > 0:
            experiment.logger.log_metric(self.LOG_KEY_HAMILTONIAN + '_max_deviation', max(abs(ham - ham[0])))


class SymplecticityExperimentModule(MorPipelineModule):
    NAME = 'symplecticity'
    '''A MorPipelineModule to evaluate the symplecticity in every time step.'''
    def __init__(self, log_pattern: bool = False, compute_sympl_snapshots: bool = False) -> None:
        super().__init__()
        self.log_pattern = log_pattern
        self.compute_sympl_snapshots = compute_sympl_snapshots
        self._current_run_name = None
        self.results_dict = {}

    def initialize(self, run_name: str, experiment: 'MorPipeline', rom: ReducedModel):
        self._current_run_name = run_name
        if isinstance(rom, SymplecticPulledbackReducedModel):
            self._hook_fcns = [SymplecticityErrorHookFunction(
                experiment.manifold,
                rom.vector_field.J_y,
                experiment.logger,
                log_pattern=self.log_pattern
            )]
        else:
            self._hook_fcns = []
        self.results_dict[self._current_run_name] = dict()
        for i_mu in range(len(experiment.snapshots.mus)):
            self.results_dict[self._current_run_name][i_mu] = dict()

    def eval(self, experiment: MorPipeline, rom: ReducedModel, mu: dict, red_sol_x: np.ndarray, red_sol_t: np.ndarray) -> None:
        i_mu = experiment.snapshots.get_mu_index(mu)
        if len(self._hook_fcns) > 0:
            sympl_hook = self._hook_fcns[0]
            assert isinstance(sympl_hook, SymplecticityErrorHookFunction)
            err_in_sympl = sympl_hook.results()
            mse_err_in_sympl = np.mean(err_in_sympl**2) / experiment.manifold.dim**2
            experiment.logger.log_metric('mse_err_in_sympl', mse_err_in_sympl)
            self.results_dict[self._current_run_name][i_mu]['mse_err_in_sympl'] = mse_err_in_sympl

        if self.compute_sympl_snapshots and isinstance(rom, SymplecticPulledbackReducedModel):
            sympl_hook = SymplecticityErrorHookFunction(
                experiment.manifold,
                rom.vector_field.J_y,
                None,
                log_pattern=self.log_pattern
            )
            snapshots = experiment.snapshots
            sol_t = snapshots.sol_t
            snap_mat = snapshots.get(mu)
            sympl_hook.initialize(snap_mat[0], sol_t[0], snapshots.dt, len(sol_t), mu)
            for i_t, (x, t) in enumerate(zip(snap_mat, sol_t)):
                sympl_hook.eval(experiment.manifold.inv_map(x), t, i_t)
            mse_err_in_sympl_snapshots = np.mean(sympl_hook.results()**2) / experiment.manifold.dim**2
            experiment.logger.log_metric('mse_err_in_sympl_snapshots', mse_err_in_sympl_snapshots)
            self.results_dict[self._current_run_name][i_mu]['mse_err_in_sympl_snapshots'] = mse_err_in_sympl_snapshots
    
    def results(self):
        return self.results_dict.copy()


##
## HOOK FUNCTIONS
##
class SymplecticityErrorHookFunction(HookFunction):
    '''Advantage: Jacobian is loaded from cache.'''
    def __init__(self, manifold: TrialManifold, J_y, logger: Logger, log_pattern: bool = False):
        self.manifold = manifold
        self.J_y = J_y
        self.error_in_symplecticity = None
        self.j2N = self.gen_J(int(manifold.ambient_dim/2))
        # self.j2n = self.gen_J(int(manifold.dim/2))
        self.eye_2n = np.eye(manifold.dim)
        self.logger = logger
        self.log_pattern = log_pattern
        if log_pattern:
            self.path_pattern = os.path.join('cache', 'err_in_sympl_approx_eye_{}.png')

    def gen_J(self, l):
        J = np.zeros((2*l, 2*l))
        J[:l, l:] = np.eye(l)
        J[l:, :l] = -np.eye(l)
        return J

    def initialize(self, x_0: np.ndarray, t_0: float, dt: float, n_t: int, mu: dict):
        self.error_in_symplecticity = np.zeros(n_t)

    def eval(self, x: np.ndarray, t: float, i_t: int):
        tan_map = self.manifold.tangent_map(x)
        approx_j2n = tan_map.T @ self.j2N @ tan_map
        approx_eye2n = np.zeros_like(approx_j2n)
        for i_col, col_approx_j2n in enumerate(approx_j2n):
            approx_eye2n[:, i_col] = self.J_y(x, col_approx_j2n)
        self.error_in_symplecticity[i_t] = np.sqrt(np.sum((approx_eye2n -self.eye_2n)**2))
        if not self.logger is None:
            self.logger.log_metric('err_in_sympl', self.error_in_symplecticity[i_t], i_t)
            if self.log_pattern:
                f_name = self.path_pattern.format(i_t)
                ims = plt.imshow((approx_eye2n - self.eye_2n)**2)
                plt.colorbar(ims)
                plt.savefig(f_name)
                plt.close()
                self.logger.log_artifact(f_name)

    def results(self):
        return self.error_in_symplecticity


class CollectVectorFieldHookFunction(HookFunction):
    def __init__(self, rom):
        self.rom = rom
        self.collected_vector_fields = None

    def initialize(self, x_0: np.ndarray, t_0: float, dt: float, n_t: int, mu: dict):
        self.collected_vector_fields = np.zeros((n_t, self.rom.manifold.ambient_dim))

    def eval(self, x: np.ndarray, t: float, i_t: int):
        self.collected_vector_fields[i_t] = self.rom.vector_field.reconstruct(x)
    
    def finalize(self, all_x: np.ndarray, all_t: np.ndarray, failed: bool = False):
        if failed:
            # shorted collected_vector_fields
            n_t = len(all_x)
            self.collected_vector_fields = self.collected_vector_fields[:n_t]

    def results(self):
        return {'collected_vector_fields': self.collected_vector_fields}

