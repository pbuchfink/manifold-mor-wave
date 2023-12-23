'''A class for reproducable and cachable experiments.
This way, e.g. snapshots of certain models can be easily cached.'''
import os
import random
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, OrderedDict

import numpy.random as nprandom
import torch
from manifold_mor.context import get_current_context
from manifold_mor.experiments.caching import PATH_CACHED_RESULTS, cached_result
from manifold_mor.utils.logging import Logger, MlflowLogger, MultiLogger


class ManifoldMorExperiment(ABC):
    PATH_LINKED_EXPERIMENTS = 'linked_experiments'
    PATH_EXPERIMENTS = 'experiments'
    PATH_REGISTER = 'register'
    DEFAULT_SEED = 28
    '''Describes an experiment with parameters.'''
    def __init__(self, **parameters) -> None:
        if parameters is None:
            parameters = dict()

        # consistency check
        if not self.has_all_required_param_keys(parameters):
            missing = ', '.join(self.missing_keys(parameters))
            raise RuntimeError('The experiment lacks the following keys: {}'.format(missing))

        self._parameters = parameters
        # keys of parameters that are changed via setter function experiment['key'] = val
        self.modified_parameter_keys = set()
        self._previous_working_dir = None

        # automatically set debugging mode
        if self.get('debug', False):
            get_current_context().options['debug'] = True

        # instance for loggers
        self._loggers = dict()
    
    # name methods
    @abstractmethod
    def get_experiment_type(self) -> str:
        ...
    
    @abstractmethod
    def get_experiment_name(self) -> str:
        ...
    
    def _get_required_param_keys(self) -> set:
        return set()

    # getter setter methods
    def __setitem__(self, key, value):
        self._parameters[key] = value
        self.modified_parameter_keys.add(key)
    
    def __getitem__(self, key):
        return self._parameters[key]
    
    def get(self, key, default=None):
        return self._parameters.get(key, default)
    
    def keys(self):
        return self._parameters.keys()
    
    def items(self):
        return self._parameters.items()
    
    # reproducability
    @cached_result
    def _get_rng_states(self) -> dict:
        """Get random number generator (rng) states for reproducability.

        Returns:
            dict: dict with all rng states.
        """
        return {
            'random': random.getstate(),
            'numpy': nprandom.get_state(),
            'torch': torch.get_rng_state(),
        }
    
    def init_rng(self) -> None:
        """Init random number generators (rng) for reproducability.
        """
        # get (cached) rng states
        rng_states = self._get_rng_states()
        # set rng states
        random.setstate(rng_states['random'])
        nprandom.set_state(rng_states['numpy'])
        torch.set_rng_state(rng_states['torch'])

    def seed_rng(self, seed: int = DEFAULT_SEED) -> None:
        """Seed random number genratros

        Be careful with seeding as this may change rng, if the seeded experiment is run by another
        experiment

        Args:
            seed (int, optional): The seed. Defaults to DEFAULT_SEED.
        """
        random.seed(seed)
        nprandom.seed(seed)
        torch.manual_seed(seed)

    # loggers methods
    def get_logger(self, name: str) -> MlflowLogger:
        try:
            return self._loggers[name]
        except KeyError:
            raise KeyError('Logger {} not found. Init logger before.'.format(name))
    
    def _init_logger(self, name: str) -> None:
        if name not in self._loggers.keys():
            log_folder = self.get_logging_folder()
            if not os.path.isdir(log_folder):
                os.makedirs(log_folder)
            self._loggers[name] = MlflowLogger(name, path_tracking=log_folder)
    
    def init_loggers(self, names: List[str]) -> None:
        for name in names:
            self._init_logger(name)

    def get_available_logger_names(self):
        return self._loggers.keys()

    def multi_log(self, name: str, new_logger: Logger):
        self._init_logger(name)
        old_logger = self._loggers[name]
        if isinstance(old_logger, MultiLogger):
            old_logger.append_logger(new_logger)
        else:
            if isinstance(new_logger, MultiLogger):
                self._loggers[name] = MultiLogger([old_logger] + new_logger.loggers)
            else:
                self._loggers[name] = MultiLogger([old_logger, new_logger])

    # clearing methods
    def clear_cached_data(
        self,
        include_rng_state: bool=False,
        include_linked_experiments: bool = True
    ):
        """Removes cached data correspoding to the experiment.

        Args:
            include_rng_state (bool, optional): Flag, whether to also clear the rng states.
                Defaults to False.
            include_linked_experiments (bool, optional): Flag, whether 
        """
        folders = [os.path.join(self.get_experiment_folder(), PATH_CACHED_RESULTS)]

        if include_linked_experiments:
            linked_exp_folder = os.path.join(self.get_experiment_folder(), self.PATH_LINKED_EXPERIMENTS, ManifoldMorExperiment.PATH_REGISTER)
            if os.path.isdir(linked_exp_folder):
                for type_folder in os.listdir(linked_exp_folder):
                    linked_exp_of_type = os.path.join(linked_exp_folder, type_folder)
                    folders += [os.path.join(linked_exp_of_type, f, PATH_CACHED_RESULTS) for f in os.listdir(linked_exp_of_type)]

        for folder in folders:
            if not os.path.isdir(folder):
                return
            if include_rng_state:
                # delete whole caching folder
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
            else:
                for fname in os.listdir(folder):
                    if not fname == '{}.pkl'.format(self._get_rng_states.__name__):
                        fpath = os.path.join(folder, fname)
                        if os.path.isfile(fpath) or os.path.islink(fpath):
                            os.unlink(fpath)
                        elif os.path.isdir(fpath):
                            shutil.rmtree(fpath)
        

    def clear_experiment(self):
        """Removes all files correspoding to the experiment.
        """
        folder = self.get_experiment_folder()
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    
    def modified_parameters(self):
        # sort modified_parameter_keys to ensure that string representation is not ambigous due to
        # permutations in modified_parameter_keys
        return OrderedDict([(k, self._parameters[k]) for k in sorted(self.modified_parameter_keys)])

    def __str__(self) -> str:
        name = self.get_experiment_name()
        if len(self.modified_parameter_keys) > 0:
            mod_str_list = []
            for mod_param_key, mod_param_val in self.modified_parameters().items():
                try:
                    str_val = str(mod_param_val)
                except Exception:
                    str_val = '[str failed]'
                mod_str_list.append('{}:{}'.format(mod_param_key, str_val))
            name += '({})'.format(','.join(mod_str_list))

        # TODO: nicer solution, this can lead to non-unique file names
        # truncate too long names (otherwise file error)
        if len(name) > 255:
            name = name[:252] + '...'
        return name

    def __repr__(self) -> str:
        return str(self)

    def get_params_for_logging(self):
        '''Get parameters for logging. Two modes are supported:
            1.) context.options['log_experiment_params_all'] = True
                all parameters are reported for logging
            2.) context.options['log_experiment_params_all'] = False (default)
                only modified parameters are reported for logging
                this reduces the amount of logged (redundant parameters)
        '''
        context = get_current_context()
        if context.options['log_experiment_params_all']:
            return self._parameters.copy()
        else:
            return self.modified_parameters()

    def missing_keys(self, parameters: dict) -> set:
        return self._get_required_param_keys().difference(parameters.keys())

    def has_all_required_param_keys(self, parameters) -> bool:
        return len(self.missing_keys(parameters)) == 0
    
    def _get_relative_experiment_folder(self) -> str:
        folders = []
        experiment_type = self.get_experiment_type()
        if experiment_type is not None:
            folders += [ManifoldMorExperiment.PATH_REGISTER, experiment_type]
        folders += [str(self)]
        return os.path.join(*folders)
    
    def get_experiment_folder(self) -> str:
        context = get_current_context()
        return os.path.join(
            context.root_folder,
            ManifoldMorExperiment.PATH_EXPERIMENTS,
            self._get_relative_experiment_folder(),
        )
    
    def get_logging_folder(self) -> str:
        exp_folder = self.get_experiment_folder()
        return os.path.join(exp_folder, MlflowLogger.DEFAULT_FOLDER)

    def run(self, **technical_params) -> None:
        """Run the experiment.

        Args:
            technical_params: should only be technical parameters (like e.g. verbosity). All other
                parameters should be stored in self._parameters to ensure reproducability.
        """
        verbose = technical_params.get('verbose', True)
        if verbose:
            exp_type = self.get_experiment_type()
            exp_name = '{}/{}'.format(exp_type, str(self)) if exp_type is not None else str(self)
            print('Running {}.'.format(exp_name))
        self._prepare(**technical_params)
        # check if cached results already exist and experiment does not need to be executed
        if not self.cached_results_exist(**technical_params):
            self.init_rng()
            self._execute(**technical_params)
        self._finalize(**technical_params)
        if verbose:
            print('Finished {}.'.format(exp_name))
    
    def cached_results_exist(self, **technical_params) -> bool:
        """Check if cached results of the experiment do already exist and thus does not need to be run.

        Returns:
            bool: flag, whether cached results exist and experiment is not required to be run.
                Defaults to False.
        """
        return False

    def _prepare(self, **technical_params) -> None:
        """Prepare the experiment.

        Args:
            technical_params: should only be technical parameters (like e.g. verbosity). All other
                parameters should be stored in self._parameters to ensure reproducability.
        """
        # change os directory to experiment folder
        # this is required to use the cached_result machinery
        cwd = os.getcwd()
        experiment_folder = self.get_experiment_folder()
        if os.path.normpath(cwd) != os.path.normpath(experiment_folder):
            self._previous_working_dir = cwd
            if not os.path.isdir(experiment_folder):
                os.makedirs(experiment_folder)
            os.chdir(experiment_folder)

    @abstractmethod
    def _execute(self, **technical_params) -> None:
        """Execute the (prepared) experiment.

        Args:
            technical_params: should only be technical parameters (like e.g. verbosity). All other
                parameters should be stored in self._parameters to ensure reproducability.
        """
        ...

    def _finalize(self, **technical_params) -> None:
        """Finalize the (prepared and executed) experiment.

        Args:
            technical_params: should only be technical parameters (like e.g. verbosity). All other
                parameters should be stored in self._parameters to ensure reproducability.
        """
        # this is required for nested experiments
        if self._previous_working_dir:
            os.chdir(self._previous_working_dir)
            self._previous_working_dir = None

    def link_experiment(
        self,
        experiment_to_link: 'ManifoldMorExperiment',
        bidirectional: bool=True,
        use_relative_paths: bool=True,
    ):
        """Link an experiment to the current experiment.

        Args:
            experiment_to_link (ManifoldMorExperiment): the experiment to link.
            bidirectional (bool, optional): flag, whether to link folders bidirectionally. Defaults to True.
            use_relative_paths (bool, optional): flag, whether to use relative paths
        """
        linked_file = os.path.join(
            self.get_experiment_folder(),
            self.PATH_LINKED_EXPERIMENTS,
            experiment_to_link._get_relative_experiment_folder(),
        )
        linked_file_dirname = os.path.dirname(linked_file)
        if not os.path.isdir(linked_file_dirname):
            os.makedirs(linked_file_dirname)
        target = experiment_to_link.get_experiment_folder()
        abs_target = target
        if use_relative_paths:
            target = os.path.relpath(target, linked_file_dirname)
        # check for correctness if link exists already
        if os.path.exists(linked_file):
            exist_linked_file = os.readlink(linked_file)
            if not exist_linked_file in (target, abs_target):
                raise FileExistsError((
                    'Cannot link experiments. Link already exists'
                    '\n\t Existing link points to {}.'
                    '\n\t New link wants to point to {}.'
                ).format(exist_linked_file, target))
        else:
            os.symlink(target, linked_file, target_is_directory=True)
        # make bidirectional linking if requested
        if bidirectional:
            experiment_to_link.link_experiment(self, bidirectional=False,
                                               use_relative_paths=use_relative_paths)

    def copy_mlflow(
        self,
        experiment_source: 'ManifoldMorExperiment',
        logger_names: Dict[str, str] = None,
    ):
        """Copy mlflow from experiment_to_copy with logger_source to logger_target.

        But the use of a |MultiLogger| is adviced instead of copying experiments. The advantage is
        that the results are logged "live" and the logging times are correct if the |MultiLogger| is
        used.

        Args:
            experiment_source (ManifoldMorExperiment): the experiment_to_copy from.
            logger_names (Dict, optional): logger names as dict with targets as keys and sources as
                values. Default is None which takes all loggers available in source.
        """
        if logger_names is None:
            logger_names = {n: n for n in experiment_source.get_available_logger_names()}

        for logger_name_target, logger_name_source in logger_names.items():
            if logger_name_target not in self.get_available_logger_names():
                self._init_logger(logger_name_target)
            logger_target = self.get_logger(logger_name_target)
            logger_source = experiment_source.get_logger(logger_name_source)
            exp_type = experiment_source.get_experiment_type()
            exp_name = str(experiment_source)
            source_name = '{}/{}'.format(exp_type, exp_name) if exp_type is not None else exp_name
            logger_target.copy_experiment(logger_source, source_name=source_name) 
