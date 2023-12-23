'''Helper class for logging with Mlflow or Tune.'''
import os
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import numpy as np
from mlflow.entities import Metric, Param, RunStatus, RunTag
from mlflow.exceptions import MlflowException
from mlflow.store.tracking import (DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
                                   SEARCH_MAX_RESULTS_THRESHOLD)
from mlflow.tracking.client import MlflowClient
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_RUN_NOTE
from mlflow.utils.validation import MAX_ENTITIES_PER_BATCH, _validate_param


class Logger(ABC):
    METRIC_LOSS_TRAINING = 'loss_training'
    METRIC_LOSS_VALIDATION = 'loss_validation'
    METRIC_ERROR = 'error'
    METRIC_TIMESTEPPER_SOLVERITERATIONS = 'timeStepping_solverIterations'
    METRIC_TIMESTEPPER_RESIDUAL = 'timeStepping_residual'
    PARAM_MANIFOLD = 'manifold'
    PARAM_NO_PARAMS_DECODER = 'no_params_decoder'
    PARAM_NO_PARAMS_TOTAL = 'no_params_total'
    PARAM_COMPOSITION = 'composition'
    PARAM_REDUCTION = 'reduction'
    PARAM_DIM = 'dim'

    STATUS_FINISHED = RunStatus.to_string(RunStatus.FINISHED)
    STATUS_FAILED = RunStatus.to_string(RunStatus.FAILED)
    STATUSES = (STATUS_FINISHED, STATUS_FAILED)
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    @abstractmethod
    def is_running(self):
        ...

    @abstractmethod
    def start_run(self, run_name=None, tags=None):
        ...

    @abstractmethod
    def end_run(self, status=STATUS_FINISHED):
        ...

    @abstractmethod
    def log_param(self, key, value):
        ...

    def log_params(self, param_dict):
        for param_key, param_value in param_dict.items():
            self.log_param(param_key, param_value)

    @abstractmethod
    def log_metric(self, key, value, step=None):
        ...
    
    def log_metrics(self, metric_dict, step=None):
        for key, value in metric_dict:
            self.log_metric(key, value, step=step)

    def log_artifact(self, path, artifact_path=None, delete=False):
        raise NotImplementedError()

    @abstractmethod
    def add_note(self, text):
        ...

    @abstractmethod
    def has_active_run(self):
        ...

    def report_training_callback(self, **kwargs):
        self.log_metrics(kwargs, step=kwargs['epoch'])

    def report_time_stepper_callback(self, **kwargs):
        self.log_metric(self.METRIC_TIMESTEPPER_RESIDUAL, kwargs['residual'], step=kwargs['time_step'])
        self.log_metric(self.METRIC_TIMESTEPPER_SOLVERITERATIONS, kwargs['solver_iterations'], step=kwargs['time_step'])
        return 0


class MlflowLogger(Logger):
    '''Logging with MLflow'''
    DEFAULT_FOLDER = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
    def __init__(
        self,
        experiment_name: str,
        path_tracking: str = None,
        mlflow_client: MlflowClient = None,
        ignore_invalid_params: bool = True,
        start_experiment: bool = True,
    ):
        super().__init__(experiment_name)
        if mlflow_client is None:
            tracking_uri = None
            if path_tracking is not None:
                tracking_uri = path_to_local_file_uri(os.path.abspath(path_tracking))
            mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        self.mlflow_client = mlflow_client
        self.experiment_id = None
        self.active_run = None
        self.run_notes = ''
        self.ignore_invalid_params = ignore_invalid_params
        if start_experiment:
            self.start_experiment()

    def is_running(self):
        return not self.experiment_id is None

    def start_experiment(self):
        assert self.experiment_id is None
        experiment = self.mlflow_client.get_experiment_by_name(
            self.experiment_name
        )
        if experiment is None:
            # If it does not exist, create the experiment
            self.experiment_id = self.mlflow_client.create_experiment(self.experiment_name)
        else:
            # If it already exists then get the id
            self.experiment_id = experiment.experiment_id

    def start_run(self, run_name=None, tags=None):
        assert self.active_run is None, 'only non-nested runs'
        if not run_name is None:
            if tags is None:
                tags = dict()
            tags[MLFLOW_RUN_NAME] = run_name
        self.active_run = self.mlflow_client.create_run(self.experiment_id, tags=tags)
        self.run_notes = ''

    def end_run(self, status=Logger.STATUS_FINISHED):
        assert status in self.STATUSES
        self.mlflow_client.set_terminated(self.active_run.info.run_id, status)
        self.active_run = None

    def log_param(self, key, value):
        if isinstance(value, dict):
            self.log_params({key + '_' + k: v for k, v in value.items()})
        else:
            self.mlflow_client.log_param(self.active_run.info.run_id, key, value)
    
    def log_params(self, param_dict):
        params_arr = []
        def add_to_params_arr(key, value):
            if not isinstance(value, str):
                value = str(value)
            try:
                _validate_param(key, value)
                params_arr.append(Param(key, value))
            except MlflowException as e:
                msg = 'Cannot log param {} due to: {}'.format(key, e)
                if self.ignore_invalid_params:
                    warnings.warn(msg)
                else:
                    raise MlflowException(msg) from e

        for key, value in param_dict.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    add_to_params_arr(key + '_' + k, v)
            else:
                add_to_params_arr(key, value)

        self.mlflow_client.log_batch(self.active_run.info.run_id, params=params_arr)

    def log_metric(self, key, value, step=None):
        if not isinstance(value, (float, int)):
            print('Skipped logging metric with key={}, value={}, step={} due to non-numeric value.'.format(key, value, step))
        # remove inf values since mlflow crashes on inf
        if np.isinf(value):
            value = np.nan
        self.mlflow_client.log_metric(self.active_run.info.run_id, key, value, step=step)
    
    def log_metrics(self, metric_dict: dict, step=None):
        metrics_arr = []
        timestamp = int(time.time() * 1000)
        for key, value in metric_dict.items():
            if isinstance(value, (float, int)):
                # remove inf values since mlflow crashes on inf
                if np.isinf(value):
                    value = np.nan
                metrics_arr.append(Metric(key, value, timestamp, step or 0))
            else:
                print('Skipped logging metric with key={}, value={}, step={} due to non-numeric value.'.format(key, value, step))
        self.mlflow_client.log_batch(self.active_run.info.run_id, metrics=metrics_arr)

    def log_artifact(self, path, artifact_path=None, delete=False):
        self.mlflow_client.log_artifact(self.active_run.info.run_id, path, artifact_path=artifact_path)
        if delete:
            os.remove(path)

    def add_note(self, text):
        self.run_notes += '\n' + text
        self.mlflow_client.set_tag(self.active_run.info.run_id, MLFLOW_RUN_NOTE, self.run_notes)

    def has_active_run(self):
        return not self.active_run is None

    def copy_experiment(self, logger_source: 'MlflowLogger', add_copy_tags=True, source_name: str = None):
        # copy runs
        runs = logger_source.mlflow_client.search_runs(
            logger_source.experiment_id,
            max_results=SEARCH_MAX_RESULTS_THRESHOLD,
        )
        for run in runs:
            source_run_id = run.info.run_id
            self.start_run()
            target_run_id = self.active_run.info.run_id
            # metrics with full history
            metrics = []
            for k in run.data.metrics.keys():
                metrics += logger_source.mlflow_client.get_metric_history(source_run_id, k)
            # parameters
            params = [Param(k, v) for k, v in run.data.params.items()]
            # tags
            tags = [RunTag(k, v) for k, v in run.data.tags.items() if not k == MLFLOW_RUN_NOTE]
            if add_copy_tags:
                tags.append(RunTag('copied_from_run_id', source_run_id))
            # log batch, respecting MAX_ENTITIES_PER_BATCH
            if len(metrics) + len(params) + len(tags)  > MAX_ENTITIES_PER_BATCH:
                # might also occur for params or tags but unlikely
                n_metrics = 0
                contents = dict(params=params, tags=tags)
                while n_metrics < len(metrics):
                    inc = MAX_ENTITIES_PER_BATCH - sum(len(c) for c in contents.values())
                    contents['metrics'] = metrics[n_metrics: n_metrics + inc]
                    self.mlflow_client.log_batch(target_run_id, **contents)
                    contents = dict()
                    n_metrics += inc
            else:
                self.mlflow_client.log_batch(target_run_id, metrics, params, tags)
            # artifacts
            artifacts_folder = logger_source.mlflow_client.download_artifacts(source_run_id, '')
            self.mlflow_client.log_artifacts(target_run_id, artifacts_folder)
            self.end_run(run.info.status)
        # copy experiment meta data
        source_experiment = logger_source.mlflow_client.get_experiment(logger_source.experiment_id)
        source_exp_note = None
        for k, v in source_experiment.tags.items():
            if k == MLFLOW_RUN_NOTE:
                source_exp_note = v
                continue
            self.mlflow_client.set_experiment_tag(self.experiment_id, k, v)
        # compose experiment note
        target_experiment = self.mlflow_client.get_experiment(self.experiment_id)
        target_exp_note = target_experiment.tags.get(MLFLOW_RUN_NOTE, '')
        if source_name is None:
            source_specifier = logger_source.experiment_name
        else:
            source_specifier = '{}/{}'.format(source_name, logger_source.experiment_name)
        target_exp_note += '\n imported data from {} on {}'.format(
            source_specifier,
            str(datetime.now())
        )
        if source_exp_note is not None:
            target_exp_note += '\n' + target_exp_note
        self.mlflow_client.set_experiment_tag(self.experiment_id, MLFLOW_RUN_NOTE, target_exp_note)


class TuneLogger(Logger):
    '''Logging with MLflow'''
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        self._buffered_metrics_with_iter = dict()

    def is_running(self):
        return True

    def start_experiment(self):
        pass

    def start_run(self, run_name=None, tags=None):
        pass

    def end_run(self, status=Logger.STATUS_FINISHED):
        pass

    def log_param(self, key, value):
        pass #these get neglected (found no way to report through tune to mlflow)

    def log_metric(self, key, value, step=None):
        self._buffered_metrics_with_iter[key] = value
    
    def log_metrics(self, metric_dict, step=None):
        self._buffered_metrics_with_iter.update(metric_dict)

    def add_note(self, text):
        pass #these get neglected (found no way to report through tune to mlflow)

    def has_active_run(self):
        return True

    def report_training_callback(self, **kwargs):
        summary = kwargs.get('summary', None)
        assert not summary is None and isinstance(summary, dict)
        summary.update(self._buffered_metrics_with_iter)
        self._buffered_metrics_with_iter = {}


class MultiLogger(Logger):
    def __init__(self, loggers: List[Logger], experiment_name=None):
        if experiment_name is None:
            experiment_name = 'multilog_{}'.format(loggers[0].experiment_name)
        super().__init__(experiment_name)
        self.loggers = loggers

    def append_logger(self, new_logger: Logger):
        if isinstance(new_logger, MultiLogger):
            self.loggers += new_logger.loggers
        else:
            self.loggers.append(new_logger)

    def is_running(self):
        return any(l.is_running() for l in self.loggers)

    def start_run(self, run_name=None, tags=None):
        for l in self.loggers:
            l.start_run(run_name=run_name, tags=tags)

    def end_run(self, status=Logger.STATUS_FINISHED):
        for l in self.loggers:
            l.end_run(status)

    def log_param(self, key, value):
        for l in self.loggers:
            l.log_param(key, value)
    
    def log_params(self, param_dict):
        for l in self.loggers:
            l.log_params(param_dict)

    def log_metric(self, key, value, step=None):
        for l in self.loggers:
            l.log_metric(key, value, step=step)

    def log_metrics(self, metric_dict, step=None):
        for l in self.loggers:
            l.log_metrics(metric_dict, step=step)

    def log_artifact(self, path, artifact_path=None, delete=False):
        for l in self.loggers:
            l.log_artifact(path, artifact_path=artifact_path, delete=delete)

    def add_note(self, text):
        for l in self.loggers:
            l.add_note(text)

    def has_active_run(self):
        return any(l.has_active_run() for l in self.loggers)
