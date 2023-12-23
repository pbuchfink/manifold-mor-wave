'''The context is an element useful for parallel execution with ray.
It is used to share low-memory data (such as settings and default paths) to different workers.'''
import os
import pwd
import socket
import subprocess
import warnings

import numpy as np
import torch
from mlflow.utils.mlflow_tags import (MLFLOW_GIT_COMMIT, MLFLOW_SOURCE_NAME,
                                      MLFLOW_USER)

CURRENT_CONTEXT = None


class Context:
    def __init__(self) -> None:
        self.reinit()

    def reinit(self):
        '''Call reinit once on parallel units to copy context to CURRENT_CONTEXT on the parallel instance.'''
        global CURRENT_CONTEXT
        CURRENT_CONTEXT = self

    def get_meta_data(self):
        return {
            MLFLOW_GIT_COMMIT: subprocess.check_output(["git", "describe", "--always"]).strip().decode('utf8'),
            MLFLOW_USER: pwd.getpwuid(os.getuid())[0],
            MLFLOW_SOURCE_NAME: socket.gethostname()
        }


class ParallelContext(Context):
    '''Specify resources per task'''
    TASK_TUNE_RUN = 'tune_run'
    def __init__(self) -> None:
        self._resources_per_task = dict()
        super().__init__()

    def get_resources(self, task: str):
        try:
            return self._resources_per_task[task]
        except KeyError:
            return None

    def get_resources_remote(self, task: str):
        '''Load resources and rename to fit the syntax of ray remote.'''
        resources = self.get_resources(task)
        if not all(k in ('cpu', 'gpu') for k in resources.keys()):
            raise NotImplementedError('get_resources_remote is yet only implemented for cpu und gpu key.')
        return {
            'num_cpus': resources.get('cpu', None),
            'num_gpus': resources.get('gpu', None),
        }

    def specify_parallel_task(self, task: str, **kwargs):
        self._resources_per_task[task] = kwargs.copy()

    @property
    def parallel_tasks(self):
        return self._resources_per_task.keys()

    def is_parallel(self, task):
        return task in self.parallel_tasks \
            and not self._resources_per_task[task] is None

    def uses(self, task: str, resource: str):
        return self.is_parallel(task) \
            and resource in self._resources_per_task[task].keys()

    def uses_cpu(self, task: str):
        return self.uses(task, 'cpu')

    def uses_gpu(self, task: str):
        return self.uses(task, 'gpu')


class ManifoldMorContext(ParallelContext):
    TASK_GENERATE_SNAPSHOTS = 'generate_snapshots'
    TASK_MOR = 'mor'
    TASK_TRAINING = 'training'
    def __init__(self) -> None:
        super().__init__()
        self.options = {
            'log_experiment_params_all': False,
            'use_caching': True,
            'debug': False,
            'raise_errors_during_timestepping': False,
        }
        self.root_folder = os.getcwd()

    def reinit(self):
        # set dtype to double
        from manifold_mor.chart.auto_encoder import AutoEncoderChart
        AutoEncoderChart._torch_dtype = np.float64
        torch.set_default_dtype(torch.float64)
        return super().reinit()

    def raise_errors_during_timestepping(self, val: bool = True):
        self.options['raise_errors_during_timestepping'] = val


def get_current_context() -> ManifoldMorContext:
    if CURRENT_CONTEXT is None:
        raise RuntimeError('Context has not been initialized yet.')
    return CURRENT_CONTEXT
