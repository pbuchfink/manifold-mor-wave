"""Basic class for a manifold consists of one or multiple charts.
The current implementations restric to manifolds with a single chart."""
import os
import pickle
from abc import ABC, abstractmethod

from manifold_mor.chart.basic import Chart, IterativelyTrainableChart, TrainableChart
from manifold_mor.context import get_current_context
from manifold_mor.fields.covector import CovectorField, ReducedCovectorField
from manifold_mor.fields.metric import Metric, ReducedMetric
from manifold_mor.fields.vector import VectorField
from manifold_mor.fields.vector_reduced import ReducedVectorField, VectorFieldProjector

# Problem: want to work in low-dimensional coordinates

class TrialManifold(ABC):
    '''Describes the |TrialManifold| the system is constrained to.
    The |TrialManifold| is formulated as an embedded submanifold S ⊆ M of an ambient manifold M.
    The charts are given by one or multiple |Chart|s
        hat{φ_S}: R^{dim} ⊇ φ_S(U_S) → φ_M(U_M) ⊆ R^{ambient_dim}
    where (U_M, φ_M) and (U_S, φ_S) are charts of M and S respectively.
    '''

    def __init__(self, model_name, name):
        self.model_name = model_name
        self.name = name
        # dim and ambient_dim are set by the chart
        # this enables that POD decides on dim based on the singular values
        self.dim = None
        self.ambient_dim = None
        # user-defined parameters that can get logged in experiments
        self.loggable_parameters = dict()

    @abstractmethod
    def map(self, xr):
        '''Map reduced xr to full x.'''
        ...

    @abstractmethod
    def inv_map(self, x):
        '''Map full x to reduced xr.'''
        ...
    
    def project_state(self, x):
        '''Project state onto manifold.'''
        return self.map(self.inv_map(x))

    @abstractmethod
    def tangent_map(self, xr):
        '''Compute tangent map at reduced coordinate xr.'''
        ...

    @abstractmethod
    def pullback_vector_field(self, vector_field: VectorField, projector: VectorFieldProjector):
        '''Pullback a |VectorField| with a |VectorFieldProjector|.'''
        ...

    def __str__(self):
        '''Gives an identification of the manifold as string. Is used in logging.'''
        return self.name

    def pullback(self, other):
        '''Pullback a |CovectorField| or |Metric| to the reduced manifold.'''
        if isinstance(other, VectorField):
            raise RuntimeError('Use pullback_vector_field with a specific projector instead.')
        elif isinstance(other, CovectorField):
            return ReducedCovectorField(other, self)
        elif isinstance(other, Metric):
            return ReducedMetric(other, self)
        else:
            raise ValueError('Pullback not implemented for ' + type(other) + '.')

    def is_valid(self):
        '''Checks if components are initialized.'''
        return self.dim is not None and self.ambient_dim is not None
    
    @staticmethod
    def _default_cache_path(model_name, name):
        return os.path.join(TrialManifold._default_cache_base_path(model_name), name)
    
    @staticmethod
    def _default_cache_base_path(model_name):
        context = get_current_context()
        return os.path.join('manifolds', model_name)
    
    @staticmethod
    def get_available_manifold_paths(model_name):
        folder = TrialManifold._default_cache_base_path(model_name)
        if os.path.exists(folder):
            candidates = os.listdir(folder)
            return [c for c in candidates if len(c) > 4 and c[-4:] == '.pkl']
        else:
            return []

    def save(self, path:str = None):
        if path is None:
            path = TrialManifold._default_cache_path(self.model_name, self.name)
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not path.endswith('.pkl'):
            path += '.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> 'TrialManifold':
        if not path.endswith('.pkl'):
            path += '.pkl'
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    
    @staticmethod
    def load_from_names(model_name, name) -> 'TrialManifold':
        return TrialManifold.load(TrialManifold._default_cache_path(model_name, name))


class SingleChartTrialManifold(TrialManifold):
    def __init__(self, chart: Chart, model_name: str, name: str):
        super().__init__(model_name=model_name, name=name)
        chart.register_manifold(self)
        self.chart = chart
        # copy ambient_dim and dim of chart if available
        # for some trainable charts, these attributes are only available after training
        if self.chart.dim:
            self.dim = self.chart.dim
        if self.chart.ambient_dim:
            self.ambient_dim = self.chart.ambient_dim

    def is_valid(self):
        return super().is_valid() \
            and isinstance(self.chart, Chart) and self.chart.is_valid()

    def map(self, xr):
        return self.chart.map(xr)

    def inv_map(self, x):
        return self.chart.inv_map(x)

    def tangent_map(self, xr):
        return self.chart.tangent_map(xr)

    def encoder_tangent_map(self, x):
        return self.chart.encoder_tangent_map(x)
    
    def encoder_jvp(self, x, v):
        return self.chart.encoder_jvp(x, v)

    def pullback_vector_field(self, vector_field: VectorField, projector: VectorFieldProjector):
        if not self.chart.is_valid_projector(projector):
            raise RuntimeError('Projector ({}) is not valid for this chart ({}).'.format(
                type(projector), type(self.chart)
            ))
        return ReducedVectorField(vector_field, self, projector)


class TrainableTrialManifold(TrialManifold):
    @abstractmethod
    def train(self, snapshots, training_params, logger=None, model=None):
        '''Train one or multiple |Chart|s from |Snapshots| with training_parameters.'''
        ...
    
    @abstractmethod
    def get_epoch(self) -> int:
        '''Get current epoch of training. Returns none if not iteratively trainable.'''
        ...


class SingleChartTrainableTrialManifold(SingleChartTrialManifold,TrainableTrialManifold):
    def __init__(self, chart, model_name, name):
        assert isinstance(chart, TrainableChart)
        super().__init__(chart=chart, model_name=model_name, name=name)

    def train(self, snapshots, training_params, logger=None, model=None, **kwargs):
        if logger:
            end_run = False
            if not logger.has_active_run():
                end_run = True
                manifold_name = training_params['manifold_name'] if 'manifold_name' in training_params.keys() else self.name
                logger.start_run(manifold_name)
            manifold_string = training_params['manifold_string'] if 'manifold_string' in training_params.keys() else str(self)
            logger.log_param(logger.PARAM_MANIFOLD, manifold_string)
            logger.log_params(training_params)

        summary = self.chart.train(snapshots, training_params, logger=logger, model=model, **kwargs)
        if not summary is None:
            self.loggable_parameters.update(summary)
        self.ambient_dim = self.chart.ambient_dim
        self.dim = self.chart.dim

        if logger:
            logger.log_param(logger.PARAM_NO_PARAMS_DECODER, self.chart.number_of_parameters_in_decoder())
            logger.log_param(logger.PARAM_NO_PARAMS_TOTAL, self.chart.number_of_parameters_in_total())
            logger.log_param(logger.PARAM_DIM, str(self.dim))
            if end_run:
                logger.end_run()
        return summary

    def get_epoch(self) -> int:
        if isinstance(self.chart, IterativelyTrainableChart):
            return self.chart.get_epoch()
        else:
            return None
