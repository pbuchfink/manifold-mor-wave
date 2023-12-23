'''
Default settings for manifolds (and more) disussed in Lee & Carlberg 2020
'''
import copy
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from manifold_mor_demos.hamiltonian_wave.model_experiments import BumpModelExperiment
from manifold_mor.chart.auto_encoder import AutoEncoderChart
from manifold_mor.chart.auto_encoder_conv1d import Conv1dAutoEncoderChart
from manifold_mor.chart.basic import (IterativelyTrainableCallback,
                                      IterativelyTrainableChart,
                                      TrainableChart)
from manifold_mor.chart.identity import IdentityChart
from manifold_mor.chart.linear import PcaChart
from manifold_mor.chart.symplectic_linear import PsdCotangentLiftChart
from manifold_mor.chart.torch_losses import (LOSS_MSE_SCALED, LOSS_TRAINING,
                                             LOSS_VALIDATION, DataLoss,
                                             HybridLoss, WeakSymplecticLoss)
from manifold_mor.chart.torch_modules import TorchAutoEncoder
from manifold_mor.context import get_current_context
from manifold_mor.experiments.caching import cached_result
from manifold_mor.experiments.model import ModelExperiment
from manifold_mor.experiments.trial_manifold import (HasReductionExperiment,
                                                     TrialManifoldExperiment)
from manifold_mor.fields.vector_reduced import ReducedVectorField
from manifold_mor.models.basic import Model
from manifold_mor.mor.snapshots import Snapshots
from manifold_mor.trial_manifold.basic import (
    SingleChartTrainableTrialManifold, SingleChartTrialManifold, TrialManifold)
from manifold_mor.trial_manifold.component import \
    ComponentwiseTrainableManifold
from manifold_mor.utils.logging import Logger

if TYPE_CHECKING:
    from manifold_mor_demos.hamiltonian_wave.reduction_experiments import \
        BumpModelReductionExperiment
    from manifold_mor.fields.vector import VectorField
    from manifold_mor.fields.vector_reduced import VectorFieldProjector

# reduced dimensions investigated in the original paper
RED_DIMS = (2, 4, 6, 8, 10, 12, 18, 24, 30)
N_BOUDARY_NODES = 4 #2 for p, 2 for q

class BumpTrialManifold(ComponentwiseTrainableManifold):
    def __init__(self, inner_chart: TrainableChart, model_name, name):
        self.inner_chart = inner_chart
        name_inner = name + '_inner'
        name_boundary = name + '_boundary'
        name_boundary_q = name_boundary + '_q'
        name_boundary_p = name_boundary + '_p'

        sys_dim = inner_chart.ambient_dim + N_BOUDARY_NODES
        idx_boundary_q = (0, sys_dim//2-1)
        idx_boundary_p = (sys_dim//2, sys_dim-1)
        idx_boundary = set(idx_boundary_q).union(idx_boundary_p)
        idx_inner = tuple(set(np.arange(sys_dim)).difference(idx_boundary))
        
        super().__init__(
            manifolds=[
                SingleChartTrialManifold(IdentityChart(N_BOUDARY_NODES // 2), model_name, name_boundary_q),
                SingleChartTrainableTrialManifold(inner_chart, model_name, name_inner),
                SingleChartTrialManifold(IdentityChart(N_BOUDARY_NODES // 2), model_name, name_boundary_p)
            ],
            component_indices=[
                idx_boundary_q,
                idx_inner,
                idx_boundary_p
            ],
            model_name=model_name,
            idx_trainable=1,
            name=name,
        )

    def get_epoch(self) -> int:
        if isinstance(self.inner_chart, IterativelyTrainableChart):
            return self.inner_chart.get_epoch()
        else:
            return None

    def pullback_vector_field(self, vector_field: 'VectorField', projector: 'VectorFieldProjector'):
        if not self.inner_chart.is_valid_projector(projector):
            raise RuntimeError('Projector ({}) is not valid for this chart ({}).'.format(
                type(projector), type(self.inner_chart)
            ))
        return ReducedVectorField(vector_field, self, projector)

    # modify J_y
    def J_y(self, x, y):
        n = y.shape[0] // 2
        if y.ndim == 1:
            return np.hstack([y[-2:], y[n:-2], -y[2:n], -y[:2]])
        else:
            return np.vstack([y[-2:, ...], y[n:-2, ...], -y[2:n, ...], -y[:2, ...]])

    def inv_J_y(self, x, y):
        return -self.J_y(x, y)


class BumpPcaTrialManifold(BumpTrialManifold):
    def __init__(self, model_name, name):
        super().__init__(PcaChart(), model_name, name)


class BumpAutoEncoderTrialManifold(BumpTrialManifold):
    def __init__(self, config, model_name, name):
        chart = Conv1dAutoEncoderChart.from_layer_params(config)

        # use GPU if avaibale
        context = get_current_context()
        if context.uses_gpu(context.TASK_TRAINING) and torch.cuda.is_available():
            chart.to('cuda')

        super().__init__(chart, model_name, name)


class BumpManifoldExperiment(TrialManifoldExperiment):
    def __init__(
        self,
        red_dim: int,
        debug: bool=False,
        **kwargs
    ) -> None:
        model_experiment = BumpModelExperiment(
            shifted=True,
            mu_scenario=BumpModelExperiment.MU_TRAINING,
            debug=debug,
        )
        super().__init__(
            model_experiment = model_experiment,
            red_dim = red_dim,
            debug = debug,
            **kwargs,
        )
        self.modified_parameter_keys.update(['red_dim'])
        self.model = model_experiment.get_model()
        if debug:
            self.modified_parameter_keys.add('debug')

    def get_reduction_experiment(
        self,
        reduction: str,
        essentials: dict,
        summary: dict,
        manifold_experiment: TrialManifoldExperiment = None,
    ) -> 'BumpModelReductionExperiment':
        from manifold_mor_demos.hamiltonian_wave.reduction_experiments import \
            BumpModelReductionExperiment
        return BumpModelReductionExperiment(
            self if manifold_experiment is None else manifold_experiment,
            reduction=reduction,
            debug=self['debug']
        )
    
    def get_training_snapshots(self) -> Snapshots:
        assert self.model_experiment['mu_scenario'] == ModelExperiment.MU_TRAINING

        # TODO: nicer
        sys_dim = self.model.get_dim()
        idx_boundary_q = (0, sys_dim//2-1)
        idx_boundary_p = (sys_dim//2, sys_dim-1)
        idx_boundary = set(idx_boundary_q).union(idx_boundary_p)
        idx_inner = tuple(set(np.arange(sys_dim)).difference(idx_boundary))

        snapshots = self.model_experiment.get_snapshots().get_subsampled_snapshots(idx_inner, '1')
        snapshots.make_dataset(use_rhs_if_available=False) # save some RAM
        return snapshots

class BumpLinearSubspaceExperiment(BumpManifoldExperiment):
    BASIS_GENERATION = [
        'pca',
        'psd_cotan',
        'psd_svdlike',
    ]
    def __init__(
        self,
        red_dim: int,
        basis_generation: str,
        debug: bool = False,
    ) -> None:
        assert basis_generation in self.BASIS_GENERATION
        super().__init__(red_dim, debug)
        self['basis_generation'] = basis_generation

    def get_untrained_manifold(self) -> BumpTrialManifold:
        if self['basis_generation'] == 'pca':
            inner_chart = PcaChart()
        elif self['basis_generation'] == 'psd_cotan':
            inner_chart = PsdCotangentLiftChart()

        model = self.model_experiment.get_model()
        
        # set ambient dim for BumpTrialManifold
        inner_chart.ambient_dim = model.get_dim() - N_BOUDARY_NODES
        return BumpTrialManifold(inner_chart, model.name, str(self))

    def get_experiment_name(self) -> str:
        return 'bump_linear_subspace'




class BumpAutoEncoderBasedManifoldExperiment(BumpManifoldExperiment):
    '''Specifies options used by all Auto-Encoder-based manifolds corresponding to A0 in the paper.'''
    def __init__(
        self,
        red_dim: int,
        debug: bool=False,
        shift: str=AutoEncoderChart.SHIFT_INITIAL_VALUE_AFTER_TRAINING,
        **kwargs,
    ) -> None:
        super().__init__(
            red_dim = red_dim,
            percentage_validation = .2,
            channel_names = ['q', 'p'],
            track_channels = True,
            n_epochs = 20 if debug else 1000,
            n_epochs_early_stop = 5 if debug else 100,
            shift=shift,
            loss_fun = LOSS_MSE_SCALED,
            fully_activation = torch.nn.ELU(),
            fully_batch_norm = False,
            activation = torch.nn.ELU(),
            batch_norm = False,
            scaling = Conv1dAutoEncoderChart.SCALING_NORMALIZE,
            debug=debug,
            **kwargs,
        )
        if shift != AutoEncoderChart.SHIFT_INITIAL_VALUE_AFTER_TRAINING:
            self.modified_parameter_keys.add('shift')

    def get_untrained_manifold(self) -> TrialManifold:
        return BumpAutoEncoderTrialManifold(self, str(self.model_experiment), str(self))

    def get_essentials(self, manifold: BumpAutoEncoderTrialManifold) -> dict:
        essentials = super().get_essentials(manifold)
        torch_ae = manifold.inner_chart._torch_ae
        # training and validation loss are based on the same loss
        # however, it is important to use 2 separate objects of the component loss functions
        # since the loss values are buffered in the objects and if the object
        # is shared between train and val, then the buffered losses get mixed
        essentials['train_loss_func'] = self._get_loss(torch_ae, LOSS_TRAINING)
        essentials['val_loss_func'] = self._get_loss(torch_ae, LOSS_VALIDATION)
        return essentials

    def _get_loss(self, torch_ae: TorchAutoEncoder, name: str) -> HybridLoss:
        # data loss
        loss_fun = self['loss_fun']
        loss_config = copy.deepcopy(self._parameters) #TODO: nicer
        if loss_fun == LOSS_MSE_SCALED:
            idx_scaling = list(dict(torch_ae.encoder.named_children())).index(Conv1dAutoEncoderChart.SCALING_NORMALIZE)
            scaling_layer = torch.nn.Sequential(
                torch_ae.encoder[:idx_scaling+1],
                torch.nn.Flatten(),
            )
            loss_config['loss_params'] = {'scaling_layer': scaling_layer}
        else:
            raise NotImplementedError('Unknown parameter loss_fun={}'.format(loss_fun))

        losses = [DataLoss.from_loss_params(loss_config, self.model)]
        weights = [self['loss_weight_data']]

        # weak symplectic loss
        if not np.isclose(self['loss_weight_data'], 1.):
            weights += [1. - self['loss_weight_data']]
            losses += [WeakSymplecticLoss(
                torch_ae, 2 * self.model_experiment['n_x'] - N_BOUDARY_NODES, self['red_dim'])]

        return HybridLoss(losses, weights=weights, name=name)


class BumpA0AutoEncoderManifoldExperiment(BumpAutoEncoderBasedManifoldExperiment):
    '''Parameters of the DCAs A_{0,s} and A_{0} depending on loss_weight_data'''
    VALUES_LOSS_WEIGHT_DATA = (0.9, 1.)
    def __init__(
        self,
        loss_weight_data: float,
        red_dim: int,
        shift: str=AutoEncoderChart.SHIFT_INITIAL_VALUE_AFTER_TRAINING,
        debug: bool=False,
    ) -> None:
        assert loss_weight_data in self.VALUES_LOSS_WEIGHT_DATA
        kernel_size = 17
        super().__init__(
            red_dim = red_dim,
            conv_channels = (64, 32, 16, 8, 4, 2, 2),
            conv_lengths = (2, 16, 64, 128, 256, 512, 2048),
            stride = (8, 4, 2, 2, 2, 4),
            fully_sizes = (red_dim, 128),
            lr = 0.00044320502453207137, # learning rate
            batch_size = 15,
            initialization = {
                'type': 'Kaiming_ReLu',
                'distribution': 'normal',
            },
            kernel_size = kernel_size,
            # "half" padding (which does NOT preserve size since s>1)
            padding = kernel_size // 2,
            shift = shift,
            debug = debug,
        )
        self['loss_weight_data'] = loss_weight_data

    def get_experiment_name(self) -> str:
        return 'bump_ae_0'


class BumpA1AutoEncoderManifoldExperiment(BumpAutoEncoderBasedManifoldExperiment):
    '''Parameters of the DCA A_{1}'''
    def __init__(
        self,
        red_dim: int,
        shift: str=AutoEncoderChart.SHIFT_INITIAL_VALUE_AFTER_TRAINING,
        debug: bool=False,
    ) -> None:
        kernel_size = 33
        super().__init__(
            red_dim = red_dim,
            conv_channels = (64, 32, 16, 8, 4, 2),
            conv_lengths = (4, 128, 256, 512, 1024, 2048),
            stride = (32, 2, 2, 2, 2),
            fully_sizes = (red_dim, 132, 256),
            lr = 0.00010453821425234425, # learning rate
            batch_size = 25,
            initialization = {
                'type': 'Xavier',
                'distribution': 'uniform',
            },
            kernel_size = kernel_size,
            # "half" padding (which does NOT preserve size since s>1)
            padding = kernel_size // 2,
            loss_weight_data = 1.,
            shift = shift,
            debug = debug,
        )

    def get_experiment_name(self) -> str:
        return 'bump_ae_1'
