"""All classes for Deep Convolutional Autoencoders as proposed in Lee & Carlberg 2020.
In our framework, the DCA is interpreted as a special chart.
"""
import warnings
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from manifold_mor.chart.auto_encoder import AutoEncoderChart
from manifold_mor.chart.basic import IterativelyTrainableCallback
from manifold_mor.chart.torch_losses import (LOSS_VALIDATION, HybridLoss,
                                             RelativeChannelMseLoss)
from manifold_mor.chart.torch_modules import (InversePermutation,
                                              InvStandardization, Permutation,
                                              Standardization, TorchAutoEncoder, Unflatten)
from manifold_mor.models.basic import Model
from manifold_mor.mor.snapshots import Snapshots
from manifold_mor.utils.logging import Logger
from manifold_mor.experiments.basic import ManifoldMorExperiment


class Conv1dAutoEncoderChart(AutoEncoderChart):
    SCALING_NORMALIZE = 'scaling_normalize'
    INV_SCALING_NORMALIZE = 'inv_scaling_normalize'
    LAYER_PARAM_KEYS = (
        'conv_lengths', 'conv_channels', 'kernel_size', 'stride',
        'padding', 'activation', 'fully_sizes', 'fully_activation', 'scaling',
        'batch_norm', 'fully_batch_norm'
    )
    LAYER_KEY_UNFLATTEN = 'unflatten'

    @classmethod
    def from_layer_params(cls, layer_params):
        assert isinstance(layer_params, (dict, ManifoldMorExperiment))
        missing_keys = set(Conv1dAutoEncoderChart.LAYER_PARAM_KEYS).difference(layer_params.keys())
        assert len(missing_keys) == 0, 'layer_params misses the keys: {}'.format(str(missing_keys))
        assert isinstance(layer_params['conv_channels'], (list, tuple, np.ndarray))
        assert isinstance(layer_params['fully_sizes'], (list, tuple, np.ndarray))

        # set sizes
        n_conv_layers = len(layer_params['conv_channels']) - 1
        n_full_layers = len(layer_params['fully_sizes']) - 1

        # check an standardize tensor_in
        conv_lengths = _check_or_make_array(layer_params['conv_lengths'], (int, np.integer), n_conv_layers + 1)
        conv_channels = _check_or_make_array(layer_params['conv_channels'], (int, np.integer), n_conv_layers + 1)
        kernel_size = _check_or_make_array(layer_params['kernel_size'], (int, np.integer), n_conv_layers)
        stride = _check_or_make_array(layer_params['stride'], (int, np.integer), n_conv_layers)
        padding = _check_or_make_array(layer_params['padding'], (int, np.integer), n_conv_layers)
        batch_norm = _check_or_make_array(layer_params['batch_norm'], (bool, np.bool_), n_conv_layers)
        activation = _check_or_make_array(layer_params['activation'], torch.nn.Module, n_conv_layers)
        fully_sizes = _check_or_make_array(layer_params['fully_sizes'], (int, np.integer), n_full_layers + 1)
        fully_batch_norm = _check_or_make_array(layer_params['fully_batch_norm'], (bool, np.bool_), n_full_layers)
        fully_activ = _check_or_make_array(layer_params['fully_activation'], torch.nn.Module, n_full_layers)
        scaling = layer_params['scaling']
        permutation = layer_params.get('permutation', None)
        shift = layer_params.get('shift', AutoEncoderChart.SHIFT_INITIAL_VALUE_AFTER_TRAINING)

        ## build encoder
        encoder_layers = OrderedDict()

        if permutation is not None:
            encoder_layers['permutation'] = Permutation(permutation)

        encoder_layers[cls.LAYER_KEY_UNFLATTEN] = Unflatten(-1, (conv_channels[-1], conv_lengths[-1]))

        scaling_layer = None
        if scaling:
            if scaling == Conv1dAutoEncoderChart.SCALING_NORMALIZE:
                scaling_layer = Standardization(
                    scaling=[1.]*conv_channels[-1],
                    shift=[0.]*conv_channels[-1],
                )
                encoder_layers[Conv1dAutoEncoderChart.SCALING_NORMALIZE] = scaling_layer
            else:
                raise RuntimeError('Undefined scaling: {}'.format(scaling))

        for i_layer in reversed(range(n_conv_layers)):
            # sanity check - conv network sizes
            in_len = conv_lengths[i_layer + 1]
            # from pyTorch doc (dialation = 1)
            out_len = int((in_len + 2*padding[i_layer] - kernel_size[i_layer]) / stride[i_layer] + 1)
            assert out_len == conv_lengths[i_layer],\
                'Output dimension of the conv layer {} does not match. Expected: {}. Given: {}.'.format(i_layer, out_len, conv_lengths[i_layer])

            encoder_layers['conv1d_{}'.format(i_layer)] = torch.nn.Conv1d(
                conv_channels[i_layer+1],
                conv_channels[i_layer],
                kernel_size[i_layer],
                stride[i_layer],
                padding[i_layer]
            )
            if batch_norm[i_layer]:
                encoder_layers['batch_norm_{}'.format(i_layer)] = torch.nn.BatchNorm1d(conv_channels[i_layer])
            encoder_layers['activation_{}'.format(i_layer)] = activation[i_layer]

        encoder_layers['flatten'] = torch.nn.Flatten()

        # sanity check - conv to fully network size
        assert fully_sizes[-1] == (conv_channels[0] * conv_lengths[0]),\
            'Dimensions of last conv layer and full layer do not match!'

        # fully connected part
        for i_layer in reversed(range(n_full_layers)):
            encoder_layers['full_encoder_{}'.format(i_layer)] = \
                torch.nn.Linear(fully_sizes[i_layer+1], fully_sizes[i_layer])
            if i_layer > 0:
                if fully_batch_norm[i_layer]:
                    encoder_layers['full_batch_norm_{}'.format(i_layer)] = torch.nn.BatchNorm1d(fully_sizes[i_layer])
                encoder_layers['full_activation_{}'.format(i_layer)] = fully_activ[i_layer]

        encoder = torch.nn.Sequential(encoder_layers)

        ## build decoder
        decoder_layers = OrderedDict()
        
        # fully connected part
        for i_layer in range(n_full_layers):
            if i_layer > 0:
                if fully_batch_norm[i_layer]:
                    decoder_layers['full_batch_norm_{}'.format(i_layer)] = torch.nn.BatchNorm1d(fully_sizes[i_layer])
                decoder_layers['full_activation_{}'.format(i_layer)] = fully_activ[i_layer]
            decoder_layers['full_decoder_{}'.format(i_layer)] = \
                torch.nn.Linear(fully_sizes[i_layer], fully_sizes[i_layer+1])

        decoder_layers[cls.LAYER_KEY_UNFLATTEN] = Unflatten(-1, (conv_channels[0], int(fully_sizes[-1] / conv_channels[0])))

        # convolutional part
        for i_layer in range(n_conv_layers):
            # compute correct output_padding
            in_len = conv_lengths[i_layer]
            # from pyTorch doc (dialation = 1)
            out_len = (in_len - 1) * stride[i_layer] - 2 * padding[i_layer] + kernel_size[i_layer]
            output_padding = conv_lengths[i_layer + 1] - out_len

            if batch_norm[i_layer]:
                decoder_layers['batch_norm_{}'.format(i_layer)] = torch.nn.BatchNorm1d(conv_channels[i_layer])
            decoder_layers['activation_{}'.format(i_layer)] = activation[i_layer]
            decoder_layers['transposed_conv1d_{}'.format(i_layer)] = torch.nn.ConvTranspose1d(
                conv_channels[i_layer],
                conv_channels[i_layer+1],
                kernel_size[i_layer],
                stride=stride[i_layer],
                padding=padding[i_layer],
                output_padding=output_padding
            )

        inv_scaling_layer = None
        if scaling:
            if scaling == Conv1dAutoEncoderChart.SCALING_NORMALIZE:
                inv_scaling_layer = InvStandardization(
                    scaling=[1.]*conv_channels[-1],
                    shift=[0.]*conv_channels[-1],
                )
                decoder_layers[Conv1dAutoEncoderChart.INV_SCALING_NORMALIZE] = inv_scaling_layer
            else:
                raise RuntimeError('Undefined scaling: {}'.format(scaling))

        decoder_layers['flatten'] = torch.nn.Flatten()

        if permutation is not None:
            decoder_layers['inv_permutation'] = InversePermutation(permutation)

        decoder = torch.nn.Sequential(decoder_layers)

        torch_ae = TorchAutoEncoder(encoder, decoder, scaling_layer, inv_scaling_layer)

        chart = cls(torch_ae, shift=shift)
        chart.ambient_dim = conv_lengths[-1] * conv_channels[-1]
        chart.dim = layer_params['fully_sizes'][0]

        # apply initialization, if specified
        init_params = layer_params.get('initialization', None)
        if not init_params is None:
            assert isinstance(init_params, dict)

            init_fun = lambda module: cls.initialize(module, init_params)
            torch_ae.apply(init_fun)

        return chart

    def train_setup(
        self,
        snapshots: Snapshots,
        training_params: dict,
        logger: Optional[Logger] = None,
        model: Optional[Model] = None,
        callbacks: Optional[List[IterativelyTrainableCallback]] = None,
        essentials: Optional[dict] = None,
    ) -> dict:
        essentials = super().train_setup(snapshots, training_params, logger, model, callbacks, essentials)

        # set scaling in scaling layers according to training data
        # respects different input channels with different scalings
        if training_params['n_epochs'] > 0 \
            and self._torch_ae.has_scaling():

            #TODO: nicer
            assert snapshots.dataset is not None

            idx_scaling = list(dict(self._torch_encoder.named_children())).index(Conv1dAutoEncoderChart.SCALING_NORMALIZE)
            with torch.no_grad():
                channeled_data = (self._torch_encoder[:idx_scaling])(snapshots.dataset.snapshot_tensor)
            # compute min / max for each channel (where -2 is the dimension related to the channels)
            n_channels = channeled_data.shape[-2]
            data_min, data_max = np.zeros(n_channels), np.zeros(n_channels)
            for i_channel in range(n_channels):
                current_channel_data = channeled_data.select(-2, i_channel)
                data_min[i_channel] = current_channel_data.min()
                data_max[i_channel] = current_channel_data.max()
            data_scaling = data_max - data_min
            data_min = super().from_numpy(data_min)
            data_scaling = super().from_numpy(data_scaling)
            self._torch_ae.scaling.update_parameters(data_min, data_scaling)
            self._torch_ae.inv_scaling.update_parameters(data_min, data_scaling)
            # del(channeled_data)

            # # sanity check
            # updated_channeled_data = (self._torch_encoder[:idx_scaling+1])(snapshots.dataset.snapshot_tensor)
            # assert (
            #     np.isclose(updated_channeled_data.min().item(), 0.)
            #     and np.isclose(updated_channeled_data.max().item(), 1.)
            # ), 'rescaled data is not in the range [0, 1]'
            # del(updated_channeled_data)

        if not logger is None and training_params.get('track_channels', False):
            # track validation loss of different channels separately
            val_loss_func = essentials['val_loss_func']
            if hasattr(self._torch_encoder, self.LAYER_KEY_UNFLATTEN):
                encoder_unflatten = getattr(self._torch_encoder ,self.LAYER_KEY_UNFLATTEN)
                n_channels = encoder_unflatten.unflattened_size[0]
                channel_names = training_params.get('channel_names', list(range(n_channels)))
                assert len(channel_names) == n_channels
                idx_unflatten = list(dict(self._torch_encoder.named_children())).index(self.LAYER_KEY_UNFLATTEN)
                splitter = self._torch_encoder[:idx_unflatten+1]
                if not isinstance(val_loss_func, HybridLoss):
                    val_loss_func = HybridLoss(val_loss_func)
                # add channel losses with weight 0 to validation loss
                for i_ch, ch_name in enumerate(channel_names):
                    loss_fun = RelativeChannelMseLoss(splitter, i_ch, name='')
                    loss_fun.LOSS_NAME = '{}_{}'.format(loss_fun.LOSS_NAME, ch_name)
                    val_loss_func.append_loss(loss_fun, weight=0.)
            else:
                warnings.warn('Can not track channels since unflatten layer can not be found.')

        return essentials
    
    def map(self, xr):
        mapped_xr = super().map(xr)
        # special case: batch size of 1 shall be retained
        if xr.ndim == 2 and mapped_xr.ndim == 1:
            mapped_xr = mapped_xr[np.newaxis, :]
        return mapped_xr

    def inv_map(self, x):
        mapped_x = super().inv_map(x)
        # special case: batch size of 1 shall be retained
        if x.ndim == 2 and mapped_x.ndim == 1:
            mapped_x = mapped_x[np.newaxis, :]
        return mapped_x

    def from_numpy(self, val: np.ndarray):
        if val.ndim == 1:
            val = val[np.newaxis, ...]
        return super().from_numpy(val)

    def to_numpy(self, val: torch.Tensor):
        val_size = val.size()
        if len(val_size) == 4 and val_size[0] == 1 and val_size[2] == 1:
            # Jacobian with batch size 1
            val = val.squeeze(2).squeeze(0)
        if len(val_size) == 3 and val_size[0] == 1:
            #for result of torch_modules.get_jacobian_jvp
            val = val.squeeze(0)
        elif len(val_size) >= 2 and val_size[-2] == 1:
            # vector with batch size 1
            val = val.squeeze(-2)
        return super().to_numpy(val)

    @staticmethod
    def initialize(module, params):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear, torch.nn.ConvTranspose1d)):
            distribution = params['distribution']
            weight = module.weight
            method_string = 'torch.nn.init.'
            if params['type'] == 'Kaiming_ReLu':
                method_string += 'kaiming_{}_(weight, nonlinearity="relu")'.format(distribution)
            elif params['type'] == 'Kaiming_Sigmoid':
                method_string += 'kaiming_{}_(weight, nonlinearity="sigmoid")'.format(distribution)
            elif params['type'] == 'Xavier':
                method_string += 'xavier_{}_(weight)'.format(distribution)
            eval(method_string)
            if not module.bias is None:
                module.bias.data.fill_(0.)

def _check_or_make_array(value, check_type, length):
    if isinstance(value, (list, tuple, np.ndarray)):
        assert len(value) == length and all(isinstance(val_i, check_type) for val_i in value)
        return value
    else:
        assert isinstance(value, check_type) or value is None
        return length*(value,)
