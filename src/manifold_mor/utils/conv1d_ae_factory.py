'''A module to build different architectures of a Deep Convolutional Autoencoder.'''
from typing import List
import numpy as np
import torch
from manifold_mor.utils.logging import Logger
from ray import tune


class NanStopper(tune.Stopper):
    def __call__(self, trial_id, result):
        return np.isnan(result[Logger.METRIC_LOSS_VALIDATION]) or np.isnan(result[Logger.METRIC_LOSS_TRAINING])
    
    def stop_all(self):
        return False

class Conv1dAeFactory:
    TYPE_PADDING_HALF = 'HALF_PADDING'
    def __init__(
        self,
        in_features: int,
        in_channels: int,
        out_features: int,
        kernel_sizes: List[int],
        type_padding: str,
        min_num_convs: int,
        max_num_convs: int,
        min_log2_red_fac: float,
        max_log2_red_fac: float = None,
    ) -> None:
        # sanity check
        log2_in_channels = np.log2(in_channels)
        assert np.isclose(log2_in_channels, int(log2_in_channels)), 'only powers of two are allowed'
        
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_features = out_features
        self.kernel_sizes = kernel_sizes
        self.type_padding = type_padding
        self.min_num_convs = min_num_convs
        self.max_num_convs = max_num_convs
        self.min_log2_red_fac = min_log2_red_fac
        self.max_log2_red_fac = max_log2_red_fac
        self.strides = self.build_stride_library()
    
    def get_config_entries(self):
        return {
            "in_features": self.in_features,
            "in_channels": self.in_channels,
            "out_features": self.out_features,
            "kernel_size": tune.choice(self.kernel_sizes),
            "padding": tune.sample_from(self.padding_from_spec),
            "num_conv": tune.sample_from(self.num_conv_from_spec),
            "stride": tune.sample_from(self.stride_from_spec),
            "conv_lengths": tune.sample_from(self.conv_lengths_from_spec),
            "log2_reduction_factor": tune.sample_from(self.log2_reduction_factor_from_spec),
            "conv_channels": tune.sample_from(self.conv_channels_from_spec),
            "batch_norm": tune.sample_from(self.batch_norm_from_activation),
            "num_fully_layers": tune.choice([1,2,3]),
            "fully_sizes": tune.sample_from(self.fully_sizes_from_spec),
            "fully_activation": tune.sample_from(self.fully_activation_from_spec),
            "fully_batch_norm": tune.sample_from(self.fully_batch_norm_from_spec),
            "initialization": {
                'type': tune.sample_from(self.initialization_type_from_spec),
                'distribution': tune.choice(['uniform', 'normal']),
            },
        }
    
    def paddings_from_kernel_size(self, kernel_size):
        if self.type_padding == self.TYPE_PADDING_HALF:
            return (kernel_size // 2,)
        else:
            raise NotImplementedError('Unkown type_paddings_from_kernel_size: {}'.format(self.type_paddings_from_kernel_size))

    def compute_conv_lengths(self, kernel_size, padding, stride):
        n_conv_layers = len(stride)
        conv_lengths = [None] * (n_conv_layers + 1)
        conv_lengths[-1] = self.in_features
        for i_l in reversed(range(n_conv_layers)):
            conv_lengths[i_l] = int((conv_lengths[i_l+1] + 2*padding - kernel_size) / stride[i_l] + 1)
        return conv_lengths

    def build_stride_library(self):
        assert isinstance(self.kernel_sizes, (list, tuple, np.ndarray))
        assert callable(self.paddings_from_kernel_size) \
            and isinstance(self.paddings_from_kernel_size(self.kernel_sizes[0]), (list, tuple))
        strides = dict()
        for kernel_size in self.kernel_sizes:
            paddings = self.paddings_from_kernel_size(kernel_size)
            for padding in paddings:
                current_strides = dict()
                for n_convs in range(self.min_num_convs, self.max_num_convs+1):
                    valid_strides = []
                    level_stride_heap = [(1, 2*np.ones((n_convs,), dtype=int))]
                    while len(level_stride_heap) > 0:
                        level, stride = level_stride_heap.pop()
                        # check if stride values are valid
                        if self.compute_conv_lengths(kernel_size, padding, stride)[0] > 1:
                            # add to valid strides
                            valid_strides.append(stride)
                            # add children to the heap
                            # each child replaces 2**level with 2**(level+1) starting from index zero
                            i_replace = 0
                            while i_replace < n_convs and stride[i_replace] == 2**level:
                                child = stride.copy()
                                child[:i_replace+1] = 2**(level+1)
                                level_stride_heap.append((level+1, child))
                                i_replace += 1
                    if len(valid_strides) > 0:
                        current_strides[n_convs] = valid_strides
                assert len(current_strides) > 0, \
                    'no valid strides available for the pairing kernel_size: {}, padding: {}'.format(kernel_size, padding)
                strides[(padding, kernel_size)] = current_strides
        return strides

    def padding_from_spec(self, spec):
        return np.random.choice(self.paddings_from_kernel_size(spec['config']['kernel_size']))

    def num_conv_from_spec(self, spec):
        current_strides = self.strides[(
            spec['config']['padding'],
            spec['config']['kernel_size']
        )]
        return np.random.choice(list(current_strides.keys()))

    def stride_from_spec(self, spec):
        current_strides = self.strides[(
            spec['config']['padding'],
            spec['config']['kernel_size']
        )]
        valid_strides = current_strides[spec['config']['num_conv']]
        return valid_strides[np.random.choice(len(valid_strides))]

    def conv_lengths_from_spec(self, spec):
        return self.compute_conv_lengths(
            spec['config']['kernel_size'],
            spec['config']['padding'],
            spec['config']['stride'],
        )

    def log2_reduction_factor_from_spec(self, spec):
        '''The reduction factor is a meassure for the ratio
            number of inputs / number of outputs
        of the convultion part of the autoencoder.
        The reduction factor is computed such that the output of the convolutional part of
        the autoencoder is bigger than or equal to out_features.'''
        in_features = spec['config']['in_features']
        in_channels = spec['config']['in_channels']
        out_features = max(
            spec['config']['out_features'],
            spec['config']['conv_lengths'][0]
        )
        corrected_max_log2_red_fac = min(
            np.log2(in_features * in_channels/out_features),
            self.max_log2_red_fac if self.max_log2_red_fac else 0
        )
        return np.random.uniform(low=self.min_log2_red_fac, high=corrected_max_log2_red_fac)

    def compute_conv_channels(self, log2_reduction_factor, n_conv):
        log2_out_channel = np.ceil(np.log2(self.in_features * self.in_channels/self.out_features) - log2_reduction_factor)
        np_conv_channels = 2**np.linspace(np.log2(self.in_channels), log2_out_channel, num=n_conv+1, dtype=int)[::-1]
        return tuple(int(conv_channel) for conv_channel in np_conv_channels)

    def conv_channels_from_spec(self, spec):
        '''Computes convolution channels in the convolution part of the autoencoder.
        It is constructed such that the general meassure "reduction_factor"
        is used to decide on the number of convolution channels.'''
        return self.compute_conv_channels(
            spec['config']['log2_reduction_factor'],
            spec['config']['num_conv']
        )

    def batch_norm_from_activation(self, spec):
        # run sigmoid only WITH batch normalization
        # otherwise the experiment give a lot of nans
        if isinstance(spec['config']['activation'], torch.nn.Sigmoid):
            return True
        else:
            return np.random.choice([False, True])

    def fully_sizes_from_spec(self, spec):
        fully_layers = spec['config']['num_fully_layers']
        assert fully_layers >= 1
        np_fully_sizes = np.linspace(
            spec['config']['out_features'],
            spec['config']['conv_lengths'][0] * spec['config']['conv_channels'][0],
            fully_layers+1,
            dtype=int
        )
        return tuple(np_fully_sizes)

    def fully_activation_from_spec(self, spec):
        return spec['config']['activation']

    def fully_batch_norm_from_spec(self, spec):
        return spec['config']['batch_norm']

    def initialization_type_from_spec(self, spec):
        activation = spec['config']['activation']
        if isinstance(activation, torch.nn.ELU):
            return np.random.choice(('Kaiming_ReLu', 'Xavier'))
        elif isinstance(activation, torch.nn.Sigmoid):
            return 'Kaiming_Sigmoid'
        else:
            raise ValueError('Initialization not defined for activation of type: {}'.format(type(activation)))
