'''All common losses used for training of charts in pyTorch
which yet means all Autoencoder charts.'''
from abc import ABC, abstractmethod
from typing import Iterable, List

import torch
from manifold_mor.chart.torch_modules import (eval_mode, get_jacobian)
from structured_nn.utils.loss import Loss, MetaLoss

LOSS_MSE = 'mse'
LOSS_MSE_SCALED = 'mse_scaled'

LOSS_VALIDATION = 'loss_val'
LOSS_TRAINING = 'loss_train'
LOSS_EVALUATION = 'loss_eval'
LOSS_DATA = 'data'

def loss_from_params(params, model, name='loss'):
    loss_fun = params['loss_fun']
    device = params.get('device', 'cpu')
    if loss_fun == LOSS_MSE_SCALED:
        scaling_layer = params['loss_params']['scaling_layer']
        loss_func = ScaledLoss(scaling_layer, name=name)
    else:
        raise ValueError('Unknown parameter loss_fun={}'.format(loss_fun))
    if not device == 'cpu':
        loss_func.to(device)
    return loss_func


class MSELoss(Loss):
    def __init__(self, name='loss'):
        super().__init__(name=name)
        self.mse_loss = torch.nn.MSELoss()
    
    def set_name(self, name):
        super().set_name(name + '_' + LOSS_MSE)
    
    def forward(self, tensor_in, target):
        loss = self.mse_loss(tensor_in, target)
        self._loss_item_buffer[self.name].append(loss.item())
        return loss


class ScaledLoss(Loss):
    def __init__(self, scaling_layer, name='loss'):
        super().__init__(name=name)
        self.scaling_layer = scaling_layer
        self.mse_loss = torch.nn.MSELoss()
    
    def set_name(self, name):
        super().set_name(name + '_' + LOSS_MSE_SCALED)
    
    def forward(self, tensor_in, target):
        loss = self.mse_loss(self.scaling_layer(tensor_in), self.scaling_layer(target))
        self._loss_item_buffer[self.name].append(loss.item())
        return loss


class DictLoss(ABC):
    @abstractmethod
    def required_keys(self) -> Iterable[str]:
        ...


class DataLoss(DictLoss,MetaLoss):
    def __init__(self, loss_funcs: List[Loss], meta_name_suffix: str = '', weights: List[float] = None, name='loss'):
        meta_name = LOSS_DATA
        if len(meta_name_suffix) > 0:
            meta_name += '_' + meta_name_suffix
        super().__init__(loss_funcs, meta_name, weights=weights, name=name)

    @classmethod
    def from_loss_params(cls, params, model, name='loss'):
        return cls(loss_from_params(params, model, name=name), name=name)
    
    def forward(self, dict_tensor_in, dict_target):
        return self.forward_loss_funcs(dict_tensor_in['data'], dict_target['data'])
    
    def required_keys(self) -> Iterable[str]:
        return ['data']


class HybridLoss(DictLoss,MetaLoss):
    def __init__(self, loss_funcs: List[Loss], weights: List[float] = None, name='loss'):
        if isinstance(loss_funcs, tuple):
            loss_funcs = list(loss_funcs)
        if weights is None:
            weights = [1.] * len(loss_funcs)
        super().__init__(loss_funcs, '', weights=weights, name=name)

    def forward(self, tensor_in, target):
        return self.forward_loss_funcs(tensor_in, target)
    
    def append_loss(self, loss, weight=1.):
        assert isinstance(loss, Loss)
        self.loss_funcs.append(loss)
        self.weights.append(weight)
        loss.set_name(self.name)
    
    def summary_dict(self):
        summary = super().summary_dict()
        # accumulate summary from child losses
        for loss_fun, weight in zip(self.loss_funcs, self.weights):
            summary['{}_weight'.format(loss_fun.name)] = weight
        return summary
    
    def required_keys(self) -> Iterable[str]:
        return set(k for l in self.loss_funcs if isinstance(l, DictLoss) for k in l.required_keys())


class IndexedMseLoss(Loss):
    LOSS_NAME = 'mse_rel'
    def __init__(self, index_start, index_end, name='loss'):
        super().__init__(name=name)
        self.mse_loss = torch.nn.MSELoss()
        self.index_start = index_start
        self.index_end = index_end

    def set_name(self, name):
        super().set_name('{}_{}'.format(name, self.LOSS_NAME))

    def forward(self, tensor_in, target):
        sliced_tensor_in = tensor_in[:, self.index_start:self.index_end]
        sliced_target = target[:, self.index_start:self.index_end]
        loss = self.mse_loss(sliced_tensor_in, sliced_target) / torch.mean(sliced_target**2)
        self._loss_item_buffer[self.name].append(loss.item())
        return loss


class RelativeChannelMseLoss(Loss):
    LOSS_NAME = 'mse_rel'
    def __init__(self, splitter, i_channel, name='loss'):
        super().__init__(name=name)
        self.mse_loss = torch.nn.MSELoss()
        self.splitter = splitter
        self.i_channel = i_channel

    def set_name(self, name):
        super().set_name('{}_{}'.format(name, self.LOSS_NAME))

    def forward(self, dict_tensor_in, dict_target):
        sliced_tensor_in = self.splitter(dict_tensor_in['data'])[:, self.i_channel]
        sliced_target = self.splitter(dict_target['data'])[:, self.i_channel]
        loss = self.mse_loss(sliced_tensor_in, sliced_target) / torch.mean(sliced_target**2)
        self._loss_item_buffer[self.name].append(loss.item())
        return loss


class WeakSymplecticLoss(Loss):
    def __init__(self, net, ambient_dim, dim, name='loss'):
        assert dim % 2 == 0, 'dim has to be even dimensional'
        assert ambient_dim % 2 == 0, 'ambient_dim has to be even dimensional'
        super(WeakSymplecticLoss, self).__init__(name=name)
        self.net = net
        J2 = lambda m: torch.vstack([
            torch.hstack([torch.zeros((m, m)), torch.eye(m)]),
            torch.hstack([-torch.eye(m), torch.zeros((m, m))])
        ])
        self.J2_DIM = J2(dim//2)
        self.ambient_dim = ambient_dim

    def set_name(self, name):
        super().set_name(name + '_weak_sympl')

    def forward(self, dict_tensor_in, dict_target):
        # compute jacs if not already present (e.g. by call to JacobianBasedTangentLoss)
        jacs = dict_target.get('jacs', None)
        if jacs is None:
            # get jacobians
            with eval_mode(self.net) as net_training:
                xr = self.net.encoder(dict_target['data'])
                jacs = get_jacobian(self.net.decoder, xr, self.ambient_dim, create_graph=net_training)
            dict_target['jacs'] = jacs

        # compute canonical symplectic product
        N = self.ambient_dim // 2
        sympl = torch.bmm(jacs[:, :N, :].transpose(1,2), jacs[:, N:, :])
        sympl -= torch.bmm(jacs[:, N:, :].transpose(1,2), jacs[:, :N, :])
        # compute different to canonical Poisson matrix
        sympl -= self.J2_DIM
        loss = torch.mean(sympl**2)

        # # debugging code
        # J2 = lambda m: torch.vstack([
        #     torch.hstack([torch.zeros((m, m)), torch.eye(m)]),
        #     torch.hstack([-torch.eye(m), torch.zeros((m, m))])
        # ])
        # J2N = J2(N)
        # diffs = []
        # for jac, sympl_prod in zip(jacs, sympl):
        #     diffs.append(
        #         abs(sympl_prod + self.J2_DIM - jac.T @ J2N @ jac).max()
        #     )
        # print('diffs, max: {}, min: {}'.format(max(diffs), min(diffs)))

        self._loss_item_buffer[self.name].append(loss.item())
        return loss

