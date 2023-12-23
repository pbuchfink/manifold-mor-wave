'''Helpful additional modules that are used in the Autoencoder charts.'''
from contextlib import contextmanager

import numpy as np
import torch
from structured_nn.invertable import Scaling
from torch.nn import Sequential


class TorchAutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder, scaling=None, inv_scaling=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.scaling = scaling
        self.inv_scaling = inv_scaling

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def is_valid(self):
        """Check if scaling and inv_scaling are valid."""
        if not self.has_scaling():
            return True
        # check scaling
        assert isinstance(self.scaling, (Standardization, Scaling))
        assert isinstance(self.inv_scaling, (InvStandardization, Scaling))
        if isinstance(self.encoder, Sequential):
            assert self.scaling in self.encoder.modules()
        if isinstance(self.decoder, Sequential):
            assert self.inv_scaling in self.decoder.modules()
        return (torch.allclose(self.scaling.shift, self.inv_scaling.shift)
                and torch.allclose(self.scaling.scaling, self.inv_scaling.scaling))

    def has_scaling(self):
        return not(self.scaling is None and self.inv_scaling is None)


class InverseCELU(torch.nn.Module):
    def __init__(self, alpha):
        self._alpha = alpha
        super().__init__()

    def forward(self, x):
        return torch.max(0, x) + torch.min(0, self._alpha*(torch.log(x/self._alpha) + 1))


# self-implemented version of unflatten (which was introduced in 1.7.0)
# does not use tensor.unflatten on purpose
class Unflatten(torch.nn.Module):
    def __init__(self, dim, unflattened_size):
        super().__init__()
        assert isinstance(dim, int)
        assert isinstance(unflattened_size, (tuple, list)) \
            and all(isinstance(s, (np.integer, int)) for s in unflattened_size)
        self.dim = dim
        self.unflattened_size = tuple(unflattened_size)

    def forward(self, tensor_in):
        assert isinstance(tensor_in, torch.Tensor)
        tensor_in_shape = tuple(tensor_in.shape)
        assert abs(self.dim) < len(tensor_in_shape) \
            and tensor_in_shape[self.dim] == np.prod(self.unflattened_size)

        if self.dim == -1:
            tensor_in_shape = tensor_in_shape[:self.dim] + self.unflattened_size
        else:
            tensor_in_shape = tensor_in_shape[:self.dim] + self.unflattened_size + tensor_in_shape[self.dim+1:]
        return torch.reshape(tensor_in, tensor_in_shape)


class Standardization(torch.nn.Module):
    def __init__(self, shift=0., scaling=1.):
        super().__init__()
        self.shift = torch.nn.parameter.Parameter(torch.tensor(shift), requires_grad=False)
        self.scaling = torch.nn.parameter.Parameter(torch.tensor(scaling), requires_grad=False)
        self.update_parameters(shift, scaling)

    def forward(self, torch_in):
        # use broadcasting in the last dimension
        return ((torch_in.transpose(-2, -1) - self.shift) / self.scaling).transpose(-2, -1)

    def update_parameters(self, shift, scaling):
        if not isinstance(shift, torch.Tensor):
            shift = self.shift.new_tensor(shift)
        if not isinstance(scaling, torch.Tensor):
            scaling = self.scaling.new_tensor(scaling)
        self.shift.data = torch.nn.parameter.Parameter(shift, requires_grad=False)
        self.scaling.data = torch.nn.parameter.Parameter(scaling, requires_grad=False)

    def _apply(self, fn):
        super(Standardization, self)._apply(fn)
        self.shift = fn(self.shift)
        self.scaling = fn(self.scaling)
        return self


class InvStandardization(torch.nn.Module):
    def __init__(self, shift=0., scaling=1.):
        super().__init__()
        self.shift = torch.nn.parameter.Parameter(torch.tensor(shift), requires_grad=False)
        self.scaling = torch.nn.parameter.Parameter(torch.tensor(scaling), requires_grad=False)
        self.update_parameters(shift, scaling)

    def forward(self, torch_in):
        # use broadcasting in the last dimension
        return ((torch_in.transpose(-2, -1) * self.scaling) + self.shift).transpose(-2, -1)

    def update_parameters(self, shift, scaling):
        if not isinstance(shift, torch.Tensor):
            shift = self.shift.new_tensor(shift)
        if not isinstance(scaling, torch.Tensor):
            scaling = self.scaling.new_tensor(scaling)
        self.shift.data = torch.nn.parameter.Parameter(shift, requires_grad=False)
        self.scaling.data = torch.nn.parameter.Parameter(scaling, requires_grad=False)

    def _apply(self, fn):
        super(InvStandardization, self)._apply(fn)
        self.shift = fn(self.shift)
        self.scaling = fn(self.scaling)
        return self


class ShiftModule(torch.nn.Module):
    def __init__(self, shift):
        if isinstance(shift, np.ndarray):
            shift = torch.from_numpy(shift)
        super().__init__()
        self.shift = torch.nn.parameter.Parameter(shift, requires_grad=False)

    def forward(self, torch_in):
        return torch_in + self.shift

    def _apply(self, fn):
        super(ShiftModule, self)._apply(fn)
        self.shift = fn(self.shift)
        return self


class Permutation(torch.nn.Module):
    '''Apply permutation to last tensor dimension.'''
    def __init__(self, permutation):
        if isinstance(permutation, np.ndarray):
            # copy required if used with ray tune as permutation comes from the
            # config-dict and otherwise raises a UserWarning
            permutation = torch.from_numpy(permutation.copy())
        super().__init__()
        self.permutation = torch.nn.parameter.Parameter(permutation, requires_grad=False)
        assert len(set(permutation)) == len(permutation), 'there are duplicate indices'
    
    def forward(self, tensor_in):
        assert isinstance(tensor_in, torch.Tensor)
        tensor_in_shape = tuple(tensor_in.shape)
        assert tensor_in_shape[-1] == len(self.permutation)

        return (tensor_in.transpose(-1, 0)[self.permutation]).transpose(-1, 0)

    def _apply(self, fn):
        super(Permutation, self)._apply(fn)
        self.permutation = fn(self.permutation)
        return self

class InversePermutation(Permutation):
    def __init__(self, permutation):
        if isinstance(permutation, np.ndarray):
            permutation = torch.from_numpy(permutation.copy())
        assert len(permutation.shape) == 1
        # init with inverse permutation
        super().__init__(torch.argsort(permutation))

def get_jacobian(net, x, noutputs, create_graph=False):
    ninputs = x.shape[-1]
    with eval_mode(net):
        if ninputs < noutputs:
            jac = get_jacobian_jvp(net, x, create_graph=create_graph)
        else:
            # from https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa.js
            if x.ndim > 1:
                x = x.squeeze(1)
            if x.ndim > 0:
                raise NotImplementedError('Only implemented for non-batched evaluation.')
            n = x.size()[0]
            x = x.repeat(noutputs, 1)
            x.requires_grad_(True)
            y = net(x)
            y.backward(torch.eye(noutputs, device=x.device, dtype=x.dtype), create_graph=create_graph)
            jac = x.grad.data
    return jac

def get_jacobian_jvp(net: torch.nn.Module, x: torch.Tensor, create_graph: bool):
    assert not net.training, 'not sure what happens with double-backward-trick in jvp and batch_norm'
    ninputs = x.shape[-1]
    reshape_jacobian = True
    if x.ndim == 1:
        x = x[None, :]
        reshape_jacobian = False
    else:
        assert x.ndim == 2
    len_x = len(x)
    xx = x.repeat_interleave(ninputs, dim=0)
    v = torch.eye(ninputs, dtype=x.dtype, device=x.device)
    vv = v.repeat(len_x, 1)
    with torch.enable_grad():
        jacs = torch.autograd.functional.jvp(net, xx, vv, create_graph=create_graph)[1]
    if reshape_jacobian:
        noutputs = jacs.shape[-1]
        jacs = jacs.reshape(len_x, ninputs, noutputs)
    return jacs.transpose(-1, -2)

@contextmanager
def eval_mode(net):
    net_training = net.training
    net.eval()
    try:
        yield net_training
    finally:
        if net_training:
            net.train()
