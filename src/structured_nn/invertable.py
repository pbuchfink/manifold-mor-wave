import numpy as np
import torch

from structured_nn.basic import InvertableModule, make_torch_parameter


class Flatten(InvertableModule):
    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size
    
    def _forward_eval(self, tensor_in):
        if self.dim < 0:
            start_dim = self.dim - len(self.unflattened_size) + 1
            end_dim = self.dim
        else:
            start_dim = self.dim
            end_dim = self.dim + len(self.unflattened_size) - 1
        return tensor_in.flatten(start_dim=start_dim, end_dim=end_dim)

    def _forward_inv_eval(self, tensor_in):
        return tensor_in.unflatten(self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        repr_str = super().extra_repr()
        return repr_str + ',dim={},unflattened_size={}'.format(self.dim, self.unflattened_size)


class Scaling(InvertableModule):
    def __init__(self, scaling, shift):
        super().__init__()
        self.shift = make_torch_parameter(shift, requires_grad=False)
        self.scaling = make_torch_parameter(scaling, requires_grad=False)
    
    def update_parameters(self, shift, scaling):
        if not isinstance(shift, torch.Tensor):
            shift = torch.tensor(shift)
        if not isinstance(scaling, torch.Tensor):
            scaling = torch.tensor(scaling)
        self.shift.data = shift
        self.scaling.data = scaling
    
    def _forward_eval(self, torch_in):
        # use broadcasting in the last dimension
        return (torch_in - self.shift) / self.scaling

    def _forward_inv_eval(self, torch_in):
        # use broadcasting in the last dimension
        return torch_in * self.scaling + self.shift

    def extra_repr(self) -> str:
        repr_str = super().extra_repr()
        return repr_str + ',shift.shape={},scaling.shape={}'.format(self.shift.shape, self.scaling.shape)


class Permutation(InvertableModule):
    '''Apply permutation to last tensor dimension.'''
    def __init__(self, permutation):
        if isinstance(permutation, np.ndarray):
            # copy required if used with ray tune as permutation comes from the
            # config-dict and otherwise raises a UserWarning
            permutation = torch.from_numpy(permutation.copy())
        super().__init__()
        self.permutation = make_torch_parameter(permutation, requires_grad=False)
        assert len(set(permutation)) == len(permutation), 'there are duplicate indices'
        self.inv_permutation = torch.argsort(permutation)
    
    def _forward_eval(self, tensor_in):
        tensor_in_shape = tuple(tensor_in.shape)
        assert tensor_in_shape[-1] == len(self.permutation)

        return tensor_in[..., self.permutation]
    
    def _forward_inv_eval(self, tensor_in):
        tensor_in_shape = tuple(tensor_in.shape)
        assert tensor_in_shape[-1] == len(self.inv_permutation)

        return tensor_in[..., self.inv_permutation]
    
    def extra_repr(self) -> str:
        repr_str = super().extra_repr()
        return repr_str + ',len(permutation)={}'.format(len(self.permutation))
