from abc import ABC, abstractmethod
import torch
from copy import copy

class InvertableModule(torch.nn.Module, ABC):
    '''A super class for invertable modules.'''
    def __init__(self) -> None:
        super().__init__()
        self.inverse_mode = make_torch_parameter(False, requires_grad=False)
    
    def forward(self, *tensor_in):
        '''Evaluates the module depending on inverse_mode.'''
        if self.inverse_mode:
            return self._forward_inv_eval(*tensor_in)
        else:
            return self._forward_eval(*tensor_in)
    
    def forward_inverse(self, *tensor_in):
        '''Evaluates the inverse of the module depending on inverse_mode.'''
        if self.inverse_mode:
            return self._forward_eval(*tensor_in)
        else:
            return self._forward_inv_eval(*tensor_in)
    
    @abstractmethod
    def _forward_eval(self, *tensor_in):
        '''Evaluation of the module.'''
        ...
    
    @abstractmethod
    def _forward_inv_eval(self, *tensor_in):
        '''Inverse evaluation of the module.'''
        ...
    
    def invert(self):
        inv_module = copy(self)
        # copy parameters and delete inverse_mode such that it is no longer a shared parameter
        inv_module._parameters = self._parameters.copy()
        del(inv_module._parameters['inverse_mode'])
        inv_module.inverse_mode = make_torch_parameter(not self.inverse_mode, requires_grad=False)
        return inv_module

    def extra_repr(self) -> str:
        return 'inverse_mode={}'.format(self.inverse_mode)


def make_torch_parameter(input, requires_grad=True):
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(input, torch.nn.Parameter):
        input = torch.nn.Parameter(input, requires_grad=requires_grad)
    return input
