'''Base classes for loss functions.'''
from typing import List

import numpy as np
import torch


class Loss(torch.nn.Module):
    def __init__(self, name='loss'):
        super().__init__()
        self._loss_item_buffer = dict()
        self.last_avg_losses = dict()
        self.name = None
        if not name is None:
            self.set_name(name)
    
    def avg_loss_over_batches(self):
        avg_losses = dict()
        for loss_key, loss_vals in self._loss_item_buffer.items():
            if len(loss_vals) > 0:
                avg_losses[loss_key] = sum(loss_vals) / len(loss_vals)
            self._loss_item_buffer[loss_key] = []
        self.last_avg_losses = avg_losses

    def summary_dict(self):
        return self.last_avg_losses
    
    def set_name(self, name):
        self._loss_item_buffer = dict()
        self._loss_item_buffer[name] = []
        self.name = name

class MetaLoss(Loss):
    def __init__(self, loss_funcs: List[Loss], meta_name: str, weights: List[float] = None, name='loss'):
        """A loss function based on one or multiple different losses.

        Args:
            loss_funcs (List[Loss]): |Loss| functions the loss is based on
            meta_name (str): the additional identifier to add in the loss
            weights (List[float], optional): weights for the losses. Defaults to None.
            name (str, optional): name of the loss. Might be overwritten. Defaults to 'loss'.
        """
        super().__init__(name=None) # do not set name here
        if isinstance(loss_funcs, Loss):
            loss_funcs = [loss_funcs]
        assert isinstance(loss_funcs, list) and all(isinstance(l, Loss) for l in loss_funcs)
        if weights is None:
            weights = [1.] * len(loss_funcs)
        assert isinstance(weights, list) and all(isinstance(w, float) for w in weights)
        self.loss_funcs = loss_funcs
        self.weights = weights
        self.meta_name = meta_name
        self.set_name(name) # set name here, now that loss_funcs is initialized
    
    def set_name(self, name):
        if len(self.meta_name) > 0:
            name += '_' + self.meta_name
        super().set_name(name)
        for loss_func in self.loss_funcs:
            loss_func.set_name(self.name)
    
    def avg_loss_over_batches(self):
        super().avg_loss_over_batches()
        # compute averages of child losses
        for loss_func in self.loss_funcs:
            loss_func.avg_loss_over_batches()
    
    def summary_dict(self):
        # if len(self.loss_funcs) == 1:
        #     # return summary dict of single child
        #     return self.loss_funcs[0].summary_dict()
        # else:
        # accumulate summaries of all children
        summary = super().summary_dict()
        for loss_fun in self.loss_funcs:
            loss_fun_summary = loss_fun.summary_dict()
            if len(loss_fun_summary) > 0:
                summary.update(loss_fun_summary)
        return summary

    def forward_loss_funcs(self, tensor_in, target_entry):
        total_loss = 0
        for weight, loss_fun in zip(self.weights, self.loss_funcs):
            loss_val = loss_fun(tensor_in, target_entry)
            if not np.isclose(weight, 0.):
                total_loss += weight * loss_val
            else:
                total_loss += weight * loss_val
        self._loss_item_buffer[self.name].append(total_loss.item())
        return total_loss

    def has_loss_named(self, name):
        return any(loss.name == name for loss in self.loss_funcs)
        
    def to(self, device):
        for loss in self.loss_funcs:
            if hasattr(loss, 'to') and callable(loss.to):
                loss.to(device)
