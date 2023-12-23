"""All classes for Autoencoders (in general).
In our framework, the DCA is interpreted as a special chart.
"""
import copy
import os
from time import time
from typing import Dict, Iterable, List
from warnings import warn

import numpy as np
import torch
from manifold_mor.chart.basic import (IterativelyTrainableCallback,
                                      IterativelyTrainableChart,
                                      ReportIterativelyTrainingCallback,
                                      TorchEvaluatable)
from manifold_mor.chart.torch_losses import LOSS_VALIDATION, DataLoss
from manifold_mor.chart.torch_modules import (TorchAutoEncoder, eval_mode,
                                              get_jacobian)
from manifold_mor.fields.vector_reduced import (
    EncoderVectorFieldProjector, MoorePenroseVectorFieldProjector,
    VectorFieldProjector, WeightedMoorePenroseVectorFieldProjector)
from manifold_mor.models.basic import Model
from manifold_mor.mor.snapshots import Snapshots, SnapshotDataset
from manifold_mor.utils.logging import Logger

BATCH_SIZE = 35

def execute_batched(func):
    '''Decorator to execute a function with batched tensors'''
    def batched_func(self, *tensors: np.ndarray) -> np.ndarray:
        results = []
        shape = tensors[0].shape
        assert all(t.shape == shape for t in tensors), 'execute_batched assumes all tensor to be of the same shape'
        
        if tensors[0].ndim == 1:
            return func(self, *tensors)
        else:
            idx_processed = 0
            while idx_processed < shape[0]:
                results.append(func(self, *[t[slice(idx_processed, idx_processed+BATCH_SIZE)] for t in tensors]))
                idx_processed += BATCH_SIZE
            return np.vstack(results)
    return batched_func


class AutoEncoderChart(IterativelyTrainableChart, TorchEvaluatable):
    _default_learning_rate = 1e-4
    _default_loss_func = torch.nn.MSELoss()
    _default_n_epochs = 100
    _torch_dtype = np.float32
    _default_batch_size = 1
    _default_shuffle = True
    _default_percentage_validation = .1
    _default_n_epochs_early_stop = None
    SHIFT_ZEROS = 'zeros'
    SHIFT_INITIAL_VALUE = 'initial_value'
    SHIFT_INITIAL_VALUE_AFTER_TRAINING = 'initial_value_after_training'  # method as proposed by Lee & Carlberg 2020

    def __init__(
        self,
        torch_ae: TorchAutoEncoder,
        is_trained: bool = False,
        verbose: bool = True,
        shift: str = SHIFT_INITIAL_VALUE_AFTER_TRAINING,
    ):
        super().__init__(is_trained=is_trained, verbose=verbose)
        self._torch_encoder = torch_ae.encoder
        self._torch_decoder = torch_ae.decoder
        self._torch_ae = torch_ae
        self.device = 'cpu'
        self.shift = shift
        self._shift_torch = None
        self._shift_non_zero = False

    def to(self, device):
        self.device = device
        self._torch_ae = self._torch_ae.to(self.device)

    def update_shift(self, training_finished=False):
        if (self.shift == self.SHIFT_ZEROS
            or (self.shift == self.SHIFT_INITIAL_VALUE_AFTER_TRAINING and not training_finished)):
            self._shift_torch = torch.zeros(self.ambient_dim)
            self._shift_non_zero = False
        elif (self.shift == self.SHIFT_INITIAL_VALUE
              or (self.shift == self.SHIFT_INITIAL_VALUE_AFTER_TRAINING and training_finished)):
            self._shift_torch = self._torch_ae(torch.zeros((1, self.ambient_dim)))[0]
            self._shift_non_zero = True
        else:
            raise NotImplementedError('Unknown shift: {}'.format(self.shift))

    def train_setup(
        self,
        snapshots: Snapshots,
        training_params: dict,
        logger: Logger,
        model: Model,
        callbacks: List[IterativelyTrainableCallback],
        essentials: dict,
    ) -> dict:
        dataset = snapshots.dataset or SnapshotDataset(snapshots, device=self.device)
        if dataset.snapshot_tensor.device != self.device:
            dataset.snapshot_tensor.to(self.device)
        if (dataset.rhs_snapshot_tensor is not None
            and dataset.rhs_snapshot_tensor.device != self.device):
            dataset.rhs_snapshot_tensor.to(self.device)

        torch_ae = self._torch_ae
        torch_ae.train()

        params = training_params
        n_epochs = params.get('n_epochs', self._default_n_epochs)
        loss_func = params.get('loss_func', self._default_loss_func)
        batch_size = params.get('batch_size', self._default_batch_size)
        val_batch_size = params.get('val_batch_size', batch_size)
        shuffle = params.get('shuffle', self._default_shuffle)
        n_epochs_early_stop = params.get('n_epochs_early_stop', self._default_n_epochs_early_stop)
        percentage_validation = params.get('percentage_validation', self._default_percentage_validation)

        if any(isinstance(c, EarlyStopper) for c in callbacks):
            assert isinstance(callbacks[0], EarlyStopper) \
                and isinstance(callbacks[-1], EarlyStopperReportSummaryDict)
        elif n_epochs_early_stop:
            early_stopper = EarlyStopper(
                self._torch_ae,
                n_epochs_early_stop,
                n_epochs,
                synced_chart = essentials.get("synced_ae_chart", None)
            )
            early_stopper_report = EarlyStopperReportSummaryDict(early_stopper)
            callbacks.insert(0, early_stopper)
            callbacks.append(early_stopper_report)

        # split training and validation data
        assert 0 <= percentage_validation <= 1
        n_data = len(dataset)
        n_val_data = int(n_data * percentage_validation)
        n_train_data = n_data - n_val_data
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [n_train_data, n_val_data]
        )
        if val_batch_size < 0:
            # use one batch to compute validation loss (which is faster)
            val_batch_size = len(val_subset)
        if 'loader_num_workers' in params.keys() and not 'cuda' in self.device:
            # do not use num_workers > 0 for gpu data
            # "Otherwise multiple CUDA contexts will be initialized yielding [an] error."
            num_workers = params['loader_num_workers']
        else:
            num_workers = 0 #torch default value
        if len(train_subset) > 0:
            essentials['train_loader'] = torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                generator=torch.Generator(self.device) if shuffle else None
            )
        else:
            if n_epochs > 0:
                warn('Training runs multiple iterations without optimizing!')
            essentials['train_loader'] = None

        if len(val_subset) > 0:
            essentials['val_loader'] = torch.utils.data.DataLoader(
                val_subset,
                batch_size=val_batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                generator=torch.Generator(self.device) if shuffle else None
            )
        else:
            essentials['val_loader'] = None

        if not logger is None:
            callbacks.append(ReportIterativelyTrainingCallback())
            logger.log_metrics({
                'train_batch_size': batch_size,
                'val_batch_size': val_batch_size,
            })
        
        if not 'train_loss_func' in essentials.keys():
            essentials['train_loss_func'] = DataLoss([loss_func])

        if not 'val_loss_func' in essentials.keys():
            essentials['val_loss_func'] = DataLoss([loss_func])

        self.train_loss_func_required_keys = essentials['train_loss_func'].required_keys()
        self.val_loss_func_required_keys = essentials['val_loss_func'].required_keys()

        if not self.device == 'cpu':
            train_loss_func = essentials['train_loss_func']
            if hasattr(train_loss_func, 'to') and callable(train_loss_func.to):
                train_loss_func.to(self.device)

            val_loss_func = essentials['val_loss_func']
            if hasattr(val_loss_func, 'to') and callable(val_loss_func.to):
                val_loss_func.to(self.device)

        if not 'optimizer' in essentials.keys():
            learning_rate = params['learning_rate'] if 'learning_rate' in params.keys() else self._default_learning_rate
            optim_params = {
                'weight_decay': params.get('weight_decay', 0),
                'eps': params.get('adam_eps', 1e-8),
            }
            essentials['optimizer'] = torch.optim.Adam(torch_ae.parameters(), lr=learning_rate, **optim_params)

        if not 'gradient_clipping' in essentials.keys():
            essentials['gradient_clipping'] = None
        
        # update shift
        self.update_shift()

        return essentials
    
    def _compute_ae_batch(
        self,
        batch: Dict[str, torch.Tensor],
        required_keys: Iterable[str],
        mus: Iterable[dict],
        torch_model: Model,
    ):
        ae_batch = dict()
        # compute ae_x together with tangents, if required
        if 'tangents' in required_keys:
            # enable grad, otherwise jvp does not work
            with torch.enable_grad():
                # set net temporarily to evaluation mode to avoid (possible) problems with double-backward-trick and batch-norm
                with eval_mode(self._torch_ae) as net_training:
                    ae_x, ae_tan = torch.autograd.functional.jvp(self._torch_ae, batch['data'], batch['tangents'], create_graph=net_training)
            # shift does not incluence ae_tan as jacobian is identity
            if self._shift_non_zero:
                ae_x -= self._shift_torch
            ae_batch.update({'data': ae_x, 'tangents': ae_tan})

        if 'data' in required_keys and 'data' not in ae_batch.keys():
            ae_x = self.project_x_torch(batch['data'])
            ae_batch = {'data': ae_x}

        if 'red_tangents' in required_keys:
            eval_tensor = ae_batch.get('data', self.project_x_torch(batch['data']))
            # compute rhs on reduced manifold
            mu_idxs = batch['mu_idx']
            # group snapshots by mu (to comptue rhs in batches)
            unique_mu_idxs = torch.unique(mu_idxs)
            rhs = torch.empty_like(batch['tangents'])
            for unique_mu_idx in unique_mu_idxs:
                mask = (mu_idxs == unique_mu_idx)
                torch_model.set_mu(mus[unique_mu_idx])
                rhs[mask] = torch_model.vector_field.eval(eval_tensor[mask])
            # save tangets on trained maniold
            batch['red_tangents'] = rhs
            # enable grad, otherwise jvp does not work
            with torch.enable_grad():
                # set net temporarily to evaluation mode to avoid (possible) problems with double-backward-trick and batch-norm
                with eval_mode(self._torch_ae) as net_training:
                    _, ae_red_tan = torch.autograd.functional.jvp(self._torch_ae, eval_tensor, rhs, create_graph=net_training)
            ae_batch['red_tangents'] = ae_red_tan

        return ae_batch
    
    def train_step(self, essentials: dict, summary: dict, logger: Logger = None):
        i_epoch = summary['epoch']
        torch_ae = self._torch_ae
        optimizer = essentials['optimizer']
        gradient_clipping = essentials['gradient_clipping']

        # update shift (with grad) if necessary
        if self._shift_non_zero:
            self.update_shift()
            assert self._shift_torch.requires_grad == True

        train_loader = essentials['train_loader']
        train_loss_func = essentials['train_loss_func']
        if i_epoch >= 0 and not train_loader is None:
            time_train_start = time()
            torch_ae.train()
            for train_batch in train_loader:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                ae_train_batch = self._compute_ae_batch(
                    train_batch,
                    self.train_loss_func_required_keys,
                    essentials.get('mus', None),
                    essentials.get('torch_model', None),
                )
                loss = train_loss_func(ae_train_batch, train_batch)
                loss.backward()
                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(torch_ae.parameters(), gradient_clipping)
                optimizer.step()

                # update shift if necessary
                if self._shift_non_zero:
                    self.update_shift()
            train_loss_func.avg_loss_over_batches()
            time_train = time() - time_train_start
            summary.update({
                'time_train_epoch_in_s': time_train,
                'time_train_epoch_in_s_avg_per_batch': time_train / len(train_loader),
            })
        summary.update(train_loss_func.summary_dict())

        # update shift (without grad) if necessary
        if self._shift_non_zero:
            with torch.no_grad():
                self.update_shift()
                assert self._shift_torch.requires_grad == False

        val_loader = essentials['val_loader']
        val_loss_func = essentials['val_loss_func']
        if not val_loader is None:
            time_val_start = time()
            torch_ae.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    ae_val_batch = self._compute_ae_batch(
                        val_batch,
                        self.val_loss_func_required_keys,
                        essentials.get('mus', None),
                        essentials.get('torch_model', None),
                    )
                    _ = val_loss_func(ae_val_batch, val_batch)
            val_loss_func.avg_loss_over_batches()
            time_val = time() - time_val_start
            summary.update({
                'time_val_epoch_in_s': time_val,
                'time_val_epoch_in_s_avg_per_batch': time_val / len(val_loader),
            })
        summary.update(val_loss_func.summary_dict())

    def train_finished(self, essentials: dict, summary: dict, logger: Logger):
        self.is_trained = True
        self._torch_ae.eval()
        self._torch_ae.to('cpu')
        with torch.no_grad():
            self.update_shift(training_finished=True)
        assert self._shift_torch.requires_grad == False

    def load_state_dict_torch_ae(self, state_dict: dict):
        self._torch_ae.load_state_dict(state_dict)
        with torch.no_grad():
            self.update_shift()

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        essentials: dict,
        summary: dict,
        checkpoint_name_prefix: str = None,
    ):
        checkpoint_name = "checkpoint"
        if checkpoint_name_prefix:
            checkpoint_name += checkpoint_name_prefix
        checkpoint_name += '.pt'
        saved_content = {'torch_ae': self._torch_ae.state_dict()}
        if essentials is not None:
            saved_content.update({
                'optimizer': essentials['optimizer'].state_dict(),
                'train_loader_subset_indices': essentials['train_loader'].dataset.indices,
                'train_loader_generator_state': essentials['train_loader'].generator.get_state(),
                'val_loader_subset_indices': essentials['val_loader'].dataset.indices,
                'val_loader_generator_state': essentials['val_loader'].generator.get_state(),
            })
        torch.save(saved_content, os.path.join(checkpoint_dir, checkpoint_name))

    def load_checkpoint(
        self,
        checkpoint_dir: str,
        essentials: dict = None,
        summary: dict = None,
        checkpoint_name_prefix: str = None,
    ):
        checkpoint_name = "checkpoint"
        if checkpoint_name_prefix:
            checkpoint_name += checkpoint_name_prefix
        checkpoint_name += '.pt'
        states = torch.load(os.path.join(checkpoint_dir, checkpoint_name))
        self.load_state_dict_torch_ae(states['torch_ae'])
        if not essentials is None:
            essentials['optimizer'].load_state_dict(states['optimizer'])
            # set training subset indices
            if 'train_loader_subset_indices' in states.keys(): #backward-compatible
                num_old_indices_train = len(essentials['train_loader'].dataset.indices)
                essentials['train_loader'].dataset.indices = states['train_loader_subset_indices']
                if not len(essentials['train_loader'].dataset.indices) == num_old_indices_train:
                    warn('Number of indices in training subset changed when loading checkpoint.')
            # set training generator state
            if 'train_loader_generator_state' in states.keys(): #backward-compatible
                essentials['train_loader'].generator.set_state(states['train_loader_generator_state'])
            # set validation subset indices
            if 'val_loader_subset_indices' in states.keys(): #backward-compatible
                num_old_indices_val = len(essentials['val_loader'].dataset.indices)
                essentials['val_loader'].dataset.indices = states['val_loader_subset_indices']
                if not len(essentials['val_loader'].dataset.indices) == num_old_indices_val:
                    warn('Number of indices in validation subset changed when loading checkpoint.')
            # set validation generator state
            if 'val_loader_generator_state' in states.keys(): #backward-compatible
                essentials['val_loader'].generator.set_state(states['val_loader_generator_state'])

    def map_torch(self, xr):
        x = self._torch_decoder(xr)
        if self._shift_non_zero:
            x -= self._shift_torch
        return x

    def map(self, xr):
        x = self._load_cached_x(xr)
        if x is None:
            with torch.no_grad():
                x = self.to_numpy(self.map_torch(self.from_numpy(xr)))
            self._save_cached_x(xr, x)
        return x

    def inv_map_torch(self, x):
        # no shift in inv map, as encoder is learned without shift
        return self._torch_encoder(x)

    def inv_map(self, x):
        with torch.no_grad():
            xr = self.to_numpy(self.inv_map_torch(self.from_numpy(x)))
        return xr

    def project_x_torch(self, x):
        # no shift in inv map, as encoder is learned without shift
        x = self._torch_ae(x)
        if self._shift_non_zero:
            x -= self._shift_torch
        return x

    def project_x(self, x):
        with torch.no_grad():
            x = self.to_numpy(self.project_x_torch(self.from_numpy(x)))
        return x

    @execute_batched
    def project_tangent_with_encoder(self, x: np.ndarray, v: np.ndarray):
        assert x.shape[-1] == v.shape[-1]
        # no shift in inv map, as encoder is learned without shift
        # repeat x if single x is provided (since for jvp torch_x and torch_v have to be of the same size)
        if v.ndim == 2 and x.ndim == 1:
            x = np.outer(np.ones(v.shape[0]), x)
        else:
            assert v.ndim == x.ndim and x.ndim <= 2
        # project tangent
        with torch.enable_grad():
            assert not self._torch_ae.training
            _, jvp = torch.autograd.functional.jvp(
                self._torch_ae,
                self.from_numpy(x),
                self.from_numpy(v)
            )
        return self.to_numpy(jvp)

    def number_of_parameters_in_decoder(self):
        return sum(p.numel() for p in self._torch_decoder.parameters() if p.requires_grad)

    def number_of_parameters_in_total(self):
        return sum(p.numel() for p in self._torch_ae.parameters() if p.requires_grad)

    # def apply_transposed_jacobian(self, xr, v):
    #     torch_xr = self.from_numpy(xr).requires_grad_()
    #     decoded_xr = self._torch_decoder(torch_xr)
    #     decoded_xr.zero_grad()
    #     decoded_xr.backward(self.from_numpy(v))
    #     return self.to_numpy(decoded_xr), self.to_numpy(torch_xr.grad)

    def tangent_map(self, xr):
        jac = self._load_cached_jac(xr)
        if jac is None:
            torch_xr = self.from_numpy(xr)
            jac = self.to_numpy(get_jacobian(self._torch_decoder, torch_xr, self.manifold.ambient_dim))
            self._save_cached_jac(xr, jac)
        return jac

    def encoder_tangent_map(self, x):
        # no shift in inv map, as encoder is learned without shift
        torch_x = self.from_numpy(x)
        jac = self.to_numpy(get_jacobian(self._torch_encoder, torch_x, self.manifold.dim))
        return jac
    
    def encoder_jvp(self, x: np.ndarray, v: np.ndarray):
        assert x.shape[-1] == v.shape[-1]
        # no shift in inv map, as encoder is learned without shift
        # repeat x if single x is provided (since for jvp torch_x and torch_v have to be of the same size)
        if v.ndim == 2 and x.ndim == 1:
            x = np.outer(np.ones(v.shape[0]), x)
        else:
            assert v.ndim == x.ndim and x.ndim <= 2
        torch_x = self.from_numpy(x)
        torch_v = self.from_numpy(v)
        jvps = torch.autograd.functional.jvp(self._torch_encoder, torch_x, torch_v)[1]
        return self.to_numpy(jvps)
    
    def shift_ready_for_mor(self):
        if self.shift in [self.SHIFT_INITIAL_VALUE, self.SHIFT_INITIAL_VALUE_AFTER_TRAINING]:
            # check if shift is present
            # if not, you should call chart.update_shift(training_finished=True)
            recomputed_shift = self._torch_ae(torch.zeros((1, self.ambient_dim)))[0]
            return self._shift_non_zero is True \
                and self._shift_torch.requires_grad == False \
                and torch.allclose(self._shift_torch, recomputed_shift)
        return True

    def is_valid(self):
        return super().is_valid() and self._torch_ae.is_valid() and self.shift_ready_for_mor()

    def is_valid_projector(self, projector: VectorFieldProjector):
        return isinstance(projector, (MoorePenroseVectorFieldProjector, WeightedMoorePenroseVectorFieldProjector, EncoderVectorFieldProjector))

    def hessian(self, xr):
        warn('Computation of Hessian is very expensive.')
        torch_xr = self.from_numpy(xr)
        def jac(xr):
            return torch.autograd.functional.jacobian(self._torch_decoder, xr, create_graph=True)
        hess = torch.autograd.functional.jacobian(jac, torch_xr)
        return self.to_numpy(hess)

    def from_numpy(self, val: np.ndarray):
        return torch.from_numpy(val.astype(self._torch_dtype)).to(self.device)

    def to_numpy(self, val: torch.Tensor):
        return val.to('cpu').detach().numpy()


class EarlyStopper(IterativelyTrainableCallback):
    IDENTIFIER_BEST_SUMMARY = 'early_stopper_best_summary_'
    def __init__(self, net, n_epochs_early_stop, n_epochs, synced_chart=None):
        self.net = net
        self.best_val_loss = np.inf
        self.best_state_dict = None
        self.best_summary = None
        self.n_epochs_no_improv = 0
        self.n_epochs_early_stop = n_epochs_early_stop
        self.n_epochs = n_epochs
        self.stopping_criterion = None
        self.synced_chart = synced_chart
    
    def train_step(self, essentials: dict, summary: dict, logger: Logger = None):
        val_loss = summary[LOSS_VALIDATION]
        net = self.net

        if val_loss < self.best_val_loss:
            summary[IterativelyTrainableChart.SUMMARY_KEY_SHOULD_CHECKPOINT] = 1 #True cannot be logged with Mlflow (since it only accepts float)
            self.best_val_loss = val_loss
            self.best_summary = summary
            self.best_state_dict = copy.deepcopy(net.state_dict())
            if self.synced_chart is not None and self.synced_chart._epoch != summary['epoch']:
                self.synced_chart.load_state_dict_torch_ae(self.best_state_dict)
                self.synced_chart._epoch = summary['epoch']
            self.n_epochs_no_improv = 0

            if summary['epoch'] == self.n_epochs-1:
                print('Early Stopper: Final iteration is best iteration'.format(
                    self.best_val_loss
                ))
                self.stopping_criterion = 'last_is_best'
                summary.update({
                    'early_stopper_best_loss': self.best_val_loss,
                    'early_stopper_criterion_{}'.format(self.stopping_criterion): 1,
                    IterativelyTrainableChart.SUMMARY_KEY_SHOULD_STOP: 1, #1 instead of True since mlflow can only log float
                })
        else:
            self.n_epochs_no_improv += 1
            if self.n_epochs_no_improv >= self.n_epochs_early_stop:
                print('Early Stopper: Early stop with val_loss={}. No improvement in validation loss for {} epochs'.format(
                    self.best_val_loss,
                    self.n_epochs_no_improv
                ))
                net.load_state_dict(self.best_state_dict)
                self.stopping_criterion = 'early_stop'
                summary.update({
                    'early_stopper_best_loss': self.best_val_loss,
                    'early_stopper_criterion_{}'.format(self.stopping_criterion): 1,
                    IterativelyTrainableChart.SUMMARY_KEY_SHOULD_STOP: 1, #1 instead of True since mlflow can only log float
                })
            if summary['epoch'] == self.n_epochs-1:
                print('Early Stopper: Loaded best state so far at the end of training with best_val_loss={} instead of val_loss={}'.format(
                    self.best_val_loss, val_loss
                ))
                net.load_state_dict(self.best_state_dict)
                self.stopping_criterion = 'loaded_best_at_end'
                summary.update({
                    'early_stopper_best_loss': self.best_val_loss,
                    'early_stopper_criterion_{}'.format(self.stopping_criterion): 1,
                    IterativelyTrainableChart.SUMMARY_KEY_SHOULD_STOP: 1, #1 instead of True since mlflow can only log float
                })

    def summary_dict(self):
        summary = {
            'early_stopper_best_loss': self.best_val_loss,
            'early_stopper_criterion': self.stopping_criterion,
        }
        for key, val in self.best_summary.items():
            summary['{}{}'.format(self.IDENTIFIER_BEST_SUMMARY, key)] = val
        return summary


class EarlyStopperReportSummaryDict(IterativelyTrainableCallback):
    def __init__(self, early_stopper: EarlyStopper) -> None:
        super().__init__()
        self.early_stopper = early_stopper

    def train_finished(self, essentials: dict, summary: dict, logger: Logger = None):
        if self.early_stopper.stopping_criterion is not None:
            summary.update(self.early_stopper.summary_dict())
