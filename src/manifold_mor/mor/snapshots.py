'''A snapshot class that allows to store snapshots in a location
where they can be reused for different experiments.'''
import os
import pickle
import warnings
from typing import TYPE_CHECKING

import numpy as np
import ray
import scipy.sparse as sparse
import torch
from manifold_mor.context import get_current_context
from scipy.linalg import schur, svd

if TYPE_CHECKING:
    from manifold_mor.experiments.model import ModelExperiment


class SnapshotGenerator(object):
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger
    
    @classmethod
    def from_model_experiment(cls, model_experiment: 'ModelExperiment') -> 'SnapshotGenerator':
        return cls(model_experiment.get_model(), model_experiment.get_logger('fom'))
    
    @staticmethod
    def generate_from_model_experiment(model_experiment: 'ModelExperiment') -> 'Snapshots':
        generator = SnapshotGenerator.from_model_experiment(model_experiment)
        mus = model_experiment.get_mus()
        return generator.generate(
            mus,
            *model_experiment.get_temporal_data(),
            load=False,
            name='snapshots_{}'.format(model_experiment['mu_scenario']),
            # callbacks=callbacks, #TODO
            compute_rhs=model_experiment['compute_rhs'],
        )

    def generate(self, mus, t_0, t_end, dt, load=False, name='snapshots', callbacks=[], compute_rhs=False) -> 'Snapshots':
        assert isinstance(callbacks, (list, tuple))
        assert len(callbacks) == 0 or all(callable(c) for c in callbacks)

        save = False
        if load:
            snapshots_path = self._get_caching_path_base(name)
            snapshots_path_pkl = snapshots_path + '.pkl'
            try:
                pkl_file = open(snapshots_path_pkl, 'rb')
                loaded_snapshots = pickle.load(pkl_file)
                pkl_file.close()
                if not loaded_snapshots.is_computed_for(mus, t_0, t_end, dt)\
                    or (compute_rhs and loaded_snapshots.rhs_matrix is None):

                    warnings.warn('Snapshots are computed freshly due to non-matching parameters')
                    raise IOError()
                return loaded_snapshots
            except (FileNotFoundError, IOError):
                save = True
        
        if isinstance(mus, dict):
            mus = (mus,)
        assert isinstance(mus, (list, tuple))

        snapshots = self._compute_snapshots(mus, t_0, t_end, dt, name, callbacks)

        if compute_rhs:
            self._compute_rhs(snapshots)

        if save:
            folder = os.path.dirname(snapshots_path_pkl)
            if not os.path.exists(folder):
                os.makedirs(folder)
            pkl_file = open(snapshots_path_pkl, 'wb')
            pickle.dump(snapshots, pkl_file)
            pkl_file.close()

        return snapshots

    def _compute_snapshots(self, mus, t_0, t_end, dt, name, callbacks) -> 'Snapshots':
        print('Computing snapshots ...')
        n_t = np.ceil((t_end-t_0)/dt).astype(int)
        n_x = self.model.get_dim()

        if self.logger and not self.logger.is_running():
            self.logger.start_experiment()

        context = get_current_context()
        if context.is_parallel(context.TASK_GENERATE_SNAPSHOTS):
            num_cpus = context.get_resources(context.TASK_GENERATE_SNAPSHOTS)['cpu']
        else:
            num_cpus = None
        if num_cpus and num_cpus > 0:
            @ray.remote(num_cpus=num_cpus)
            def model_solve(mu):
                if self.logger:
                    self.logger.start_run(name)
                x, t = self.model.solve(t_0, t_end, dt, mu, logger=self.logger)
                for callback in callbacks:
                    callback(sol_x=x, sol_t=t, mu=mu, logger=self.logger, model=self.model)
                if self.logger:
                    self.logger.end_run()
                return mu, x, t
            output_ids = []
            print('Compute snapshots in parallel ...')
            for mu in mus:
                output_ids.append(model_solve.remote(mu))
            output_list = ray.get(output_ids)
            # collect results in correct order
            snapshot_matrix = np.empty((len(mus), n_t, n_x))
            for output in output_list:
                mu, x, t = output
                i_mu = mus.index(clear_under_in_mu(mu))
                snapshot_matrix[i_mu] = x
                sol_t = t
        else:
            snapshot_matrix = np.empty((len(mus), n_t, n_x))
            n_mu = len(mus)
            for i_mu, mu in enumerate(mus):
                print('Compute snapshots for parameter {}/{}'.format(i_mu+1, n_mu))
                if self.logger:
                    self.logger.start_run()
                snapshot_matrix[i_mu], sol_t = self.model.solve(t_0, t_end, dt, mu, logger=self.logger)
                for callback in callbacks:
                    callback(sol_x=snapshot_matrix[i_mu], sol_t=sol_t, mu=mu, logger=self.logger, model=self.model)
                if self.logger:
                    self.logger.end_run()

        return Snapshots(self.model.name, name, mus, t_0, t_end, dt, sol_t, snapshot_matrix)

    def _compute_rhs(self, snapshots):
        print('Computing RHS ...')
        mus = snapshots.mus
        rhs_matrix = np.empty_like(snapshots.matrix)

        context = get_current_context()
        if context.is_parallel(context.TASK_GENERATE_SNAPSHOTS):
            num_cpus = context.get_resources(context.TASK_GENERATE_SNAPSHOTS)['cpu']
        else:
            num_cpus = None
        if num_cpus and num_cpus > 0:
            @ray.remote(num_cpus=num_cpus)
            def compute_rhs_for(snapshot_block, sol_t, mu, model):
                model.set_mu(mu)
                rhs_block = np.empty_like(snapshot_block)
                for snapshot, (i_t, t) in zip(snapshot_block, enumerate(sol_t)):
                    #TODO: non-autonomous systems set t
                    rhs_block[i_t] = model.compute_rhs(snapshot)
                return mu, rhs_block
            output_ids = []
            print('Compute RHS in parallel ...')
            for mu in mus:
                output_ids.append(compute_rhs_for.remote(snapshots.get(mu), snapshots.sol_t, mu, self.model))
            output_list = ray.get(output_ids)
            # collect results in correct order
            for output in output_list:
                mu, rhs_block = output
                i_mu = mus.index(clear_under_in_mu(mu))
                rhs_matrix[i_mu] = rhs_block
        else:
            n_mu = len(mus)
            for i_mu, mu in enumerate(mus):
                print('Compute rhs for parameter {}/{}'.format(i_mu+1, n_mu))
                self.model.set_mu(mu)
                for snapshot, (i_t, t) in zip(snapshots.get(mu), enumerate(snapshots.sol_t)):
                    #TODO: non-autonomous systems set t
                    rhs_matrix[i_mu, i_t] = self.model.compute_rhs(snapshot)

        snapshots.rhs_matrix = rhs_matrix

    def _get_caching_path_base(self, snapshots_name):
        context = get_current_context()
        return os.path.join('snapshots', self.model.name, snapshots_name)

    def visualize(self, snapshots, path, subsample=None, visualize_rhs=False):
        print('Visualizing snapshots ...')
        n_mus = len(snapshots.mus)
        for idx, snapshot_matrix in enumerate(snapshots.matrix):
            print('Visualize snapshots for parameter {}/{}'.format(idx+1, n_mus))
            self.model.visualize(path + '_' + str(idx), snapshot_matrix, snapshots.sol_t, subsample=subsample)
        if visualize_rhs:
            if snapshots.rhs_matrix is None:
                warnings.warn('Could not visualize RHS snapshots, since snapshots.rhs_matrix is None!')
            else:
                for idx, snapshot_rhs_matrix in enumerate(snapshots.rhs_matrix):
                    print('Visualize RHS snapshots for parameter {}/{}'.format(idx+1, n_mus))
                    self.model.visualize(path + '_rhs_' + str(idx), snapshot_rhs_matrix, snapshots.sol_t, subsample=subsample)


class Snapshots(object):
    def __init__(self, model_name, name, mus, t_0, t_end, dt, sol_t, matrix, rhs_matrix=None):
        assert isinstance(matrix, np.ndarray) and len(matrix.shape)==3
        assert len(matrix) == len(mus)
        self.name = name
        self.model_name = model_name
        # remove temporary parameters starting with '_'
        self.mus = list(clear_under_in_mu(mu) for mu in mus)
        self.t_0 = t_0
        self.t_end = t_end
        self.dt = dt
        self.sol_t = sol_t
        self.matrix = matrix
        self._svd_data = None
        self._psd_complex_svd_data = None
        self._psd_svd_like_data = None
        self._psd_cotangent_lift_data = None
        self._subsampled_snapshots = dict() # save reference to subsampled snapshots to reuse them (and their decompositions like pod)
        self.rhs_matrix = rhs_matrix
        self.dataset = None
    
    def make_dataset(self, use_rhs_if_available: bool = True):
        self.dataset = SnapshotDataset(self, use_rhs_if_available=use_rhs_if_available)

    def is_computed_for(self, mus, t_0, t_end, dt):
        return len(self.mus) == len(mus) \
            and all(self_mu == clear_under_in_mu(mu) for self_mu, mu in zip(self.mus, mus)) \
            and self.t_0 == t_0 \
            and self.t_end == t_end \
            and self.dt == dt
    
    def get_temporal_data(self):
        return self.t_0, self.t_end, self.dt

    def get_all(self):
        shape = self.matrix.shape
        return self.matrix.reshape((shape[0]*shape[1], shape[2]))

    def get_all_rhs(self):
        if self.rhs_matrix is None:
            return None
        shape = self.rhs_matrix.shape
        return self.rhs_matrix.reshape((shape[0]*shape[1], shape[2]))
    
    def get_all_mu_idx(self):
        shape = self.matrix.shape
        if not self.rhs_matrix is None:
            assert self.rhs_matrix.shape == shape
        idx = np.arange(shape[0], dtype=int)[:, np.newaxis] @ np.ones((1,shape[1]), dtype=int)
        return idx.reshape((shape[0]*shape[1],))

    def get_dim(self):
        return self.matrix.shape[-1]
    
    def get_mu_index(self, mu: dict) -> int:
        return self.mus.index(clear_under_in_mu(mu))

    def get(self, mu):
        return self.matrix[self.get_mu_index(mu)]
    
    def get_rhs(self, mu):
        return self.rhs_matrix[self.get_mu_index(mu)]

    def get_subsampled_snapshots(self, indices, name_postfix) -> 'SubsampledSnapshots':
        str_indices = str(indices)
        if str_indices not in self._subsampled_snapshots.keys():
            self._subsampled_snapshots[str_indices] = SubsampledSnapshots(self, indices, name_postfix)
        return self._subsampled_snapshots[str_indices]

    def get_svd_data(self):
        decomp_name = 'svd'
        if not self._svd_data:
            try:
                self._svd_data = self._load_decomposition(decomp_name)
            except FileNotFoundError:
                # save svd
                U, s, _ = svd(self.get_all().T, full_matrices=False)
                self._svd_data = (U, s)
                self._save_decomposition(decomp_name, self._svd_data)
        return self._svd_data

    def get_psd_complex_svd_data(self):
        decomp_name = 'psd_complex_svd'
        if not self._psd_complex_svd_data:
            try:
                self._psd_complex_svd_data = self._load_decomposition(decomp_name)
            except FileNotFoundError:
                # save complex svd
                X_real = self.get_all()
                N = X_real.shape[-1] // 2
                X_complex = X_real[:, :N] + 1j * X_real[:, N:]
                U, s, _ = svd(X_complex.T, full_matrices=False)
                self._psd_complex_svd_data = (U, s)
                self._save_decomposition(decomp_name, self._psd_complex_svd_data)

        return self._psd_complex_svd_data

    def get_psd_svd_like_data(self, rel_tol=1e-8):
        decomp_name = 'psd_svd_like'
        if not self._psd_svd_like_data:
            try:
                self._psd_svd_like_data = self._load_decomposition(decomp_name)
            except FileNotFoundError:
                X = self.get_all()
                N = X.shape[-1] // 2
                ns = X.shape[0]
                J2 = lambda n: sparse.vstack([
                    sparse.hstack([sparse.bsr_matrix((n, n)), sparse.identity(n)]),
                    sparse.hstack([-sparse.identity(n), sparse.bsr_matrix((n, n))])
                ])
                J2N = J2(N)
                if 2*N > ns:
                    symplectic_gramian = X[:, :N] @ X[:, N:].T
                    symplectic_gramian -= X[:, N:] @ X[:, :N].T
                    DJD, Q, _ = schur(symplectic_gramian, sort=lambda x: x.imag > 0)
                    abs_diag = np.empty(DJD.shape[0])
                    abs_diag[::2] = np.abs(np.diag(DJD, -1)[::2])
                    abs_diag[1::2] = np.abs(np.diag(DJD, 1)[::2])
                    cutoff = np.where(abs_diag / abs_diag.max() < rel_tol)[0][0] // 2
                    assert abs_diag[:2*cutoff].min() > abs_diag[2*cutoff:].max(), 'sorting order in abs_diag is incorrect'
                    i_sort = range(2*cutoff)
                    i_sort = np.hstack([i_sort[::2], i_sort[1::2]])
                    Q = Q[:, i_sort]
                    SD = X.T @ Q
                    w_sympl_sing_val = np.sum(np.sum(SD**2, 0).reshape((2, -1)), 0)
                    diag_D = np.sqrt(abs_diag[i_sort])
                    S = SD / diag_D
                    # correct sign
                    d = np.diag(J2(cutoff).T @ S.T @ J2N @ S)
                    S = S * np.sign(np.hstack([d[:cutoff], np.ones((cutoff))]))
                else:
                    symplectic_gramian = J2N @ X.T @ X
                    complex_DDJ, S = np.linalg.eig(symplectic_gramian)
                    abs_diag = np.abs(complex_DDJ.imag)
                    idx_sort = np.argsort(abs_diag)[::-1]
                    abs_diag = abs_diag[idx_sort]
                    cutoff = np.where(abs_diag / abs_diag.max() < rel_tol)[0][0] // 2
                    W = np.kron(np.eye(cutoff), 1/np.sqrt(2) * np.array([[1, -1j], [1, 1j]]))
                    S = S[:, idx_sort[:2*cutoff]] @ W
                    DDJ = W.conj().T @ np.diag(complex_DDJ[:2*cutoff]) @ W
                    diag_D = np.sqrt(abs_diag[:2*cutoff])
                    if np.max(np.abs(S.imag)) > 1e-6 or np.max(np.abs(DDJ.imag)) > 1e-6:
                        raise ValueError('Something went wrong')
                    # throw away imag part which is numerically zero
                    S = S.real
                    # resort to canonical symplectic
                    idx = np.hstack([np.arange(0, 2*cutoff, 2), np.arange(1, 2*cutoff, 2)])
                    S = S[:, idx]
                    diag_D = diag_D[idx]
                    d = np.diag(J2(cutoff).T @ S.T @ J2N @ S)
                    # normalize S' * J2n * S and correct sign
                    S = S / np.sqrt(np.abs(d)) * np.sign(np.hstack([d[:cutoff], np.ones((cutoff))]))
                    # weighted sympl sing val
                    w_sympl_sing_val = np.sum(np.sum((S * diag_D)**2, 0).reshape((2, -1)), 0)

                symplecticity = np.linalg.norm(S.T @ J2N @ S - J2(cutoff))
                self._psd_svd_like_data = (S, w_sympl_sing_val, symplecticity)
                self._save_decomposition(decomp_name, self._psd_svd_like_data)

        return self._psd_svd_like_data

    def get_psd_cotangent_lift_data(self):
        decomp_name = 'psd_cotangent_lift'
        if not self._psd_cotangent_lift_data:
            try:
                self._psd_cotangent_lift_data = self._load_decomposition(decomp_name)
            except FileNotFoundError:
                X = self.get_all().T
                N = X.shape[0] // 2
                X_combined = np.hstack([X[:N, :], X[N:, :]])
                U, s, _ = svd(X_combined, full_matrices=False)
                self._psd_cotangent_lift_data = (U, s)
                self._save_decomposition(decomp_name, self._psd_cotangent_lift_data)

        return self._psd_cotangent_lift_data

    def __sizeof__(self):
        return len(self)

    def __len__(self):
        return np.prod(self.matrix.shape[:2])
    
    #TODO: decompositions in fom corresponding fom experiment
    def _save_decomposition(self, decomp_name, obj):
        partial_file_name = self._get_path()
        folder = os.path.dirname(partial_file_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        with open('{}_{}.pkl'.format(partial_file_name, decomp_name), 'wb') as f:
            pickle.dump(obj, f)
    
    #TODO: decompositions in fom corresponding fom experiment
    def _load_decomposition(self, decomp_name):
        with open('{}_{}.pkl'.format(self._get_path(), decomp_name), 'rb') as f:
            decomp = pickle.load(f)
        return decomp

    def _get_path(self):
        context = get_current_context()
        return os.path.join(context.root_folder, 'snapshots', self.model_name, self.name)


class SubsampledSnapshots(Snapshots):
    def __init__(self, snapshots, indices, name_postfix):
        state = dict()
        for k, v in snapshots.__dict__.items():
            if k == 'matrix':
                state[k] = v[:, :, indices]
            elif k == 'rhs_matrix':
                if v is None:
                    state[k] = None
                else:
                    state[k] = v[:, :, indices]
            elif k in [
                '_svd_data',
                '_psd_cotangent_lift_data',
                '_psd_complex_svd_data',
                '_psd_svd_like_data',
                '_subsampled_snapshots',
                'dataset',
            ]:
                pass
            elif k == 'name':
                state[k] = v + '_' + name_postfix
            else:
                state[k] = v
        super().__init__(**state)


class SnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, snapshots, device='cpu', use_rhs_if_available: bool = True):
        super(SnapshotDataset, self).__init__()
        self.snapshot_tensor = torch.tensor(snapshots.get_all(), device=device)
        self.mu_idx = snapshots.get_all_mu_idx()
        if not snapshots.rhs_matrix is None and use_rhs_if_available:
            self.rhs_snapshot_tensor = torch.tensor(snapshots.get_all_rhs(), device=device)
        else:
            self.rhs_snapshot_tensor = None
    
    def __len__(self):
        return self.snapshot_tensor.shape[0]
    
    def __getitem__(self, idx):
        samples = {
            'data': self.snapshot_tensor[idx],
            'mu_idx': self.mu_idx[idx],
        }
        if not self.rhs_snapshot_tensor is None:
            samples['tangents'] = self.rhs_snapshot_tensor[idx]
        return samples


def clear_under_in_mu(mu):
    return {k: v for k, v in mu.items() if not k[0] == '_'}
