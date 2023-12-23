# generate plots from paper

import os

import mlflow
import numpy as np
import pandas as pd
from manifold_mor_demos.hamiltonian_wave.main_online_phase_paper import BumpOnlineExperiment
from manifold_mor_demos.hamiltonian_wave.model_experiments import BumpModelExperiment
from manifold_mor.context import ManifoldMorContext
from manifold_mor.experiments.basic import ManifoldMorExperiment
from mlflow.tracking.client import MlflowClient

PATH_EXPORT = 'export'
TYPE_EMBEDDING = 'type_embedding'

def get_data_frame(mmexp: ManifoldMorExperiment, filter_string: str = None):
    old_cwd = os.getcwd()
    os.chdir(mmexp.get_experiment_folder())
    exps = mlflow.list_experiments()
    exp_id = None
    for exp in exps:
        if exp.name == 'mor':
            exp_id = exp.experiment_id
            break
    assert exp_id is not None, 'mor experiment not found'

    df = mlflow.search_runs(exp_id, filter_string=filter_string)

    # field for type of embedding
    mask_linear = df['params.manifold'].str.contains('basis_generation')
    df[TYPE_EMBEDDING] = 'dummy'
    # linear
    df.loc[mask_linear, TYPE_EMBEDDING] = 'bump_lin_' + df.loc[mask_linear, 'params.manifold'].str.extract(r'basis_generation:(.*),')[0]
    # nonlinear
    df.loc[~mask_linear, TYPE_EMBEDDING] = df.loc[~mask_linear, 'params.manifold'].str.split('(', expand=True, n=1)[0]
    # nonlinear, symplectic
    df.loc[df['params.manifold'].str.contains('loss_weight_data:0.9'), TYPE_EMBEDDING] += '_s'

    # adjust red_dim
    df['params.red_dim'] = df['params.red_dim'].astype(int) - 4 # boundary nodes

    os.chdir(old_cwd)

    return df


def get_metric_history(client: MlflowClient, run_ids, metric_name, target_value=None):
    if metric_name.startswith('metrics.'):
        metric_name = metric_name.split('metrics.')[-1] #remove metrics.
    rows = []
    for run_id in run_ids:
        try:
            history = client.get_metric_history(run_id, metric_name)
        except mlflow.exceptions.MlflowException:
            return None
        if not target_value is None:
            target_value_deviation = sum((m.value - target_value)**2 for m in history)
        else:
            target_value_deviation = None
        rows += [dict(key=m.key, value=m.value, timestamp=m.timestamp, step=m.step, run_id=run_id, target_value_deviation=target_value_deviation) for m in history]
    return pd.DataFrame(rows)

def gen_csv_for_pgfplots():
    path_export = os.path.abspath(os.path.join(PATH_EXPORT, 'wave_from_online'))
    if not os.path.exists(path_export):
        os.makedirs(path_export)
    exp = BumpOnlineExperiment()
    df = get_data_frame(exp)
    df['type_reduction'] = df[TYPE_EMBEDDING] + '_' + df['params.reduction']
    
    df_relevant = df[[
        'run_id',
        'params.red_dim',
        'params.mu_idx',
        'type_reduction',
        'metrics.projection_total_err_x_proj_2norm',
        'metrics.projection_total_err_x_rom_2norm',
    ]]

    for mu_idx in set(df_relevant['params.mu_idx']):
        # projection error
        pd.pivot(
            df_relevant.loc[df_relevant['params.mu_idx'] == mu_idx],
            index='params.red_dim',
            columns='type_reduction',
            values='metrics.projection_total_err_x_proj_2norm'
        ).to_csv(os.path.join(path_export, 'err_mu{}_proj.dat'.format(mu_idx)))

        # reduction error
        pd.pivot(
            df_relevant.loc[df_relevant['params.mu_idx'] == mu_idx],
            index='params.red_dim',
            columns='type_reduction',
            values='metrics.projection_total_err_x_rom_2norm'
        ).to_csv(os.path.join(path_export, 'err_mu{}_rom.dat'.format(mu_idx)))

    # Hamiltonian
    os.chdir(exp.get_experiment_folder())
    client = MlflowClient()
    df_ham = get_metric_history(
        client,
        df.loc[
            (df['status']=='FINISHED')
            &
            (df['params.manifold'].str.contains('bump_ae'))
        ]['run_id'],
        'hamiltonian',
    ).merge(df, on='run_id')
    # step to time
    nt = len(set(df_ham['step']))
    t_0 = float(df['params.t_0'][0])
    t_end = float(df['params.t_end'][0])
    df_ham['time'] = t_0 + df_ham['step'] / (nt-1) * t_end

    # compute ham err
    model_exp = BumpModelExperiment(BumpModelExperiment.MU_TEST_GENERALIZATION)
    model = model_exp.get_model()
    mus = model_exp.get_mus()

    df_ham['ham_err'] = -1

    for mu in mus:
        model.set_mu(mu)
        z0 = model.initial_value(mu)
        ham_value = model.vector_field.Ham(z0)

        mask = df_ham['params.c'] == str(mu['c'])
        df_ham.loc[mask, 'ham_err'] = abs(df_ham.loc[mask, 'value'] - ham_value)

    # select only first paramter
    df_ham = df_ham.loc[df_ham['params.mu_idx'] == '0']

    ham_data_min_max = df_ham[['time']][:nt]
    for type_reduction in set(df_ham['type_reduction']):
        df_ham_relevant = df_ham[df_ham['type_reduction'] == type_reduction]

        min_err = np.empty(nt)
        max_err = np.empty(nt)
        for i_t in range(nt): #TODO more efficient?
            relevant_ham_err = df_ham_relevant.loc[df_ham_relevant['step'] == i_t, 'ham_err']
            min_err[i_t] = min(relevant_ham_err)
            max_err[i_t] = max(relevant_ham_err)
        
        ham_data_min_max[type_reduction + '_max'] = max_err
        ham_data_min_max[type_reduction + '_min'] = min_err
    ham_data_min_max.to_csv(os.path.join(path_export, 'err_ham.dat'))

    # symplecticity
    os.chdir(exp.get_experiment_folder())
    client = MlflowClient()
    df_sympl = get_metric_history(
        client,
        df.loc[(df['params.mu_idx'] == '0') & (df['params.manifold'].str.contains('bump_ae_0\(loss_weight_data:0.9'))]['run_id'],
        'err_in_sympl',
    ).merge(df, on='run_id')
    # square sympl
    df_sympl['squared_value'] = df_sympl['value']**2
    nt = len(set(df_sympl['step']))
    t_0 = float(df['params.t_0'][0])
    t_end = float(df['params.t_end'][0])
    df_sympl['time'] = t_0 + df_sympl['step'] / (nt-1) * t_end
    pd.pivot(
        df_sympl,
        index='time',
        columns='params.red_dim',
        values='squared_value'
    ).to_csv(os.path.join(path_export, 'err_sympl.dat'))

    # bar plot: converged
    df['failed'] = df['status'] == 'FAILED'
    df['finished'] = df['status'] == 'FINISHED'
    finished = df.loc[~df['type_reduction'].str.contains('lin')].pivot_table(
        values=['failed', 'finished'], index='type_reduction', aggfunc='sum')
    map_to_latex_label = {
        'bump_ae_0_s_sympl_canonical': r'{$\cAEs{2n}$\\ SMG}',
        'bump_ae_0_lspg': r'{$\cAE{0}{2n}$\\ M-LSPG}',
        'bump_ae_0_pullback': r'{$\cAE{0}{2n}$\\ MG}',
        'bump_ae_1_lspg': r'{$\cAE{1}{2n}$\\ M-LSPG}',
        'bump_ae_1_pullback': r'{$\cAE{1}{2n}$\\ MG}',
    }
    finished.rename(map_to_latex_label).to_csv(os.path.join(path_export, 'finished.dat'))
    os.chdir(context.root_folder)

def reformat_df_err(df_err: pd.DataFrame):
    df_new = pd.wide_to_long(
        df_err.reset_index(),
        'bump',
        'params.red_dim',
        'type_reduction',
        sep='_',
        suffix=r'\w+'
    ).rename(columns={"bump": "err"}).reset_index()

    df_new['type_reduction'] = df_new['type_reduction'].str.replace('sympl_canonical', '0-S(M)G')
    df_new['type_reduction'] = df_new['type_reduction'].str.replace('pullback', '1-(M)G')
    df_new['type_reduction'] = df_new['type_reduction'].str.replace('lspg', '2-(M)LSPG')
    df_new['type_reduction'] = df_new['type_reduction'].str.replace('linear_subspace', '1-(M)G')
    df_new['reduction'] = df_new['type_reduction'].str.extract('_([^_]+$)')
    df_new['type'] = df_new['type_reduction'].str.extract('(.*)_[^_]+$')
    return df_new

def plots():
    from plotly.express import line, scatter, bar
    import plotly.graph_objects as go

    path_export = os.path.abspath(os.path.join(PATH_EXPORT, 'wave_from_online'))

    # reduction error
    err_types = ['proj', 'rom']
    filename = 'err_mu{}_{}.dat'
    dfs_err = {}
    for err_type in err_types:
        dfs = []
        for ii in range(3):
            dfs.append(pd.read_csv(
                os.path.join(path_export, filename.format(ii, err_type)))
            )
            dfs[-1]['mu_idx'] = ii
        
        df_err = pd.concat(dfs).groupby('params.red_dim').mean()
        dfs_err[err_type] = reformat_df_err(df_err.drop(['mu_idx'], axis=1))
        dfs_err[err_type]['err_type'] = err_type

    fig = scatter(pd.concat(dfs_err).sort_values(by='reduction'),
        'params.red_dim',
        'err',
        color='type',
        facet_col='reduction',
        symbol='err_type',
        symbol_sequence = ['cross', 'circle'],
        log_y=True,
        title='Projection and reduction error.',
    )
    fig.update_traces(marker={'size': 10})
    fig.update_yaxes(exponentformat="e")
    fig.show()

    # (un)finished runs
    df = pd.read_csv(os.path.join(path_export, 'finished.dat')).rename(
        columns={
            'finished': 'count-1_yes',
            'failed': 'count-2_no',
        }
    )
    df['type_reduction'] = df['type_reduction'].str.replace(
        '{$\\cAE{0}{2n}$\\\\ M-LSPG}', '2-ae_0, M-LSPG', regex=False
    ).str.replace(
        '{$\\cAE{0}{2n}$\\\\ MG}', '1-ae_0, MG', regex=False
    ).str.replace(
        '{$\\cAEs{2n}$\\\\ SMG}', '0-ae_0_s, SMG', regex=False
    ).str.replace(
        '{$\\cAE{1}{2n}$\\\\ M-LSPG}', '4-ae_1, M-LSPG', regex=False
    ).str.replace(
        '{$\\cAE{1}{2n}$\\\\ MG}', '3-ae_1, MG', regex=False
    )
    df_long = pd.wide_to_long(
        df,
        'count',
        'type_reduction',
        'is_finished',
        sep='-',
        suffix=r'\w+'
    ).reset_index()
    bar(
        df_long.sort_values(by=['type_reduction', 'is_finished']),
        x='type_reduction',
        y='count',
        color='is_finished',
        title='Number of successful compared to failed reduced simulation runs.'
    ).show()

    # error in symplecticity
    df = pd.read_csv(os.path.join(path_export, 'err_sympl.dat')).set_index('time').add_prefix('err_sympl-').reset_index()
    df_long = pd.wide_to_long(
        df,
        'err_sympl',
        'time',
        'red_dim',
        sep='-',
        suffix=r'\w+'
    ).reset_index()
    fig = line(df_long, 'time', 'err_sympl', color='red_dim', log_y=True, title='Error in symplecticity.')
    fig.update_yaxes(exponentformat="e")
    fig.show()

    # error in Hamiltonian
    df = pd.read_csv(os.path.join(path_export, 'err_ham.dat'))
    df.columns = df.columns.str.replace(
        '_pullback', ', MG'
    ).str.replace(
        '_lspg', ', M-LSPG'
    ).str.replace(
        '_sympl_canonical', ', SMG'
    )

    all_type_reduction = sorted(list(set(c.strip('_min').strip('_max') for c in df.columns if 'bump' in c)))

    fig = go.Figure()
    for type_reduction in all_type_reduction:
        y_upper = df['{}_max'.format(type_reduction)]
        y_lower = df['{}_min'.format(type_reduction)]
        fig.add_trace(go.Scatter(
            x=np.concatenate([df['time'], df['time'][::-1]]),
            y=pd.concat([y_upper, y_lower[::-1]]),
            fill='toself',
            hoveron='points',
            name=type_reduction,
        ))

    fig.update_layout(yaxis_type = "log", title='Error in the Hamiltonian.', xaxis_title='time')
    fig.update_yaxes(exponentformat="e")
    fig.show()

if __name__ == '__main__':
    context = ManifoldMorContext()
    gen_csv_for_pgfplots()
    plots()
