'''Plot functions to plot the error from a MOR experiment with a ProjectionModule.'''
import os
from typing import Dict, List, Union

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from manifold_mor.context import get_current_context
from mlflow.entities.view_type import ViewType
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE

EXP_NOTE = 'exp_note'

def get_all_entries(
    path_mlflow: str,
    experiment_ids_dict: Union[List[int], int, Dict[str, int]],
):
    # cast to list
    if isinstance(experiment_ids_dict, int):
        experiment_ids_dict = [experiment_ids_dict]

    # cast to dict
    if isinstance(experiment_ids_dict, (int, list)):
        tracking_uri = path_mlflow
        experiment_ids_dict = {tracking_uri: experiment_ids_dict}

    pd_frames = []
    for tracking_uri, experiment_ids in experiment_ids_dict.items():
        mlflow.set_tracking_uri(tracking_uri)

        for exp_id in experiment_ids:
            pd_frames.append(mlflow.search_runs(
                str(exp_id),
                run_view_type = ViewType.ALL,
                filter_string='attributes.status = "FINISHED"'
            ))
            # read experiment for experiment note in tags
            exp = mlflow.get_experiment(str(exp_id))
            exp_note = exp.tags.get(MLFLOW_RUN_NOTE, None)
            if not exp_note is None:
                pd_frames[-1][EXP_NOTE] = exp_note

    return pd.concat(pd_frames)

def mor_error_plot_from_mlflow_results(
    df: pd.DataFrame,
    title: str = None,
    plot_vf_err: bool = True,
    log_y_axis: bool = True,
    key_color: str = 'params.manifold_type',
    key_symbol: str = 'params.reduction',
    error_correction_factor: float = 1.,
    return_fig: bool = False,
    additional_fig_options: dict = None,
    additional_labels: dict = None,
):
    """Plot projection and mor errors for state x and the vector field (vf).

    Args:
        df (pd.DataFrame): DataFrame to plot from.
        title (str): Title of the figure.
        plot_vf_err (bool, optional): whether to plot vector field error. Defaults to True.
        log_y_axis (bool, optional): whether to logarithmize the y-axis. Defaults to True.
        key_color (str): key which is used to define the attribute that is colored. Defaults to 'params.manifold_type'.
        error_correction_factor (float): Can be useful if the error was computed with respect to shifted snapshots. Defaults to 1.
    """
    additional_fig_options = additional_fig_options or {}
    labels = additional_labels.copy() if additional_labels is not None else dict()
    # restucture df to fit the input of px.scatter
    dfs = pd.DataFrame(columns=list(df.columns) + ['err_type', 'err_for'])
    err_fors = ['x']
    if plot_vf_err:
        err_fors.append('vf')
    for err_type in ('proj', 'rom'):
        for err_for in err_fors:
            mod_df = df.copy()
            key_err = 'metrics.projection_total_err_{}_{}_2norm'.format(err_for, err_type)
            key_sqrt_err = 'metrics.projection_total_sqrt_err_{}_{}_2norm'.format(err_for, err_type)
            mod_df['err_type'] = err_type
            mod_df['err_for'] = err_for
            # due to the structure of ProjectionMorPipelineModule, there is an additional sqrt
            # in the keyword, if the metric does give squared errors
            if key_sqrt_err in mod_df.keys():
                key_err = key_sqrt_err
            if not np.isclose(error_correction_factor, 1.):
                mod_df[key_err] *= error_correction_factor
            mod_df['err'] = mod_df[key_err]
            labels[key_err] = '{}, {}'.format(err_for, err_type)
            dfs = dfs.append(mod_df)
    
    # convert red_dim from str to int
    dfs['params.red_dim'] = dfs['params.red_dim'].astype(int)

    # drop reduced errors for status not finished as they not computed for all snapshots
    dfs = dfs.reset_index()
    dfs = dfs.drop(dfs[(dfs['status'] != 'FINISHED') & (dfs['err_type'] == 'rom')].index)

    # gen hover_data
    hover_data = ['params.manifold'] + list(labels.keys())
    if EXP_NOTE in dfs.columns:
        hover_data.append(EXP_NOTE)

    fig = px.scatter(
        dfs,
        x='params.red_dim',
        y='err',
        color=key_color,
        symbol=key_symbol,
        facet_col='err_type',
        facet_row='err_for' if plot_vf_err else None,
        labels=labels,
        log_y=log_y_axis,
        hover_data=hover_data,
        **additional_fig_options,
    )

    if title is not None:
        if not title.startswith('mor_err_plot_'):
            title = 'mor_err_plot_{}'.format(title)
        fig.update_layout(title_text=title)

    if return_fig:
        return fig
    else:
        fig.write_html("{}.html".format(title.replace(' ', '_')))
