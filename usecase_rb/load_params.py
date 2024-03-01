import pickle
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
import config

import pandas as pd
import numpy as np

def get_surrogate_data(sur_files):
    dfs_sur = []
    for i,sur in enumerate(sur_files):
        df_sur_y = pd.DataFrame(sur.y)
        df_sur_y['sum mean'] = df_sur_y.sum(axis=1)
        df_sur_y['Objective Mean'] = (df_sur_y['sum mean'] + sur.X_df.drop(['Iteration'], axis=1).sum(axis=1)/config.m_max).astype('float')
        grouped_iteration = sur.X_df.join(df_sur_y).groupby(['Iteration'], as_index=False).apply(lambda x: x.loc[x['Objective Mean'].idxmax()])
        grouped_iteration['Trial'] = i
        dfs_sur.append(grouped_iteration)

    df_sur = pd.concat(dfs_sur).reset_index()
    df_sur['Method'] = 'Surrogate Optimization'
    return df_sur


def load_max_params(folder):
    axs = []
    for name in glob.glob(f'../../surdata/{folder}/AX_*.pkl'):
        with open(name,'rb') as file: axs.append(pickle.load(file))

    surs = []
    for name in glob.glob(f'../../surdata/{folder}/SU_*.pkl'): #Sur_qswitch_nleafnodes{nnodes}_{time}*'):
        with open(name,'rb') as file: surs.append(pickle.load(file))

    sas = []
    for name in glob.glob(f'../../surdata/{folder}/SA_*.pkl'):
        with open(name,'rb') as file: sas.append(pickle.load(file))

    gss = []
    for name in glob.glob(f'../../surdata/{folder}/GS_*.pkl'):
        with open(name,'rb') as file: gss.append(pickle.load(file))


    def get_other_data(df, collect_name, obj_name, i):
        df['Trial'] = i
        paramcolumns = df.columns[df.columns.str.contains('mem_size')]
        df['Objective Mean'] = df[collect_name] + df[paramcolumns].sum(axis=1)/config.m_max
        df['Method'] = obj_name
        return df


    df_sur = get_surrogate_data(surs)
    
    dfs_ax = [get_other_data(ax[0].get_trials_data_frame(), 'mean', 'Meta Optimization', i) for i,ax in enumerate(axs)]
    df_ax = pd.concat(dfs_ax).reset_index()

    dfs_sa = [get_other_data(sa, 'objective', 'Simulated Annealing', i) for i,sa in enumerate(sas)]
    df_sa = pd.concat(dfs_sa).reset_index()

    dfs_gs = [get_other_data(gs[0], 'objective', 'Random Grid Search', i) for i,gs in enumerate(gss)]
    df_gs = pd.concat(dfs_gs).reset_index()
    df_gs['Objective Mean'] = df_gs['Objective Mean'].apply(lambda x: np.sum(x))

    dfs = [df_sur, df_ax, df_gs, df_sa]
    paramcolumns = df_gs.columns[df_gs.columns.str.contains('mem_size')]

    dfs_obj = []
    for df in dfs:
        dfs_obj.append(df[paramcolumns.to_list()+['index', 'Objective Mean', 'Method', 'Trial']])
    dfs = pd.concat(dfs_obj)
    dfs['\# Optimization steps'] = dfs['index']
    grouped = dfs.groupby(['Method', 'Trial']).apply(
        lambda x: x.loc[x['Objective Mean'].idxmax()]
        )
    
    return grouped[paramcolumns]

