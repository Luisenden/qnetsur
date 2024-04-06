"""
Plotting script for results from `extract_best_params_and_run_exhaustive.py`.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import glob
import pickle

import config
plt.style.use("seaborn-v0_8-paper")

font = 14
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': font,
    'axes.labelsize': font,  
    'xtick.labelsize': font,  
    'ytick.labelsize': font, 
    'legend.fontsize': font,
    'legend.title_fontsize': font,
    'axes.titlesize': font
})

import warnings
warnings.filterwarnings("ignore")

def read_pkl_surrogate(folder):
    surs = []
    for name in glob.glob(f'{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))

    vals = surs[0].vals
    Xs = []
    for i, sur in enumerate(surs):
        sur.X_df['Trial'] = i
        Xs.append(sur.X_df)

    Xs = pd.concat(Xs, axis=0, ignore_index=True)
    ys = pd.concat([pd.DataFrame(sur.y, columns = pd.Series(range(config.nnodes)).astype('str')) 
                    for sur in surs], axis=0, ignore_index=True)
    ys['Utility'] = ys.sum(axis=1)
    ys_std = pd.concat([pd.DataFrame(sur.y_std, columns = pd.Series(range(config.nnodes)).astype('str')).add_suffix('_std') 
                        for sur in surs], axis=0, ignore_index=True)

    ys_std['Utility Std'] = ys_std.apply(np.square).sum(axis=1).apply(np.sqrt)
    df = Xs.join(ys)
    df = df.join(ys_std)
    df['Method'] = 'Surrogate'
    return df.reset_index(), vals

def read_pkl_meta(folder):
    metas = []
    for name in glob.glob(f'{folder}/AX_*.pkl'): 
        with open(name,'rb') as file: metas.append(pickle.load(file))
    dfs = []
    for i, meta in enumerate(metas):
        data = meta[0]#.get_trials_data_frame()
        data['Trial'] = i
        dfs.append(data)
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['mean']
    df['Method'] = 'Meta'
    return df.reset_index()

def read_pkl_gridsearch(folder):
    gss = []
    for name in glob.glob(f'{folder}/GS_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for i, gs in enumerate(gss):
        gs[0]['Trial'] = i
        dfs.append(gs[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    df['Method'] = 'Random Gridsearch'
    return df.reset_index()

def read_pkl_sa(folder):
    gss = []
    for name in glob.glob(f'{folder}/SA_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for i, gs in enumerate(gss):
        gs[0]['Trial'] = i
        dfs.append(gs[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    df['Method'] = 'Simulated Annealing'
    return df.reset_index()


def to_dataframe(res):
    df = pd.DataFrame.from_records(res)
    df_raw = df[2].transform({i: itemgetter(i) for i in range(9)}) # get raw mean per node
    df_raw = df_raw.add_prefix('Node')
    df_raw['Aggregated Completed Requests'] = df_raw.sum(axis=1)
    return df_raw


def plot_from_exhaustive(df):
    df = df.melt(id_vars=['Method', 'Aggregated Completed Requests'], var_name='User', value_name='Number of Completed Requests')
    df['User'] = df['User'].apply(lambda x: str.replace(x, 'Node', ''))
    markers = ['o', '^', 'v', 's', 'd', 'P']
    fig, axs = plt.subplots(1,2, figsize=(10,7))
    sns.pointplot(data= df, x='User', y='Number of Completed Requests', hue='Method', ax=axs[0], errorbar='se', markers=markers, linestyles=['']*9, legend=False)
    sns.pointplot(data= df, x='Method', y='Aggregated Completed Requests', hue='Method', ax=axs[1], errorbar='se', markers=markers, legend=True, linestyles=['']*6)
    axs[0].set_title('Number of Completed Requests per User')
    axs[1].set_title('Aggregated Number of Completed Requests')
    plt.xticks(['']*9)
    plt.tight_layout()
    plt.grid()
    plt.show()

def plot_optimization_results(folder):
    target_columns = ['Trial', 'Utility', 'Method']
    df = pd.concat([read_pkl_surrogate(folder)[0][target_columns], read_pkl_meta(folder)[target_columns],
                    read_pkl_sa(folder)[target_columns], read_pkl_gridsearch(folder)[target_columns]], axis=0, ignore_index=True)

    grouped = df.groupby(['Method', 'Trial'], sort=False).max()
    sns.pointplot(data=grouped, x='Method', y='Utility', errorbar='se', linestyles='None', hue='Method')
    plt.xlabel('')
    plt.title('Maximum Utility (10 Cycles per Method)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()