"""
Plotting tools for continuous-protocols use case.
"""

import pandas as pd
import re
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
    'text.usetex': False,
    'font.family': 'arial',
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

def read_pkl_surrogate_timeprofiling(folder):
    surs = []
    for name in glob.glob(f'{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))
    
    times = {'Simulation':[], 'Build':[], 'Acquisition':[], 'Simulation per Iteration Mean':[] }
    for sur in surs:
        times['Simulation'].append(np.sum(sur.sim_time))
        times['Simulation per Iteration Mean'].append(np.mean(sur.sim_time))
        times['Build'].append(np.sum(sur.build_time))
        times['Acquisition'].append(np.sum(sur.acquisition_time))

    times = pd.DataFrame.from_dict(times)
    times['Total'] = times.drop('Simulation per Iteration Mean', axis=1).sum(axis=1)
    times_relative = times.div(times['Total'], axis=0)
    return times, times_relative

def read_pkl_surrogate_benchmarking(folder):
    surs = []
    for name in glob.glob(f'{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))
    
    errors = []
    acquisition = {}
    for i,sur in enumerate(surs):
        errors.append(pd.DataFrame.from_records(sur.model_scores))
        errors[i]['Trial'] = i
        errors[i] = errors[i].reset_index(names='Iteration')
        acquisition[f'Trial {i}'] = pd.Series(sur.acquisition_time)/np.array(sur.optimize_time)[1:] # acquisition starts from iteration 1

    df_errors = pd.concat(errors).melt(id_vars=['Trial', 'Iteration'], value_name='Mean Absolute Error', var_name='Surrogate')
    fig, ax = plt.subplots()
    sns.lineplot(data=df_errors, x='Iteration', y='Mean Absolute Error', hue='Surrogate', marker='o', errorbar='sd', err_style='bars')
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df_acquisition = pd.DataFrame.from_dict(acquisition).reset_index(names='Iteration')
    df_acquisition['Iteration'] += 1. # acquisition starts from iteration 1
    df_acquisition = df_acquisition.melt(id_vars='Iteration', value_name='Acquisition Time [s] / Exec Time [s]', var_name='Trial')
    fig, ax = plt.subplots()
    sns.lineplot(data=df_acquisition, x='Iteration', y='Acquisition Time [s] / Exec Time [s]', marker='o', errorbar='sd', err_style='bars')
    plt.grid()
    plt.tight_layout()
    plt.show()
    return df_errors, df_acquisition


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
    ys = pd.concat([pd.DataFrame(sur.y, columns = pd.Series(range(len(vals['user'][0]))).astype('str')) 
                    for sur in surs], axis=0, ignore_index=True)
    ys['Utility'] = ys.sum(axis=1)
    ys_std = pd.concat([pd.DataFrame(sur.y_std, columns = pd.Series(range(len(vals['user'][0]))).astype('str')).add_suffix('_std') 
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
        data = meta[0]
        data['Trial'] = i
        dfs.append(data)
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['evaluate']
    df['Method'] = 'Meta'
    return df.reset_index()

def read_pkl_randomsearch(folder):
    gss = []
    for name in glob.glob(f'{folder}/RS_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for i, gs in enumerate(gss):
        gs[0]['Trial'] = i
        dfs.append(gs[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    df['Method'] = 'Uniform Random Search'
    return df.reset_index()

def read_pkl_sa(folder):
    gss = []
    for name in glob.glob(f'{folder}/SA_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for i, gs in enumerate(gss):
        gs['Trial'] = i
        dfs.append(gs)
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    df['Method'] = 'Simulated Annealing'
    return df.reset_index()


def to_dataframe(res, users):
    df = pd.DataFrame.from_records(res)
    df = df[0].apply(pd.Series)
    df.columns = users
    df = df.add_prefix('User')
    df['Aggregated Number of Virtual Neighbors'] = df.sum(axis=1)
    return df


def get_performance_distribution_per_method(folder):
    df_sur, _ = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_randomsearch(folder)
    columns = ['Trial', 'Method', 'Utility']
    df = pd.concat([df_sur[columns], df_meta[columns], df_sa[columns], df_gs[columns]])
    max_per_trial = df.groupby(['Method', 'Trial'])['Utility'].max()
    mean_std = max_per_trial.groupby(level='Method').agg(['min', 'max', 'mean', 'std'])
    mean_std['rel_std'] = mean_std['std']/mean_std['mean']
    return mean_std

def get_policies(folder):
    df_sur, vals = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_randomsearch(folder)

    xs = dict()
    for df in [df_sur, df_meta, df_sa, df_gs]:
        xmethod = df.iloc[df['Utility'].idxmax()][df.columns.str.contains('q_swap|Method')] 
        xs[xmethod['Method']] = xmethod.drop('Method')

    x_df = pd.DataFrame.from_records(xs).T
    return x_df, xs, vals

def get_exhaustive(folder):
    method_names = ['Surrogate', 'Meta', 'Simulated Annealing', 'Random Search']
    dfs = [None]*4
    for name in glob.glob(f'{folder}/Results_*.csv'):
        df = pd.read_csv(name)
        method = df.Method[0]
        index = method_names.index(method)
        if method == 'Random Search':
            df['Method'] = 'Uniform Random Search'
        dfs[index] = df
    df = pd.concat(dfs, axis=0)
    timelimit = re.findall('\d+', folder.split('/')[-1])[0]
    df['Time Limit [h]'] = timelimit
    df = df.drop('Unnamed: 0' , axis=1)
    return df

def plot_from_exhaustive(folder, show=True):
    x_df, _, _ = get_policies(folder)
    df = get_exhaustive(folder)
    
    df = df.melt(id_vars=['Method', 'Aggregated Number of Virtual Neighbors'], var_name='User', value_name='Number of Virtual Neighbors')
    df['User'] = df['User'].apply(lambda x: str.replace(x, 'Node', ''))
    df = df.merge(x_df, left_on='Method', right_index=True, how='left')
    if show: 
        markers = ['o', '^', 'v', 's']
        fig, axs = plt.subplots(1,1, figsize=(5,3))
        sns.pointplot(data= df, x='Method', y='Aggregated Number of Virtual Neighbors', hue='Method', ax=axs, errorbar='se', markers=markers, legend=True, linestyles=['']*4, native_scale=True)
        axs.grid()
        plt.title('Aggregated Number of Virtual Neighbors')
        plt.ylabel('Number of Virtual Neighbors')
        plt.tight_layout()
        plt.show()
    return df


def plot_from_exhaustive_multiple(folders, show=True):
    dfs = []
    for folder in folders:
        x_df, _, _ = get_policies(folder)
        df = get_exhaustive(folder)
        
        df = df.melt(id_vars=['Method', 'Aggregated Number of Virtual Neighbors', 'Time Limit [h]'], var_name='User', value_name='Number of Virtual Neighbors')
        df['User'] = df['User'].apply(lambda x: str.replace(x, 'Node', ''))
        df = df.merge(x_df, left_on='Method', right_index=True, how='left')
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0)
    if show: 
        markers = ['o', '^', 'v', 's']
        fig, axs = plt.subplots(1,1, figsize=(5,3))
        sns.pointplot(data= df, x='Time Limit [h]', y='Aggregated Number of Virtual Neighbors', hue='Method', ax=axs, errorbar='se', markers=markers, legend=True, linestyles=['-']*4, native_scale=True)
        axs.grid()
        plt.title('Aggregated Number of Virtual Neighbors for Different Time Limits')
        plt.ylabel('Number of Virtual Neighbors')
        plt.tight_layout()
        plt.show()
    return df

if __name__ == '__main__':
    folder = '../../surdata/cd_10h'
    # df = get_performance_distribution_per_method(folder)
    # print(df)

    # df_timeprofiling, df_rel_timeprofiling = read_pkl_surrogate_timeprofiling(folder)
    # print(df_rel_timeprofiling.std())

    # # df_xs, xs, vals = get_policies(folder)
    # # print('xs \n', vals['user'][0]) 

    # plot_from_exhaustive(folder)

    error, acquisition = read_pkl_surrogate_benchmarking(folder)

    # folders = ['../../surdata/cd_1h', '../../surdata/cd_5h', '../../surdata/cd_10h']
    # df = plot_from_exhaustive_multiple(folders)
    # print(df)