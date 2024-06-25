"""
Plotting tools
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re

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

def get_exhaustive(folders):
    dfs_total = []
    method_names = ['Surrogate', 'Meta', 'Simulated Annealing', 'Random Search']
    for i, folder in enumerate(folders):
        dfs = [None]*4
        for name in glob.glob(f'{folder}/Results_*.csv'):
            df_read = pd.read_csv(name, index_col=0)
            method_index = method_names.index(df_read['Method'].unique())
            dfs[method_index] = df_read
        df = pd.concat(dfs, axis=0)
        df['Time Limit [h]'] = [1,5,10][i]
        dfs_total.append(df)
    df_result = pd.concat(dfs_total, axis=0)
    return df_result

def plot_from_exhaustive_multiple(folders, show=True):
    df = get_exhaustive(folders)
    if show: 
        markers = ['o', '^', 'v', 's']
        fig, axs = plt.subplots(1,1, figsize=(5,3))
        sns.pointplot(data= df, x='Time Limit [h]', y='Aggregated Number of Virtual Neighbors', hue='Method', ax=axs, errorbar='se', markers=markers, legend=True, linestyles=['-']*4, native_scale=True)
        axs.grid()
        plt.tight_layout()
        plt.show()
    return df

def get_performance_distribution_per_method(folder):
    dfs_methods = []
    mapping = {'Surrogate':'SU', 'Meta':'AX', 'Simulated Annealing':'SA', 'Random Search':'RS'}
    for key, value in mapping.items():
        dfs = []
        for i,name in enumerate(glob.glob(folder + f'/{value}_*.pkl')): 
            with open(name,'rb') as file: dfs.append(pd.read_pickle(file))
            dfs[i]['Trial'] = i
        df = pd.concat(dfs, axis=0).reset_index()
        df['Method'] = key
        dfs_methods.append(df)

    df =pd.concat(dfs_methods, axis=0)   
    columns = ['Trial', 'Method', 'objective']
    df['Utility'] = df['objective']
    max_per_trial = df.groupby(['Method', 'Trial'])['Utility'].max()
    distr = max_per_trial.groupby(level='Method').agg(['min', 'max', 'mean', 'std'])
    distr['rel_std'] = distr['std']/distr['mean']
    return distr

def get_surrogate_timeprofiling(folder):
    dfs = []
    for i,name in enumerate(glob.glob(folder + f'/SU_*.pkl')): 
        with open(name,'rb') as file: dfs.append(pd.read_pickle(file))
        dfs[i]['Trial'] = i
    df = pd.concat(dfs, axis=0)
    times = df[df.columns[df.columns.astype('str').str.contains('\[s\]|Trial')]]
    times = times.drop_duplicates(ignore_index=True)
    relative = times.drop('Trial', axis=1).agg('mean')/times.drop('Trial', axis=1).agg('mean')['Total [s]']
    return times, relative, np.mean(np.mean(times.groupby('Trial').count()))

if __name__ == '__main__':
    
    
    # exhaustive run results (main text)
    # folders = [f'../../surdata/cd_{i}h/' for i in [1,5,10]]
    # plot_from_exhaustive_multiple(folders)

    # performance distribution (Supplementary Notes)
    time = 1
    folder = f'../../surdata/cd_{10}h/'
    distr = get_performance_distribution_per_method(folder)
    print(distr)

    # time profiling (Supplementary Notes)
    # print('\n')
    # times, relative, cycles = get_surrogate_timeprofiling(folder)
    # print('Overall:\n', times)
    # print('\n')
    # print('Relative:\n', relative)
    # print('\n')
    # print('Mean number of cycles:', cycles)

