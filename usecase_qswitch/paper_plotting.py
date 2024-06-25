"""
Plotting tools
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

plt.style.use("seaborn-paper")
font = 16
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

import sys
sys.path.append('../')
sys.path.append('../src')

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
        df['Time Limit [h]'] = [0.5,3][i]
        dfs_total.append(df)
    df_result = pd.concat(dfs_total, axis=0)
    return df_result

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

def plot_progress(folder):
    dfs_methods = []
    columns = ['Trial', 'Method', 'objective']
    mapping = {'Surrogate':'SU', 'Meta':'AX', 'Simulated Annealing':'SA', 'Random Search':'RS'}
    for key, value in mapping.items():
        dfs = []
        for i,name in enumerate(glob.glob(folder + f'/{value}_*.pkl')): 
            with open(name,'rb') as file: df_raw = pd.read_pickle(file)
            if any(df_raw.columns.str.contains('Iteration')):
                df_raw = df_raw.merge(df_raw.groupby(['Iteration'])['objective'].max(numeric_only=True), on='Iteration', suffixes=('_x', ''))
                df_raw = df_raw[['Iteration', 'objective']].drop_duplicates().reset_index()
            df_raw['Trial'] = i
            dfs.append(df_raw.reset_index())
        df = pd.concat(dfs, axis=0)
        df['Method'] = key
        dfs_methods.append(df[columns])

    df =pd.concat(dfs_methods, axis=0).rename_axis('Iteration').reset_index()
    best_trial = df.iloc[df.groupby('Method')['objective'].idxmax()][['Method','Trial']]
    df = pd.merge(df, best_trial, on=['Method', 'Trial'], how='inner')


    sns.lineplot(df, x='Iteration', y='objective', hue='Method', legend=True) #units='Trial', estimator=None
    plt.ylabel('Utility')
    plt.grid()
    plt.xlabel('Optimization Cycle')
    plt.tight_layout()
    plt.show()
    return df

def plot_from_exhaustive_multiple(folders):
    df = get_exhaustive(folders)
    markers = ['o', '^', 'v', 's']
    fig, axs = plt.subplots(1,1, figsize=(5,3))
    #sns.pointplot(data= df, x='Time Limit [h]', y='Utility', hue='Method', ax=axs, errorbar='se', markers=markers, linestyles=['-']*4)
    sns.barplot(data= df, x='Time Limit [h]', y='Utility', hue='Method', ax=axs, errorbar='se')
    plt.ylim([10,11.5])
    plt.ylabel('Aggregated Utility')
    axs.grid()
    plt.tight_layout()
    plt.show()
    return df

def plot_exhaustive_per_user(folder):
    df = get_exhaustive(folder)

    columns_utility = df.columns.str.fullmatch('\d+')
    df_utility = df.melt(id_vars='Method', value_vars=df.columns[columns_utility], value_name='Utility', var_name='User')
    fig, axs = plt.subplots(3,1)

    columns_rates = df.columns.astype('str').str.contains('Rate')
    df_rates = df.melt(id_vars='Method', value_vars=df.columns[columns_rates][1:], value_name='Rate [Hz]', var_name='User')
    sns.lineplot(df_rates, x='User', y='Rate [Hz]', hue='Method', style='Method', markers=['o', '^', 'v', 's'], ax=axs[0], legend=False)
    axs[0].grid()
    axs[0].set_xticklabels([i for i in range(5)])

    columns_fidel = df.columns.astype('str').str.contains('Fidelity')
    df_rates = df.melt(id_vars='Method', value_vars=df.columns[columns_fidel][1:], value_name='Fidelity', var_name='User')
    sns.lineplot(df_rates, x='User', y='Fidelity', hue='Method', style='Method', markers=['o', '^', 'v', 's'], ax=axs[1], legend=True)
    axs[1].grid()
    axs[1].set_xticklabels([i for i in range(5)])
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    sns.lineplot(df_utility, x='User', y='Utility', hue='Method', style='Method', markers=['o', '^', 'v', 's'], ax=axs[2], legend=False)
    axs[2].grid()
    plt.tight_layout(pad=0.1)
    plt.show()


    


if __name__ == '__main__':

    # five users at varying distances
    # folders = ['../../surdata/qswitch_30min/']
    # plot_from_exhaustive_multiple([f'../../surdata/qswitchtest/'])
   # plot_exhaustive_per_user(['../../surdata/qswitch_30min/'])
    

    # performance distribution (Supplementary Notes)
    folder = f'../../surdata/qswitch_30min/'
    distr = get_performance_distribution_per_method(folder)
    print(distr)

    # # time profiling (Supplementary Notes)
    # print('\n')
    # times, relative, cycles = get_surrogate_timeprofiling(folder)
    # print('Overall:\n', times)
    # print('\n')
    # print('Relative:\n', relative)
    # print('\n')
    # print('Mean number of cycles:', cycles)

    # plot progress
    # df = plot_progress(f'../../surdata/qswitch_15min/')
    # print(df.iloc[df.groupby('Method')['objective'].idxmax()][['Method','Trial']])