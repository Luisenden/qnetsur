"""
Plotting tools
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

def plot_from_exhaustive(folder):
    x_df= pd.read_csv(folder+'Best_found_solutions.csv',index_col=0)
    print(x_df)

    method_names = ['Surrogate', 'Meta', 'Simulated Annealing', 'Random Search', 'Even', 'Wu et. al, 2021']
    dfs = [None]*6
    for name in glob.glob(f'{folder}/Results_*.csv'):
        df = pd.read_csv(name)
        method = df.Method[0]
        index = method_names.index(method)
        dfs[index] = df

    df = pd.concat(dfs, axis=0)
    df = df.drop('Unnamed: 0' , axis=1)
    df = df.melt(id_vars=['Method', 'Aggregated Completed Requests'], var_name='User', value_name='Number of Completed Requests')
    df['User'] = df['User'].apply(lambda x: str.replace(x, 'Node', ''))
    df = df.merge(x_df, left_on='Method', right_index=True, how='left')
    markers = ['o', '^', 'v', 's', 'd', 'P']
    fig, axs = plt.subplots(1,1, figsize=(5,3))
    sns.pointplot(data= df, x='Total Number of Allocated Memories', y='Aggregated Completed Requests', hue='Method', ax=axs, errorbar='se', markers=markers, legend=True, linestyles=['']*6, native_scale=True)
    axs.grid()
    plt.ylabel('Total number of completed requests')
    plt.tight_layout()
    plt.show()

def get_performance_distribution_per_method(folder,suffix):
    df_sur = pd.read_csv(folder+'SU'+suffix)
    df_sur['Utility'] = df_sur['objective']
    df_meta = pd.read_csv(folder+'AX'+suffix)
    df_sa = pd.read_csv(folder+'SA'+suffix)
    df_rs = pd.read_csv(folder+'RS'+suffix)
    df_rs['Utility'] = df_rs['objective']

    columns = ['Trial', 'Method', 'Utility']
    df = pd.concat([df_sur[columns], df_meta[columns], df_sa[columns], df_rs[columns]])
    print(df)
    max_per_trial = df.groupby(['Method', 'Trial'])['Utility'].max()
    mean_std = max_per_trial.groupby(level='Method').agg(['min', 'max', 'mean', 'std'])
    mean_std['rel_std'] = mean_std['std']/mean_std['mean']
    return mean_std

def get_surrogate_timeprofiling(file):

    times = pd.read_csv(file)
    times = times[times.columns[times.columns.str.contains('\[s\]|Trial')]]
    times = times.drop_duplicates(ignore_index=True)
    relative = times.drop('Trial', axis=1).agg('mean')/times.drop('Trial', axis=1).agg('mean')['Total [s]']
    return times, relative, np.mean(np.mean(times.groupby('Trial').count()))

def plot_policies(file):
    df = pd.read_csv(file, index_col=0).T
    df = df.drop('Total Number of Allocated Memories').reset_index()
    df = df.drop('index', axis=1).T
    df.columns = ["NU", "StarLight", "UChicago_PME", "UChicago_HC", "Fermilab_1", \
                  "Fermilab_2", "Argonne_1", "Argonne_2", "Argonne_3"]
    df = df.reset_index(names='Method').T
    # df = df.melt(id_vars='Method', var_name='Node', value_name='Number of Memories')
    # fig, ax = plt.subplots()
    # sns.barplot(df, x='Node', y='Number of Memories', hue='Method')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 
    # plt.grid()
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    # plt.show()
    return df.to_latex()

if __name__ == '__main__':
    # folder = '../../surdata/rb_budget_25h/'
    
    # # best found solutions (Supplementary Notes)
    # best_solutions= pd.read_csv(folder+'Best_found_solutions.csv',index_col=0)
    # print('Best Found Solutions:\n', best_solutions)
    
    # # exhaustive run results (main text)
    # plot_from_exhaustive(folder)

    # # performance distribution (Supplementary Notes)
    # distr = get_performance_distribution_per_method(folder, '_rb_starlight_budget_25h.csv')
    # print(distr)

    # # time profiling (Supplementary Notes)
    # times, relative, cycles = get_surrogate_timeprofiling(folder+'SU_rb_starlight_budget_25h.csv')
    # print('Overall:\n', times)
    # print('Relative:\n', relative)
    # print('Mean number of cycles:', cycles)

    print(plot_policies('../../surdata/rb_budget_25h/Best_found_solutions.csv'))
