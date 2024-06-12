"""
Plotting tools
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import glob

plt.style.use("seaborn-paper")
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

import sys
sys.path.append('../')
sys.path.append('../src')

import warnings
warnings.filterwarnings("ignore")


def plot_from_exhaustive(df):
    fig, axs = plt.subplots(3,1, figsize=(5,12))
    sns.lineplot(data= df, x='User', y='Utility', hue='Method', ax=axs[0], errorbar='se', err_style='bars', style='Method', markers=True, markersize=10)
    sns.lineplot(data= df, x='User', y='Rate [Hz]', hue='Method', ax=axs[1], errorbar='se', err_style='bars', style='Method', markers=True, markersize=10)
    g = sns.lineplot(data= df, x='User', y='Fidelity', hue='Method', ax=axs[2], errorbar='se', err_style='bars', style='Method', markers=True, markersize=10)

    for i in range(len(axs)):
        axs[i].grid(alpha=0.3)
        axs[i].legend().remove()
        axs[i].set_xlabel('')
    
    axs[0].set_title('Utility per User')
    axs[1].set_title('Rate [Hz] per User')
    axs[2].set_title('Fidelity per User')
    axs[2].set_xlabel('User')

    g.legend(loc='lower center', bbox_to_anchor=(0.5, -1.5), ncol=1)
    plt.tight_layout()
    plt.savefig('Images/QES-example2-users.pdf')
    plt.show()

    fig, axs = plt.subplots(figsize=(5,7))
    sns.barplot(data=df, x='Method', y='Aggregated Utility')
    plt.title(r'Aggregated Utility $U(\mathbf{s_\alpha})$')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Images/QES-example2-bars.pdf')
    plt.show()

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
    sns.lineplot(df, x='Iteration', y='objective', hue='Method') #units='Trial', estimator=None
    plt.show()


if __name__ == '__main__':

    # # five users at varying distances
    # df = pd.read_csv('../../surdata/qswitch/Results_qswitch_5users_T30min.csv')
    # plot_from_exhaustive(df)

    # performance distribution (Supplementary Notes)
    folder = f'../../surdata/qswitch_3h/'
    distr = get_performance_distribution_per_method(folder)
    print(distr)

    # time profiling (Supplementary Notes)
    print('\n')
    times, relative, cycles = get_surrogate_timeprofiling(folder)
    print('Overall:\n', times)
    print('\n')
    print('Relative:\n', relative)
    print('\n')
    print('Mean number of cycles:', cycles)

    # plot progress
    plot_progress(folder)