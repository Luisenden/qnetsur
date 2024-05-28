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

def get_policies(folder, suffix):
    df_sur = pd.read_csv(folder+'SU'+suffix)
    df_meta = pd.read_csv(folder+'AX'+suffix)
    df_meta['objective'] = df_meta['evaluate']
    df_sa = pd.read_csv(folder+'SA'+suffix)
    df_rs = pd.read_csv(folder+'RS'+suffix)

    xs = dict()
    for df in [df_sur, df_meta, df_sa, df_rs]:
        xmethod = df.iloc[df['objective'].idxmax()][df.columns.str.contains('q_swap|Method')] 
        xs[xmethod['Method']] = xmethod.drop('Method')

    x_df = pd.DataFrame.from_records(xs).T
    return x_df

def get_exhaustive(folder):
    method_names = ['Surrogate', 'Meta', 'Simulated Annealing', 'Random Gridsearch']
    dfs = [None]*4
    for name in glob.glob(f'{folder}/Results_*.csv'):
        df = pd.read_csv(name)
        method = df.Method[0]
        index = method_names.index(method)
        if method == 'Random Gridsearch':
            df['Method'] = 'Uniform Random Search'
        dfs[index] = df
    df = pd.concat(dfs, axis=0)
    timelimit = re.findall('(\d+)h', folder)[0]
    df['Time Limit [h]'] = timelimit
    return df

def plot_from_exhaustive_multiple(folders, suffixes, show=True):
    dfs = []
    for folder, suffix in zip(folders, suffixes):
        x_df = get_policies(folder, suffix)
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

def get_performance_distribution_per_method(folder,suffix):
    df_sur = pd.read_csv(folder+'SU'+suffix)
    df_meta = pd.read_csv(folder+'AX'+suffix)
    df_meta['objective'] = df_meta['evaluate']
    df_sa = pd.read_csv(folder+'SA'+suffix)
    df_rs = pd.read_csv(folder+'RS'+suffix)

    columns = ['Trial', 'Method', 'objective']
    df = pd.concat([df_sur[columns], df_meta[columns], df_sa[columns], df_rs[columns]])
    df['Utility'] = df['objective']
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

if __name__ == '__main__':
    
    
    # exhaustive run results (main text)
    folders = [f'../../surdata/cd_randtree-100-{i}h/' for i in [1,5,10]]
    suffixes = [f'_randtree-100_{i}h.csv'for i in [1,5,10]]
    plot_from_exhaustive_multiple(folders, suffixes)

    # performance distribution (Supplementary Notes)
    time = 1
    folder = f'../../surdata/cd_randtree-100-{time}h/'
    distr = get_performance_distribution_per_method(folder, f'_randtree-100_{time}h.csv')
    print(distr)

    # time profiling (Supplementary Notes)
    times, relative, cycles = get_surrogate_timeprofiling(folder+f'SU_randtree-100_{time}h.csv')
    print('Overall:\n', times)
    print('Relative:\n', relative)
    print('Mean number of cycles:', cycles)