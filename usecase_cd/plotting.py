"""
Plotting tools
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import argparse
import re

plt.style.use("seaborn-v0_8-paper")
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
plt.rcParams.update({'lines.markeredgewidth': 0.1})
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
    
    basic_sols_files = glob.glob(f"{folder}/*.csv")
    dfs_basic_sols = []
    names = [r'$q_{swap}=0$', r'$q_{swap}=1$', r'$q_{swap}$ ~ Uniform(0,1)', r'$q_{swap,v=1}=0$']
    for i, file in enumerate(basic_sols_files):
        dfs_basic_sols.append(pd.read_csv(file, index_col=0))
        dfs_basic_sols[i]['Basic solution'] = names[i]
    df_basic_sols = pd.concat(dfs_basic_sols, axis=0)

    distr_max = []
    for i,f in enumerate(folders):
        _, max_per_method = get_performance_distribution_per_method(f)
        max_per_method = max_per_method.reset_index()
        max_per_method['Time Limit [h]'] = [1,5,10][i]
        max_per_method['Aggregated Number of Virtual Neighbors'] = max_per_method['Utility']
        distr_max.append(max_per_method)
    df_max = pd.concat(distr_max, axis=0)
    df_max['standard error'] = 1.9 # estimated mean standard deviation of n=20 (see associated notebook)

    if show: 

        markers = ['o', '^', 's', 'v']
        fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.05})
        method_names = ['Surrogate', 'Meta', 'Simulated Annealing', 'Random Search']
        for i,method in enumerate(method_names):
            ax_top.errorbar(x=df_max[df_max['Method']==method]['Time Limit [h]'], y= df_max[df_max['Method']==method]['Aggregated Number of Virtual Neighbors'], 
                         yerr=df_max[df_max['Method']==method]['standard error'], linestyle='dashed', marker=markers[i], capsize=0, markersize=6 , capthick=1, alpha=0.5)
        sns.pointplot(data= df, x='Time Limit [h]', y='Aggregated Number of Virtual Neighbors', hue='Method', ax=ax_top, errorbar='se', err_kws={'alpha':1}, markers=markers, legend=True, linestyles=['']*4, native_scale=True, markersize=6)
        sns.pointplot(data= df, x='Time Limit [h]', y='Aggregated Number of Virtual Neighbors', hue='Method', ax=ax_bottom, errorbar='se', err_kws={'alpha':1}, markers=markers, legend=True, linestyles=['']*4, native_scale=True, markersize=6)
        
        ax_top.grid()
        ax_bottom.grid()
        ax_bottom.hlines(np.mean(dfs_basic_sols[1]['Aggregated Number of Virtual Neighbors']), xmin=1, xmax=10, linestyles='dashed', colors='black', label=r"$U_{(1)}$ baseline quantity")
        ax_top.set_ylim(250,300)   
        ax_bottom.set_ylim(150,200)

        sns.despine(ax=ax_bottom)
        sns.despine(ax=ax_top, bottom=True)

        ax = ax_top
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

        ax2 = ax_bottom
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        #remove one of the legend
        ax_top.legend_.remove()
        ax_bottom.legend(loc='lower right', bbox_to_anchor=[0.9,0.01])
        ax_top.set_ylabel('')
        ax_bottom.set_ylabel('Aggregated Number of Virtual Neighbors', loc='bottom')
        plt.xlabel(r'Time Limit $T$ [h]')
        plt.tight_layout()
        plt.show()
    return df


def get_performance_distribution_per_method(folder):
    dfs_methods = []
    mapping = {'Surrogate':'SU', 'Meta':'AX', 'Simulated Annealing':'SA', 'Random Search':'RS'}
    for key, value in mapping.items():
        dfs = []
        for i,name in enumerate(glob.glob(folder + f'/{value}_*.csv')): 
            with open(name,'rb') as file: dfs.append(pd.read_csv(file, index_col=0))
            dfs[i]['Trial'] = i
        df = pd.concat(dfs, axis=0).reset_index()
        df['Method'] = key
        dfs_methods.append(df)

    df =pd.concat(dfs_methods, axis=0)   
    columns = ['Trial', 'Method', 'objective']
    df['Utility'] = df['objective']
    max_per_trial = df.groupby(['Method', 'Trial'], sort=False)['Utility'].max()
    distr = max_per_trial.groupby(level='Method').agg(['min', 'max', 'mean', 'std'])
    distr['rel_std'] = distr['std']/distr['mean']

    max_per_method = df.groupby(['Method'], sort=False)['Utility'].max()
    return distr, max_per_method

def get_surrogate_timeprofiling(folder):
    dfs = []
    for i,name in enumerate(glob.glob(folder + f'/SU_*.csv')): 
        with open(name,'rb') as file: dfs.append(pd.read_csv(file, index_col=0))
        dfs[i]['Trial'] = i
    df = pd.concat(dfs, axis=0)
    times = df[df.columns[df.columns.astype('str').str.contains(r'\[s\]|Trial')]]
    times = times.drop_duplicates(ignore_index=True)
    relative = times.drop('Trial', axis=1).agg('mean')/times.drop('Trial', axis=1).agg('mean')['Total [s]']
    return times, relative, np.mean(np.mean(times.groupby('Trial').count()))

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description="set directory to data to be plotted")
    
    parser.add_argument(
        "--folder",
        type=str,
        help="Set path/to/QNETSUR-DATA/continuous_distribution_protocols/. Type: str"
    )

    # Parse 
    args, _ = parser.parse_known_args()
    folder = args.folder
    Ts = [1,5,10]
    folders = [folder+f'cd_{i}h/' for i in Ts]

    # exhaustive run results (main text)
    print(folders)
    plot_from_exhaustive_multiple(folders)

    # performance distribution (supplementary notes)
    for i, folder in enumerate(folders):
        distr, _ = get_performance_distribution_per_method(folder)
        print(f'\nPerformance distribution with T={Ts[i]} hour:')
        print(distr)

        #time profiling (supplementary notes)
        print('\n')
        times, relative, cycles = get_surrogate_timeprofiling(folder)
        print('Overall:\n', times)
        print('\n')
        print('Relative:\n', relative)
        print('\n')
        print('Mean number of cycles:', cycles)