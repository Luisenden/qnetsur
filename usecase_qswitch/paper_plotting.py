"""
Plotting tools
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

def get_surrogate_timeprofiling(file):

    times = pd.read_csv(file)
    times = times[times.columns[times.columns.str.contains('\[s\]|Trial')]]
    times = times.drop_duplicates(ignore_index=True)
    relative = times.drop('Trial', axis=1).agg('mean')/times.drop('Trial', axis=1).agg('mean')['Total [s]']

    return times, relative, np.mean(np.mean(times.groupby('Trial').count()))

def get_performance_distribution_per_method(folder,suffix):
    df_sur = pd.read_csv(folder+'SU_'+suffix)
    df_meta = pd.read_csv(folder+'AX_'+suffix)
    df_meta['objective'] = df_meta['evaluate']
    df_sa = pd.read_csv(folder+'SA_'+suffix)
    df_rs = pd.read_csv(folder+'RS_'+suffix)

    columns = ['Trial', 'Method', 'objective']
    df = pd.concat([df_sur[columns], df_meta[columns], df_sa[columns], df_rs[columns]])
    df['Utility'] = df['objective']
    max_per_trial = df.groupby(['Method', 'Trial'])['Utility'].max()
    mean_std = max_per_trial.groupby(level='Method').agg(['min', 'max', 'mean', 'std'])
    mean_std['rel_std'] = mean_std['std']/mean_std['mean']
    return mean_std


if __name__ == '__main__':

    # five users at varying distances
    df = pd.read_csv('../../surdata/qswitch/Results_qswitch_5users_T30min.csv')
    plot_from_exhaustive(df)

    # performance distribution (Supplementary Notes)
    folder = '../../surdata/qswitch/'
    suffix = 'qswitch6-30min.csv'
    distr = get_performance_distribution_per_method(folder, suffix)
    print(distr)

    # time profiling (Supplementary Notes)
    times, relative, cycles = get_surrogate_timeprofiling('../../surdata/qswitch/SU_qswitch6-30min.csv')
    print('Overall:\n', times)
    print('Relative:\n', relative)
    print('Mean number of cycles:', cycles)
