import pickle
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
import src
import config

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
from cycler import cycler

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
    'legend.title_fontsize': font
})

folders = ['cd_N10_23tree_0.2h', 'cd_N10_34tree_1h', 'cd_N10_10square_3h']
methods = ['Surrogate', 'Meta', 'Simulated Annealing', 'Search']

def load_from_pkl(folder):
    axs = []
    for name in glob.glob(f'../../surdata/{folder}/AX_*.pkl'):
        with open(name,'rb') as file: axs.append(pickle.load(file))

    surs = []
    for name in glob.glob(f'../../surdata/{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))

    sas = []
    for name in glob.glob(f'../../surdata/{folder}/SA_*.pkl'):
        with open(name,'rb') as file: sas.append(pickle.load(file))

    gss = []
    for name in glob.glob(f'../../surdata/{folder}/RS_*.pkl'):
        with open(name,'rb') as file: gss.append(pickle.load(file))

    data = pd.DataFrame()
    data[methods[3]] = [gs[0].objective.apply(lambda x: np.sum(x)).max() for gs in gss]
    data[methods[2]] = [np.max(sa.objective) for sa in sas]
    data[methods[1]] = [np.max(ax[0].get_trials_data_frame()['evaluate']) for ax in axs]
    data[methods[0]] = [np.sum(sur.y, axis=1).max() for sur in surs]

    data = data/data.max().max()
    data['folder'] = folder
    return data

dfs = pd.concat([load_from_pkl(folder) for folder in folders], axis=0)
df_plot = dfs.melt(id_vars='folder', value_name='\# Virtual Neighbors', var_name='Method')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_plot, x='folder', y='\# Virtual Neighbors', hue='Method', ax=ax, palette="Pastel1")
sns.stripplot(x='folder', y='\# Virtual Neighbors', hue='Method', data=df_plot, jitter=True, ax=ax, dodge=True, palette='dark:#404040', alpha=0.6, legend=False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_xticklabels(['(2,3) Tree', '(3,4) Tree', '(10,10) Grid'])
ax.set_xlabel('')
ax.set_ylabel('VNM Ratio')
plt.grid()
plt.tight_layout(pad=1, rect=[0, 0, 0.75, 1])
plt.show()

# dfs = [df_sur, df_ax, df_gs, df_sa]#
# dfs_obj = []
# for df in dfs:
#     dfs_obj.append(df[['index', 'Objective', 'Method']])
# dfs = pd.concat(dfs_obj)
# dfs['Objective'] = dfs['Objective']
# dfs['# Optimization steps'] = dfs['index']

# fig, ax = plt.subplots()
# g = sns.lineplot(data = dfs[dfs['Objective']>0], x='# Optimization steps', y='Objective', hue='Method', style='Method', markers='^') 
# plt.title(f'Quantum Switch with {nnodes} leaf nodes')
# plt.gcf().set_size_inches(15,7)
# g.grid(which='major', color='w', linewidth=1.0)
# g.grid(which='minor', color='w', linewidth=0.5)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)
# ax.legend(fancybox=True, framealpha=0.5)
# ax.legend(loc='upper right')
# plt.tight_layout()
# plt.show()

