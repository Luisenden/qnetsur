import pickle
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
import src
from config import *

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.3)

def reduce_to_means_per_iteration(df, group_size):
    df['Iteration'] = [i for i in range(len(df)//group_size) for _ in range(group_size)]
    return pd.DataFrame(df.groupby('Iteration').mean().to_numpy(), columns=['mean']) 

sims = []
for name in glob.glob('../../surdata/RB/sim_*'):
    with open(name,'rb') as file: sims.append(pickle.load(file))

weighted = [np.mean(sim) for sim in sims[0]]
even = [np.mean(sim) for sim in sims[1]]

axs = []
for name in glob.glob('../../surdata/RB/6h/Ax_*'):
    with open(name,'rb') as file: axs.append(pickle.load(file))

surs = []
for name in glob.glob('../../surdata/RB/6h/Sur_*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

sas = []
for name in glob.glob('../../surdata/RB/6h/SA_*'):
    with open(name,'rb') as file: sas.append(pickle.load(file))

gss = []
for name in glob.glob('../../surdata/RB/6h/GS_*'):
    with open(name,'rb') as file: gss.append(pickle.load(file))

print(np.sum(sims[1], axis=1).mean())

df_ax_list = [ax[0].get_trials_data_frame() for ax in axs]
df_ax = pd.concat(df_ax_list).reset_index()
df_ax['mean'] = df_ax['mean'] + df_ax.filter(regex='mem_size').mean(axis=1)/m_max
df_ax['Method'] = 'Meta Optimizer'


df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
df_gs['mean'] = df_gs['objective'].apply(lambda x: np.mean(x))
df_gs['mean'] = df_gs['mean'] + df_gs.filter(regex='mem_size').mean(axis=1)/m_max
df_gs['Method'] = 'Random Grid Search'

dfs_sur = []
dfs_sur_X = []
for sur in surs:
    df_sur_y = pd.DataFrame(np.mean(sur.y, axis=1)+np.mean(sur.X_df, axis=1)/m_max, columns=['mean'])
    df_sur = sur.X_df.join(df_sur_y).groupby(['Iteration'], as_index=False).mean()
    dfs_sur.append(df_sur)

    df_sur_X = sur.X_df.groupby(['Iteration'], as_index=False).mean()
    df_sur_X['sum'] = df_sur_X.drop(['Iteration'], axis=1).sum(axis=1)
    dfs_sur_X.append(df_sur_X)

df_sur = pd.concat(dfs_sur).reset_index()
df_sur['Method'] = 'Our Surrogate Optimizer'
print(df_sur.iloc[df_sur['mean'].astype(float).idxmax()])

df_sur_X = pd.concat(dfs_sur_X)
df_sur_X.columns = ['# optimization steps', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'sum']
df_sur_X = df_sur_X.melt(id_vars='# optimization steps', var_name='Node', value_name='# memories per node')

df_sa = pd.concat(sas).reset_index()
df_sa['mean'] = df_sa['objective']
df_sa['mean'] = df_sa['mean'] + df_sa.filter(regex='mem_size').mean(axis=1)/m_max
df_sa['Method'] = 'Simulated Annealing'

dfs = [df_sur, df_ax, df_sa, df_gs]
dfs_obj = []
for df in dfs:
    dfs_obj.append(df[['index', 'mean', 'Method']])
dfs = pd.concat(dfs_obj)
dfs['# optimization steps'] = dfs['index']
dfs['weighted'] = np.mean(weighted)
dfs['# completed requests'] = dfs['mean']




# fig, axs = plt.subplots(1,1)
# g = sns.lineplot(data = dfs, x='# optimization steps', y='# completed requests', hue='Method', style='Method', markers=True, ax=axs) # plot the Number of Neighbours for all methods
# g.axes.axhline(np.mean(weighted), ls='--', color='red')
# g.axes.text(0.5, np.mean(weighted)+0.1, 'weighted policy [1]', fontsize=12, va='center', ha='left', color='red')
# handles, labels = axs.get_legend_handles_labels()
# axs.legend(handles=handles, labels=labels)
# axs.legend(fancybox=True, framealpha=0.5)
# axs.legend(loc='lower right')
# axs.set_title('Optimization of requests fulfilled')
# plt.show()

# fig, axs = plt.subplots(1,1)
# g = sns.lineplot(data = df_sur_X, x='# optimization steps', y='# memories per node', hue='Node', style='Node', markers=False, ax=axs, color=sns.color_palette('Dark2'), errorbar=None) 
# axs.set_title('Developement of memory sizes')
# sns.move_legend(axs, "upper left", bbox_to_anchor=(1, 1))
# plt.tight_layout()
# plt.show()
