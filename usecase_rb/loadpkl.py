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
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 40,  
})

import warnings
warnings.filterwarnings("ignore")


folder = 'rb_N10_24h'

axs = []
for name in glob.glob(f'../../surdata/{folder}/AX_*.pkl'):
    with open(name,'rb') as file: axs.append(pickle.load(file))

surs = []
for name in glob.glob(f'../../surdata/{folder}/SU_*.pkl'): #Sur_qswitch_nleafnodes{nnodes}_{time}*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

sas = []
for name in glob.glob(f'../../surdata/{folder}/SA_*.pkl'):
    with open(name,'rb') as file: sas.append(pickle.load(file))

gss = []
for name in glob.glob(f'../../surdata/{folder}/GS_*.pkl'):
    with open(name,'rb') as file: gss.append(pickle.load(file))


dfs_sur = []
nnodes=9
column_names = pd.Series(range(nnodes)).astype('str')
for sur in surs:
    df_sur_y = pd.DataFrame(sur.y, columns=column_names)
    print(sur.vals)
    print(df_sur_y)
    df_sur_y['sum'] = df_sur_y.sum(axis=1)
    df_sur_y['Objective'] = df_sur_y['sum'] + sur.X_df.drop(['Iteration'], axis=1).sum(axis=1)/config.m_max
    grouped_iteration = sur.X_df.join(df_sur_y).groupby(['Iteration'])['Objective']
    df_means = grouped_iteration.mean()
    dfs_sur.append(df_means)


df_sur = pd.concat(dfs_sur).reset_index()
df_sur['index'] = df_sur['Iteration']
df_sur['Method'] = 'Surrogate Optimization'

df_ax = pd.concat([ax[0].get_trials_data_frame() for ax in axs]).reset_index()
paramcolumns = df_ax.columns[df_ax.columns.str.contains('mem_size')]
df_ax['Objective'] = df_ax['mean'] + df_ax[paramcolumns].sum(axis=1)/config.m_max
df_ax['Method'] = 'Meta Optimization'


df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
paramcolumns = df_gs.columns[df_gs.columns.str.contains('mem_size')]
df_gs['sum'] = df_gs['objective'].apply(lambda x: np.sum(x))
df_gs['Objective'] = df_gs['sum'] + df_gs[paramcolumns].sum(axis=1)/config.m_max
df_gs['Method'] = 'Random Grid Search'

df_sa = pd.concat(sas).reset_index()
paramcolumns = df_sa.columns[df_sa.columns.str.contains('mem_size')]
df_sa['sum'] = df_sa['objective']
df_sa['Objective'] = df_sa['sum'] + df_sa[paramcolumns].sum(axis=1)/config.m_max
df_sa['Method'] = 'Simulated annealing'

dfs = [df_sur, df_ax, df_gs, df_sa]#
dfs_obj = []
for df in dfs:
    dfs_obj.append(df[['index', 'Objective', 'Method']])
dfs = pd.concat(dfs_obj)
dfs['\# Optimization steps'] = dfs['index']

# fig, ax = plt.subplots()
# g = sns.lineplot(data = dfs, x='\# Optimization steps', y='Objective', hue='Method', style='Method', markers='^', markersize=5) 
# plt.title(f'Quantum Network with {nnodes}')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)
# ax.legend(fancybox=True, framealpha=0.5)
# ax.legend(loc='upper right')
# plt.grid()
# plt.tight_layout()
# plt.savefig('test.pdf', dpi=100)

print(sur.X_df)
df_sur_X_plot = sur.X_df.melt(id_vars='Iteration', value_name='\# Memory Qubits', var_name='Node')
sns.lineplot(df_sur_X_plot, x='Iteration', y='\# Memory Qubits', hue='Node')
plt.show()
#df_sur_X_plot = df_sur_X.drop(['penalty'], axis=1).melt('Iteration', var_name='Node', value_name='Buffer size')
