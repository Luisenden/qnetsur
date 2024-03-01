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
# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.size': 40,  
# })
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
for i,sur in enumerate(surs):
    df_sur_y = pd.DataFrame(sur.y, columns=column_names)
    df_sur_y['sum mean'] = df_sur_y.sum(axis=1)
    df_sur_y['Objective Std'] = np.sqrt(np.sum(np.square(sur.y_std), axis=1)) # std of sum (assuming independent trials): sqrt of sum of variances 
    df_sur_y['Objective Mean'] = (df_sur_y['sum mean'] + sur.X_df.drop(['Iteration'], axis=1).sum(axis=1)/config.m_max).astype('float')
    df_sur_y[f'Trial'] = i
    grouped_iteration = sur.X_df.join(df_sur_y).groupby(['Iteration'], as_index=False).apply(lambda x: x.loc[x['Objective Mean'].idxmax()])
    dfs_sur.append(grouped_iteration)


df_sur = pd.concat(dfs_sur).reset_index()
df_sur['lower'] = df_sur['Objective Mean'] - df_sur['Objective Std']
df_sur['upper'] = df_sur['Objective Mean'] + df_sur['Objective Std']
df_sur['index'] = df_sur['Iteration']
df_sur['Method'] = 'Surrogate Optimization'
print(df_sur)


# fig, axs = plt.subplots(2, 5, figsize=(20, 10), sharex=True, sharey=True)
# axs = axs.flatten() 
# for trial in range(10):
#     df_plot = df_sur[df_sur.Trial == trial]
#     ax = axs[trial]
#     sns.lineplot(data=df_plot, x='Iteration', y='Objective Mean', ci=None, ax=ax)
#     ax.fill_between(df_plot['Iteration'], df_plot['lower'], df_plot['upper'], alpha=0.2)
#     ax.set_title(f'Trial {trial}')

# plt.tight_layout()
# plt.show() 

df_ax = pd.concat([ax[0].get_trials_data_frame() for ax in axs]).reset_index()
paramcolumns = df_ax.columns[df_ax.columns.str.contains('mem_size')]
df_ax['Objective Mean'] = df_ax['mean'] + df_ax[paramcolumns].sum(axis=1)/config.m_max
df_ax['Method'] = 'Meta Optimization'
print(df_ax)


df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
paramcolumns = df_gs.columns[df_gs.columns.str.contains('mem_size')]
df_gs['sum'] = df_gs['objective'].apply(lambda x: np.sum(x))
df_gs['Objective Mean'] = df_gs['sum'] + df_gs[paramcolumns].sum(axis=1)/config.m_max
df_gs['Method'] = 'Random Grid Search'

df_sa = pd.concat(sas).reset_index()
paramcolumns = df_sa.columns[df_sa.columns.str.contains('mem_size')]
df_sa['sum'] = df_sa['objective']
df_sa['Objective Mean'] = df_sa['sum'] + df_sa[paramcolumns].sum(axis=1)/config.m_max
df_sa['Method'] = 'Simulated annealing'

dfs = [df_sur, df_ax, df_gs, df_sa]#
dfs_obj = []
for df in dfs:
    dfs_obj.append(df[['index', 'Objective Mean', 'Method']])
dfs = pd.concat(dfs_obj)
dfs['\# Optimization steps'] = dfs['index']
fig, ax = plt.subplots()
g = sns.lineplot(data = dfs[dfs['index']<90], x='\# Optimization steps', y='Objective Mean', hue='Method', style='Method', markers='^', markersize=5) 
plt.title(f'Quantum Network with {nnodes}')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
ax.legend(fancybox=True, framealpha=0.5)
ax.legend(loc='upper right')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.ylabel('\# Completed Requests')
plt.grid()
plt.tight_layout()
plt.show()

# print(sur.X_df)
# df_sur_X_plot = sur.X_df.melt(id_vars='Iteration', value_name='\# Memory Qubits', var_name='Node')
# sns.lineplot(df_sur_X_plot, x='Iteration', y='\# Memory Qubits', hue='Node')
# plt.legend([],[], frameon=False)
# plt.show()
