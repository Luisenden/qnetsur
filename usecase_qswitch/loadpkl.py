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
plt.style.use('seaborn')

nnodes = 5
time = 0.5

axs = []
for name in glob.glob(f'../../surdata/qswitch/Ax_qswitch_nleafnodes10_0.50h_objective-servernode_SEED42_01-24-2024_14:39:35.pkl'):
    with open(name,'rb') as file: axs.append(pickle.load(file))

surs = []
for name in glob.glob(f'../../surdata/qswitch/Sur_qswitch_nleafnodes3_0.10h_objective-servernode_SEED42_02-01-2024_13:57:59.pkl'): #Sur_qswitch_nleafnodes{nnodes}_{time}*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

# sas = []
# for name in glob.glob('../../surdata/CD/tree3-4/SA_*'):
#     with open(name,'rb') as file: sas.append(pickle.load(file))

gss = []
for name in glob.glob(f'../../surdata/qswitch/GS_qswitch_nleafnodes10_0.50h_objective-servernode_SEED42_01-24-2024_12:24:16.pkl'): #GS_qswitch_nleafnodes{nnodes}_{time}*'):
    with open(name,'rb') as file: gss.append(pickle.load(file))


dfs_sur = []
dfs_sur_X = []

nnodes = surs[0].vals['nnodes']
column_names = ['Objective 1', 'Objective 2']
for sur in surs:
    df_sur_y = pd.DataFrame(sur.y, columns=column_names)
    df_sur_y_sum = df_sur_y.sum(axis=1)
    df_sur_y_sum.name = 'Objective'

    
    mask = sur.X_df.join(df_sur_y_sum).groupby(['Iteration'])['Objective'].idxmax().values
    df_sur_y_raw = pd.DataFrame(sur.y_raw, columns=['Share server node', 'Capacity (MHz)']).iloc[mask]
    df_sur_y = df_sur_y.iloc[mask]

    df_sur = pd.concat([df_sur_y, df_sur_y_sum.iloc[mask], df_sur_y_raw], axis=1)
    dfs_sur.append(df_sur)

    df_sur_X = sur.X_df.iloc[mask]
    dfs_sur_X.append(df_sur_X)


df_sur = pd.concat(dfs_sur).reset_index()
df_sur['Method'] = 'Surrogate Optimization'
df_sur.reset_index(inplace=True)
df_sur['index'] = df_sur['level_0']

df_sur_X = pd.concat(dfs_sur_X)
df_sur_X_plot = df_sur_X.melt('Iteration', var_name='Node', value_name='Buffer size')
fig, ax = plt.subplots()
g = sns.lineplot(data = df_sur_X_plot, x='Iteration', y='Buffer size', hue='Node', style='Node', markers='^')
plt.show()

# fig, ax = plt.subplots()
# g = sns.lineplot(data = df_sur_y_raw[['Capacity (MHz)','Share server node']], markers='^')
# plt.yscale('log')
# plt.show()


# df_ax = pd.concat([ax[0] for ax in axs]).reset_index()
# df_ax['Method'] = 'Meta Optimization'
# df_ax['Objective'] = df_ax['evaluate'] 


# df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
# df_gs['Objective'] = df_gs['objective'].apply(lambda x: np.sum(x))
# df_gs['Method'] = 'Random Grid Search'

# df_sa = pd.concat(sas).reset_index()
# df_sa['mean'] = df_sa['objective']
# df_sa['Method'] = 'Simulated annealing'

dfs = [df_sur]#, df_gs, df_ax]#
dfs_obj = []
for df in dfs:
    dfs_obj.append(df[['index', 'Objective', 'Method']])
dfs = pd.concat(dfs_obj)
dfs['Objective'] = dfs['Objective']
dfs['# Optimization steps'] = dfs['index']

fig, ax = plt.subplots()
g = sns.lineplot(data = dfs, x='# Optimization steps', y='Objective', hue='Method', style='Method', markers='^') # plot the Number of Neighbours for all methods
#g = sns.lineplot(data = dfs, x=r'# Optimization steps', y='Capacity', hue='Method', style='Method', markers='^') 
plt.title(f'Quantum Switch with {nnodes} leaf nodes')
plt.gcf().set_size_inches(15,7)
g.grid(which='major', color='w', linewidth=1.0)
g.grid(which='minor', color='w', linewidth=0.5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
ax.legend(fancybox=True, framealpha=0.5)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

