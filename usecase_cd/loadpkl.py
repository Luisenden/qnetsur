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

sns.set(font_scale=2)

def reduce_to_means_per_iteration(df, group_size):
    df['Iteration'] = [i for i in range(len(df)//group_size) for _ in range(group_size)]
    return pd.DataFrame(df.groupby('Iteration').mean().to_numpy(), columns=['mean']) 

axs = []
folder = 'cd_N10_21tree_0.05h'
for name in glob.glob(f'../../surdata/{folder}/AX*'):
    with open(name,'rb') as file: axs.append(pickle.load(file))

surs = []
for name in glob.glob(f'../../surdata/{folder}/SU*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

sas = []
for name in glob.glob(f'../../surdata/{folder}/SA_*'):
    with open(name,'rb') as file: sas.append(pickle.load(file))

gss = []
for name in glob.glob(f'../../surdata/{folder}/RS_*'):
    with open(name,'rb') as file: gss.append(pickle.load(file))


dfs_sur = []
for sur in surs:
    df_sur = pd.DataFrame(np.mean(sur.y, axis=1))
    df_sur = reduce_to_means_per_iteration(df_sur,10)
    dfs_sur.append(df_sur)

df_sur = pd.concat(dfs_sur).reset_index()
df_sur['Method'] = 'Surrogate'

df_sa = pd.concat(sas).reset_index()
df_sa['mean'] = df_sa['objective']
df_sa['Method'] = 'Simulated Annealing'

df_ax = pd.concat([ax[0].get_trials_data_frame() for ax in axs]).reset_index()
df_ax['Method'] = 'Meta'
df_ax['mean'] = df_ax['evaluate']

df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
df_gs['mean'] = df_gs['objective'].apply(lambda x: np.mean(x))
df_gs['Method'] = 'Random Grid Search'

dfs = [df_sur, df_ax, df_sa, df_gs]
dfs_obj = []
for df in dfs:
    dfs_obj.append(df[['index', 'mean', 'Method']])
dfs = pd.concat(dfs_obj)
dfs[r'# optimization steps'] = dfs['index']
dfs[r'# virtual neighbours'] = dfs['mean']

fig, ax = plt.subplots()
g = sns.lineplot(data = dfs, x=r'# optimization steps', y=r'# virtual neighbours', hue='Method', style='Method') # plot the Number of Neighbours for all methods
plt.title(f'(2,3)-tree topology [time limit: 0.1h]')
plt.gcf().set_size_inches(15,7)
g.grid(which='major', color='w', linewidth=1.0)
g.grid(which='minor', color='w', linewidth=0.5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
ax.legend(fancybox=True, framealpha=0.5)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()



# print(axs[0][1])
# print(np.mean([sum(sur.build_time) + sum(sur.optimize_time) + sum(sur.sim_time) for sur in surs]))
