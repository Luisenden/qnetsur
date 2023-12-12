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
plt.style.use('seaborn-v0_8')

def reduce_to_means_per_iteration(df, group_size):
    df['Iteration'] = [i for i in range(len(df)//group_size) for _ in range(group_size)]
    return pd.DataFrame(df.groupby('Iteration').mean().to_numpy(), columns=['mean']) 

axs = []
for name in glob.glob('../../surdata/RB/12h/Ax_*'):
    with open(name,'rb') as file: axs.append(pickle.load(file))

surs = []
for name in glob.glob('../../surdata/RB/12h/Sur_*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

sas = []
for name in glob.glob('../../surdata/RB/12h/SA_*'):
    with open(name,'rb') as file: sas.append(pickle.load(file))

gss = []
for name in glob.glob('../../surdata/RB/12h/GS_*'):
    with open(name,'rb') as file: gss.append(pickle.load(file))


mean_iterations = int(np.mean([len(ax[0].get_trials_data_frame()) for ax in axs]))
df_ax_list = [ax[0].get_trials_data_frame() for ax in axs]
df_ax = pd.concat(df_ax_list).reset_index()
df_ax['Method'] = 'Meta Optimization'

mean_iterations = int(np.mean([len(gs[0]) for gs in gss]))
df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
df_gs['mean'] = df_gs['objective'].apply(lambda x: np.mean(x))
df_gs['Method'] = 'Random Grid Search'

dfs_sur = []
for sur in surs:
    df_sur = pd.DataFrame(np.mean(sur.y, axis=1)+np.mean(sur.X_df, axis=1)/110)
    print('df_sur: ', df_sur)
    df_sur = reduce_to_means_per_iteration(df_sur,10)
    dfs_sur.append(df_sur)

df_sur = pd.concat(dfs_sur).reset_index()
df_sur['Method'] = 'Surrogate Optimization'

df_sa = pd.concat(sas).reset_index()
df_sa['mean'] = df_sa['objective']
df_sa['Method'] = 'Simulated annealing'

dfs = [df_sur, df_ax, df_sa, df_gs]
dfs_obj = []
for df in dfs:
    dfs_obj.append(df[['index', 'mean', 'Method']])
dfs = pd.concat(dfs_obj)
dfs['Optimization step'] = dfs['index']
dfs['Completed requests on avareage'] = dfs['mean']

# print(dfs)

g = sns.lineplot(data = dfs, x='Optimization step', y='Completed requests on avareage', hue='Method', style='Method', markers=True) # plot the Number of Neighbours for all methods
plt.title(f'Optimization Quantum Network')
plt.gcf().set_size_inches(15,7)
g.grid(which='major', color='w', linewidth=1.0)
g.grid(which='minor', color='w', linewidth=0.5)
plt.show()



# print(axs[0][1])
# print(np.mean([sum(sur.build_time) + sum(sur.optimize_time) + sum(sur.sim_time) for sur in surs]))