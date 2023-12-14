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

# axs = []
# for name in glob.glob('../../surdata/qswitch/tree3-4/Ax*'):
#     with open(name,'rb') as file: axs.append(pickle.load(file))

surs = []
for name in glob.glob('../../surdata/qswitch/Sur*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

# sas = []
# for name in glob.glob('../../surdata/CD/tree3-4/SA_*'):
#     with open(name,'rb') as file: sas.append(pickle.load(file))

# gss = []
# for name in glob.glob('../../surdata/CD/tree3-4/GS_*'):
#     with open(name,'rb') as file: gss.append(pickle.load(file))


# df_ax = pd.concat([ax[0].get_trials_data_frame() for ax in axs]).reset_index()
# df_ax['Method'] = 'Meta Optimization'

# df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
# df_gs['mean'] = df_gs['objective'].apply(lambda x: np.mean(x))
# df_gs['Method'] = 'Random Grid Search'

# print(df_ax)

dfs_sur = []
for sur in surs:
    df_sur_y = pd.DataFrame(np.mean(sur.y, axis=1), columns=['mean'])
    df_sur = sur.X_df.join(df_sur_y).groupby(['Iteration'], as_index=False).mean()
    print(df_sur)
    dfs_sur.append(df_sur)

df_sur = pd.concat(dfs_sur).reset_index()
df_sur['Method'] = 'Surrogate Optimization'

# df_sa = pd.concat(sas).reset_index()
# df_sa['mean'] = df_sa['objective']
# df_sa['Method'] = 'Simulated annealing'

dfs = [df_sur]#, df_ax, df_sa, df_gs]
dfs_obj = []
for df in dfs:
    dfs_obj.append(df[['index', 'mean', 'Method']])
dfs = pd.concat(dfs_obj)
dfs['Optimization step'] = dfs['index']
dfs['Number of virtual neighbours'] = dfs['mean']

g = sns.lineplot(data = dfs, x='Optimization step', y='Number of virtual neighbours', hue='Method', style='Method') # plot the Number of Neighbours for all methods
plt.title(f'Optimization Quantum Network: qswitch with a time limit of {1*60:.0f} min')
plt.gcf().set_size_inches(15,7)
g.grid(which='major', color='w', linewidth=1.0)
g.grid(which='minor', color='w', linewidth=0.5)
plt.show()
