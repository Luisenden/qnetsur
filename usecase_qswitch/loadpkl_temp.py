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
plt.style.use("seaborn-paper")
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


folder = 'qswitch'



for name in glob.glob(f'../../surdata/{folder}/SU_*.pkl'): #Sur_qswitch_nleafnodes{nnodes}_{time}*'):
    with open(name,'rb') as file: 
        sur = pickle.load(file)

raw = pd.DataFrame(sur.y_raw)
raw_sum = raw.applymap(lambda x: np.mean(x[1:]))


# df_raw_plot = sur.X_df.join(raw_sum).groupby('Iteration').mean()
# fig, ax1 = plt.subplots()
# color = 'green'
# ax1.set_xlabel('Optimization Step')
# ax1.set_ylabel('Rate [Hz]', color=color)
# ax1.errorbar(x=[i for i in range(28)], y=df_raw_plot[0], yerr=df_raw_plot[1], color=color, fmt='^')
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'blue'
# ax2.set_ylabel('Fidelity', color=color)  # we already handled the x-label with ax1
# ax2.errorbar(x=[i for i in range(28)], y=df_raw_plot[2], yerr=df_raw_plot[3], color=color, fmt='^')
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylim(0.7,1)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()


# plt.errorbar(x=[i for i in range(280)], y=raw_sum[0], yerr=[raw_sum[1]])
# plt.errorbar(x=[i for i in range(280)], y=raw_sum[2], yerr=[raw_sum[3]])
# plt.show()

df_plot = sur.X_df.groupby('Iteration').mean().reset_index().drop('Iteration', axis=1)
df_plot.columns = ['Server'] + [f'User {i}' for i in range(1,6)]
sns.lineplot(1-df_plot)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('Fidelity')
plt.xlabel('Optimization Step')
plt.show()

# dfs_sur = []
# nnodes = surs[0].vals['nnodes']-1
# column_names = pd.Series(range(nnodes)).astype('str')
# for i,sur in enumerate(surs):
#     df_sur_y = pd.DataFrame(sur.y, columns=column_names)
#     df_sur_y['Utility'] = df_sur_y.sum(axis=1)
#     df_sur_y['Utility Std'] = np.sqrt(np.sum(np.square(sur.y_std), axis=1)) # std of sum (assuming independent trials): sqrt of sum of variances 
#     df_sur_y[f'Trial'] = i
#     grouped_iteration = sur.X_df.join(df_sur_y).groupby(['Iteration'], as_index=False).apply(lambda x: x.loc[x['Utility'].idxmin()])
#     dfs_sur.append(grouped_iteration)


# df_sur = pd.concat(dfs_sur).reset_index()
# df_sur['lower'] = df_sur['Utility'] - df_sur['Utility Std']
# df_sur['upper'] = df_sur['Utility'] + df_sur['Utility Std']
# df_sur['index'] = df_sur['Iteration']
# df_sur['Method'] = 'Surrogate Optimization'
# df_sur['Objective Mean'] = df_sur['Utility']
# print(df_sur)


# # df_ax = pd.concat([ax[0] for ax in axs]).reset_index()
# # df_ax['Objective Mean'] = df_ax['evaluate']
# # df_ax['Method'] = 'Meta Optimization'
# # print(df_ax.columns)

# df_plot = pd.concat([df_sur.loc[df_sur[df_sur.Trial == trial]['Utility'].idxmax()] for trial in range(10)], axis=1).T.reset_index(drop=True)
# df_plot = (1-df_plot[[df_plot.columns.contains('bright_state')]])
# df_plot.columns = ['Server Link'] + [f'User {i} Link' for i in range(1,config.NLEAF_NODES)]

# fig, axs = plt.subplots(figsize=(20, 10))
# sns.lineplot(data=df_plot, markers='^', markersize=10)
# #plt.fill_between(df_plot['Iteration'], df_plot['lower'], df_plot['upper'], alpha=0.2)
# # plt.title(f'Trial {trial}')
# plt.tight_layout(pad=2, rect=[1, 0, 1, 1])
# plt.ylabel('Fidelity')
# plt.xlabel('Trial')
# plt.show()

# df_plot = pd.concat([df_sur.loc[df_sur[df_sur.Trial == trial]['Utility'].idxmax()] for trial in range(10)], axis=1).T.reset_index(drop=True)
# print('HERE', df_plot)
# df_plot = df_plot[['Trial', 'Utility', 'Utility Std']]
# fig, axs = plt.subplots(figsize=(20, 10))
# plt.errorbar(df_plot['Trial'], df_plot['Utility'], yerr=df_plot['Utility Std'],  fmt='o', markersize=5, capsize=5)
# #plt.fill_between(df_plot['Iteration'], df_plot['lower'], df_plot['upper'], alpha=0.2)
# # plt.title(f'Trial {trial}')
# plt.tight_layout(pad=1, rect=[1, 0, 1, 1])
# plt.ylabel('Utility')
# plt.xlabel('Trial')
# plt.ylim([1,8])
# plt.show() 


# # df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
# # paramcolumns = df_gs.columns[df_gs.columns.str.contains('mem_size')]
# # df_gs['sum'] = df_gs['objective'].apply(lambda x: np.sum(x))
# # df_gs['Objective Mean'] = df_gs['sum'] 
# # df_gs['Method'] = 'Random Grid Search'

# # df_sa = pd.concat(sas).reset_index()
# # paramcolumns = df_sa.columns[df_sa.columns.str.contains('mem_size')]
# # df_sa['sum'] = df_sa['objective']
# # df_sa['Objective Mean'] = df_sa['sum'] 
# # df_sa['Method'] = 'Simulated annealing'

# # dfs = [df_sur, df_ax, df_gs, df_sa]#
# # dfs_obj = []
# # for df in dfs:
# #     dfs_obj.append(df[['index', 'Objective Mean', 'Method']])
# # dfs = pd.concat(dfs_obj).reset_index()
# # dfs['\# Optimization steps'] = dfs['index']
# # print(dfs)
# # fig, ax = plt.subplots()
# # g = sns.lineplot(data = dfs, x='\# Optimization steps', y='Objective Mean', hue='Method', style='Method', markers='^', markersize=5) 
# # plt.title(f'Quantum Network with {nnodes}')
# # handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles=handles, labels=labels)
# # ax.legend(fancybox=True, framealpha=0.5)
# # ax.legend(loc='upper right')
# # plt.ylabel('Utility')
# # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# # plt.grid()
# # plt.tight_layout()
# # plt.show()
# # plt.savefig('test.pdf', dpi=100)

# # print(sur.X_df)
# # df_sur_X_plot = sur.X_df.melt(id_vars='Iteration', value_name='\# Memory Qubits', var_name='Node')
# # sns.lineplot(df_sur_X_plot, x='Iteration', y='\# Memory Qubits', hue='Node')
# # plt.legend([],[], frameon=False)
# # plt.show()
