import pickle
from operator import itemgetter
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
from src.utils import *
from config import *

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


def read_in_surrogate(folder):
    surs = []
    for name in glob.glob(f'../../surdata/{folder}/SU_*.pkl'): #Sur_qswitch_nleafnodes{nnodes}_{time}*'):
        with open(name,'rb') as file: surs.append(pickle.load(file))

    vals = surs[0].vals

    Xs = []
    for i, sur in enumerate(surs):
        sur.X_df['Trial'] = i
        Xs.append(sur.X_df)

    Xs = pd.concat(Xs, axis=0, ignore_index=True)
    ys = pd.concat([pd.DataFrame(sur.y, columns = pd.Series(range(sur.vals['nnodes']-1)).astype('str')) 
                    for sur in surs], axis=0, ignore_index=True)
    ys['Utility'] = ys.sum(axis=1)
    ys_std = pd.concat([pd.DataFrame(sur.y_std, columns = pd.Series(range(sur.vals['nnodes']-1)).astype('str')).add_suffix('_std') 
                        for sur in surs], axis=0, ignore_index=True)

    ys_std['Utility Std'] = ys_std.apply(np.square).sum(axis=1).apply(np.sqrt)
    df = Xs.join(ys)
    df = df.join(ys_std)
    return df, vals

def transform_result(res):
    df = pd.DataFrame.from_records(res)
    df_raw = df[2].transform({i: itemgetter(i) for i in range(4)}).drop([1, 3], axis=1)
    df_raw = df_raw.applymap(lambda x: x[1:]) # ignore entry for server node
    df_raw.columns = ['Rate [Hz]', 'Fidelity']
    
    df = df.drop([1, 2], axis=1)
    df.columns = ['Utility']
    df = df.explode('Utility').reset_index()
    df_raw = df_raw.apply(lambda x: x.explode(ignore_index=True), axis=0)
    df = df.join(df_raw)
    df['Aggregated Utility'] = df.groupby('index')['Utility'].transform(np.sum)
    df['Aggregated Rate [Hz]'] = df.groupby('index')['Rate [Hz]'].transform(np.sum)
    df['Mean Fidelity'] = df.groupby('index')['Fidelity'].transform(np.mean)
    df['User'] = df.groupby('index').cumcount()
    return df

def plotting(df):
    alpha = 0.3
    size = 2
    fig, axs = plt.subplots(2,2)
    sns.stripplot(data= df.drop('index', axis=1), x='User', y='Utility', hue='Method', ax=axs.flat[0], legend=False, size=size, dodge=True)
    sns.boxplot(data= df.drop('index', axis=1), x='User', y='Utility', hue='Method', 
                boxprops=dict(alpha=alpha), flierprops=dict(alpha=alpha), capprops=dict(alpha=alpha), medianprops=dict(alpha=alpha), whiskerprops=dict(alpha=alpha), ax=axs.flat[0])
    
    sns.stripplot(data= df.drop_duplicates(), x='Method', y='Aggregated Utility', ax=axs.flat[1], palette=sns.color_palette(), legend=False, size=size, dodge=True)
    sns.boxplot(data= df.drop_duplicates(), x='Method', y='Aggregated Utility', 
            boxprops=dict(alpha=alpha), flierprops=dict(alpha=alpha), capprops=dict(alpha=alpha), medianprops=dict(alpha=alpha), whiskerprops=dict(alpha=alpha), ax=axs.flat[1])
    axs.flat[1].set_xticks([])
    axs.flat[1].set_xlabel('Aggregated Utility')
    axs.flat[1].set_ylabel('')

    sns.stripplot(data= df.drop('index', axis=1), x='User', y='Rate [Hz]', hue='Method', ax=axs.flat[2], legend=False, size=size, dodge=True)
    sns.boxplot(data= df.drop('index', axis=1), x='User', y='Rate [Hz]', hue='Method', 
                boxprops=dict(alpha=alpha), flierprops=dict(alpha=alpha), capprops=dict(alpha=alpha), medianprops=dict(alpha=alpha), whiskerprops=dict(alpha=alpha), ax=axs.flat[2])
    
    sns.stripplot(data= df.drop_duplicates(), x='Method', y='Aggregated Rate [Hz]', ax=axs.flat[3], palette=sns.color_palette(), legend=False, size=size, dodge=True)
    sns.boxplot(data= df.drop_duplicates(), x='Method', y='Aggregated Rate [Hz]', 
            boxprops=dict(alpha=alpha), flierprops=dict(alpha=alpha), capprops=dict(alpha=alpha), medianprops=dict(alpha=alpha), whiskerprops=dict(alpha=alpha), ax=axs.flat[3])
    axs.flat[3].set_xticks([])
    axs.flat[3].set_xlabel('Aggregated Rate [Hz]')
    axs.flat[3].set_ylabel('')
    
    sns.move_legend(axs.flat[2], loc='upper right', bbox_to_anchor=(2, 1))
    sns.move_legend(axs.flat[0], loc='upper right', bbox_to_anchor=(2, 1))
    for ax in axs.flat:
        ax.grid(alpha=alpha)
    plt.show()

if __name__ == '__main__':

    folder = 'qswitch'
    df, vals = read_in_surrogate(folder=folder)
    vals['N'] = 1
    x = df.iloc[df['Utility'].idxmax()][df.columns.str.contains('bright_state')]
    sim = Simulation(simwrapper, simulation_qswitch)
    res = sim.run_exhaustive(x=x, vals=vals, N=3)

    x.update({'bright_state_0': 0.01})
    sim = Simulation(simwrapper, simulation_qswitch)
    res1 = sim.run_exhaustive(x=x, vals=vals, N=3)

    df = transform_result(res)
    df1 = transform_result(res1)

    df['Method'] = 'Surrogate'
    df1['Method'] = 'Test'

    df = pd.concat([df, df1], axis=0)
    print(df)

    plotting(df)
    
    





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


# df_gs = pd.concat([gs[0] for gs in gss]).reset_index()
# paramcolumns = df_gs.columns[df_gs.columns.str.contains('mem_size')]
# df_gs['sum'] = df_gs['objective'].apply(lambda x: np.sum(x))
# df_gs['Objective Mean'] = df_gs['sum'] 
# df_gs['Method'] = 'Random Grid Search'

# df_sa = pd.concat(sas).reset_index()
# paramcolumns = df_sa.columns[df_sa.columns.str.contains('mem_size')]
# df_sa['sum'] = df_sa['objective']
# df_sa['Objective Mean'] = df_sa['sum'] 
# df_sa['Method'] = 'Simulated annealing'

# dfs = [df_sur, df_ax, df_gs, df_sa]#
# dfs_obj = []
# for df in dfs:
#     dfs_obj.append(df[['index', 'Objective Mean', 'Method']])
# dfs = pd.concat(dfs_obj).reset_index()
# dfs['\# Optimization steps'] = dfs['index']
# print(dfs)
# fig, ax = plt.subplots()
# g = sns.lineplot(data = dfs, x='\# Optimization steps', y='Objective Mean', hue='Method', style='Method', markers='^', markersize=5) 
# plt.title(f'Quantum Network with {nnodes}')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)
# ax.legend(fancybox=True, framealpha=0.5)
# ax.legend(loc='upper right')
# plt.ylabel('Utility')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# plt.grid()
# plt.tight_layout()
# plt.show()
# plt.savefig('test.pdf', dpi=100)

# print(sur.X_df)
# df_sur_X_plot = sur.X_df.melt(id_vars='Iteration', value_name='\# Memory Qubits', var_name='Node')
# sns.lineplot(df_sur_X_plot, x='Iteration', y='\# Memory Qubits', hue='Node')
# plt.legend([],[], frameon=False)
# plt.show()
