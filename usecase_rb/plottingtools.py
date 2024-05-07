"""
Plotting tools
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import glob
import pickle

import config

plt.style.use("seaborn-v0_8-paper")
font = 14
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'arial',
    'font.size': font,
    'axes.labelsize': font,  
    'xtick.labelsize': font,  
    'ytick.labelsize': font, 
    'legend.fontsize': font,
    'legend.title_fontsize': font,
    'axes.titlesize': font
})
import warnings
warnings.filterwarnings("ignore")

def read_pkl_surrogate_benchmarking(folder):
    surs = []
    for name in glob.glob(f'{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))
    
    errors = []
    acquisition = {}
    for i,sur in enumerate(surs):
        errors.append(pd.DataFrame.from_records(sur.model_scores))
        errors[i]['Trial'] = i
        errors[i] = errors[i].reset_index(names='Iteration')
        acquisition[f'Trial {i}'] = pd.Series(sur.acquisition_time)/np.array(sur.optimize_time)[1:]

    df_errors = pd.concat(errors).melt(id_vars=['Trial', 'Iteration'], value_name='Mean Absolute Error', var_name='ML Model')
    fig, ax = plt.subplots()
    sns.pointplot(data=df_errors, x='Iteration', y='Mean Absolute Error', hue='ML Model', errorbar='se')
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df_acquisition = pd.DataFrame.from_dict(acquisition).reset_index(names='Iteration')
    df_acquisition = df_acquisition.melt(id_vars='Iteration', value_name='Execution Time [s]', var_name='Trial')
    fig, ax = plt.subplots()
    sns.pointplot(data=df_acquisition, x='Iteration', y='Execution Time [s]', hue='Trial', errorbar='se')
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return df_errors, df_acquisition

def read_pkl_surrogate_timeprofiling(folder):
    surs = []
    for name in glob.glob(f'{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))
    
    times = {'Simulation':[], 'Build':[], 'Acquisition':[], '# Iterations': []}
    for sur in surs:
        times['Simulation'].append(np.sum(sur.sim_time))
        times['Build'].append(np.sum(sur.build_time))
        times['Acquisition'].append(np.sum(sur.acquisition_time))
        times['# Iterations'].append(len(sur.sim_time))

    times = pd.DataFrame.from_dict(times)
    times['Total'] = times[['Simulation', 'Build', 'Acquisition']].sum(axis=1)
    times_relative = times.drop('# Iterations', axis=1).div(times['Total'], axis=0)
    return times, times_relative

def read_pkl_surrogate(folder):
    surs = []
    for name in glob.glob(f'{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))

    vals = surs[0].vals
    Xs = []
    for i, sur in enumerate(surs):
        sur.X_df['Trial'] = i
        Xs.append(sur.X_df)

    Xs = pd.concat(Xs, axis=0, ignore_index=True)
    ys = pd.concat([pd.DataFrame(sur.y, columns = pd.Series(range(config.nnodes)).astype('str')) 
                    for sur in surs], axis=0, ignore_index=True)
    ys['Utility'] = ys.sum(axis=1)
    ys_std = pd.concat([pd.DataFrame(sur.y_std, columns = pd.Series(range(config.nnodes)).astype('str')).add_suffix('_std') 
                        for sur in surs], axis=0, ignore_index=True)

    ys_std['Utility Std'] = ys_std.apply(np.square).sum(axis=1).apply(np.sqrt)
    df = Xs.join(ys)
    df = df.join(ys_std)
    df['Method'] = 'Surrogate'
    return df.reset_index(), vals

def read_pkl_meta(folder):
    metas = []
    for name in glob.glob(f'{folder}/AX_*.pkl'): 
        with open(name,'rb') as file: metas.append(pickle.load(file))
    dfs = []
    for i, meta in enumerate(metas):
        data = meta[0]
        data['Trial'] = i
        dfs.append(data)
    df = pd.concat(dfs, axis=0)
    df['Method'] = 'Meta'
    return df.reset_index()

def read_pkl_randomsearch(folder):
    gss = []
    for name in glob.glob(f'{folder}/RS_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for i, gs in enumerate(gss):
        gs[0]['Trial'] = i
        dfs.append(gs[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['Utility'].apply(np.nansum)
    df['Method'] = 'Random Search'
    return df.reset_index()

def read_pkl_sa(folder):
    gss = []
    for name in glob.glob(f'{folder}/SA_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for i, gs in enumerate(gss):
        gs[0]['Trial'] = i
        dfs.append(gs[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    df['Method'] = 'Simulated Annealing'
    return df.reset_index()

def to_dataframe(res):
    df = pd.DataFrame.from_records(res)
    df = df[2].transform({i: itemgetter(i) for i in range(config.nnodes)}) # get raw mean per node
    df = df.add_prefix('Node')
    df['Aggregated Completed Requests'] = df.sum(axis=1)
    return df

def plot_optimization_results(folder):
    target_columns = ['Trial', 'Utility', 'Method']
    df = pd.concat([read_pkl_surrogate(folder)[0][target_columns], read_pkl_meta(folder)[target_columns],
                    read_pkl_sa(folder)[target_columns], read_pkl_randomsearch(folder)[target_columns]], axis=0, ignore_index=True)

    grouped = df.groupby(['Method', 'Trial'], sort=False).max()
    sns.pointplot(data=grouped, x='Method', y='Utility', errorbar='se', linestyles='None', hue='Method')
    plt.xlabel('')
    plt.title(r'Maximum Utility ($K=10$ Trials per Method)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

def get_performance_distribution_per_method(folder):
    df_sur, _ = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_randomsearch(folder)
    columns = ['Trial', 'Method', 'Utility']
    df = pd.concat([df_sur[columns], df_meta[columns], df_sa[columns], df_gs[columns]])
    max_per_trial = df.groupby(['Method', 'Trial'])['Utility'].max()
    mean_std = max_per_trial.groupby(level='Method').agg(['min', 'max', 'mean', 'std'])
    mean_std['rel_std'] = mean_std['std']/mean_std['mean']
    return mean_std

def get_best_parameters(folder):
    df_sur, vals = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_randomsearch(folder)

    xs = dict()
    for df in [df_sur, df_meta, df_sa, df_gs]:
        xmethod = df.iloc[df['Utility'].idxmax()][df.columns.str.contains('mem_size|Method')] 
        xs[xmethod['Method']] = xmethod.drop('Method')

    # even distribution
    even = dict()
    for i in range(config.nnodes):
        even[f'mem_size_node_{i}'] = 50
    xs['Even'] = even
    
    # weighted distribution according to Wu X. et al., 2021
    xs['Wu et. al, 2021'] = {'mem_size_node_0': 25, 'mem_size_node_1': 91, 'mem_size_node_2': 67,
               'mem_size_node_3': 24, 'mem_size_node_4': 67, 'mem_size_node_5': 24, 
               'mem_size_node_6': 103, 'mem_size_node_7': 25, 'mem_size_node_8':24}

    x_df = pd.DataFrame.from_records(xs).T
    x_df['Total Number of Allocated Memories'] = x_df.sum(axis=1).astype(int)
    return x_df, xs, vals

def plot_from_exhaustive(folder):
    x_df, _, _ = get_best_parameters(folder)
    method_names = ['Surrogate', 'Meta', 'Simulated Annealing', 'Random Search', 'Even', 'Wu et. al, 2021']
    dfs = [None]*6
    for name in glob.glob(f'{folder}/Results_*.csv'):
        df = pd.read_csv(name)
        method = df.Method[0]
        index = method_names.index(method)
        dfs[index] = df

    df = pd.concat(dfs, axis=0)
    df = df.drop('Unnamed: 0' , axis=1)
    df = df.melt(id_vars=['Method', 'Aggregated Completed Requests'], var_name='User', value_name='Number of Completed Requests')
    df['User'] = df['User'].apply(lambda x: str.replace(x, 'Node', ''))
    df = df.merge(x_df, left_on='Method', right_index=True, how='left')
    markers = ['o', '^', 'v', 's', 'd', 'P']
    fig, axs = plt.subplots(1,1, figsize=(5,3))
    sns.pointplot(data= df, x='Total Number of Allocated Memories', y='Aggregated Completed Requests', hue='Method', ax=axs, errorbar='se', markers=markers, legend=True, linestyles=['']*6, native_scale=True)
    axs.grid()
    plt.title('Aggregated Number of Completed Requests')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    folder = '../../surdata/rb_budget'

    #plot results of optimization (Utility)
    get_performance_distribution_per_method(folder)

    # # plot from exhaustive run
    # plot_from_exhaustive(folder)

    # # plot time profiling
    # time_profile, rel_time_profile = read_pkl_surrogate_timeprofiling(folder)
    # print(rel_time_profile.mean(axis=0))

    # df = get_performance_distribution_per_method(folder)
    # print(df)

    # error, acquisition = read_pkl_surrogate_benchmarking(folder)
    # print(error, acquisition)