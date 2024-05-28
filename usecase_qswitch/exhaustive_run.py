"""
Plotting tools
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from operator import itemgetter
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
from src.utils import *
from config import *

import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-paper")
font = 18
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
    ys = pd.concat([pd.DataFrame(sur.y, columns = pd.Series(range(vals['nnodes']-1)).astype('str')) 
                    for sur in surs], axis=0, ignore_index=True)
    ys['Utility'] = ys.sum(axis=1)
    ys_std = pd.concat([pd.DataFrame(sur.y_std, columns = pd.Series(range(vals['nnodes']-1)).astype('str')).add_suffix('_std') 
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
    df['Utility'] = df['evaluate']
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
    df['Utility'] = df['objective'].apply(np.nansum)
    df['Method'] = 'Random Search'
    return df.reset_index()

def read_pkl_sa(folder):
    gss = []
    for name in glob.glob(f'{folder}/SA_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for i, gs in enumerate(gss):
        gs['Trial'] = i
        dfs.append(gs)
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    df['Method'] = 'Simulated Annealing'
    return df.reset_index()


def to_dataframe(res):
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
    df['Fidelity Mean'] = df.groupby('index')['Fidelity'].transform(np.mean)
    df['Fidelity Std'] = df.groupby('index')['Fidelity'].transform(np.std)
    df['User'] = df.groupby('index').cumcount()
    return df

def get_best_x(df):
    return df.iloc[df['Utility'].idxmax()][df.columns.str.contains('bright_state')]

def plot_surrogate_linklevelfidels(folder, trial=0):
    df, _ = read_pkl_surrogate(folder)
    
    cols_i =df.columns.str.contains('bright_state')
    cols_names = ['Server']+[f'User {i}' for i in range(4,-1,-1)]
    df[cols_names] = df[df.columns[cols_i]].applymap(lambda x: 1-x)

    df = pd.melt(df, value_vars=cols_names, var_name='Node', value_name='Link-level Fidelity', id_vars=['Iteration', 'Trial'])
    
    fig, ax = plt.subplots(figsize = (10,4))
    sns.lineplot(data=df[df.Trial==trial], x='Iteration', y='Link-level Fidelity', hue='Node', style='Node')
    sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))
    plt.title('Link-level Fidelity per Node')
    plt.ylabel('Fidelity')
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    folder = '../../surdata/qswitch'
    result_folder = '../../surdata/qswitch/Results_qswitch_5users_T30min.csv'

    df_sur, vals = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_randomsearch(folder)

    xs = []
    for df in [df_sur, df_meta, df_sa, df_gs]:
        x = get_best_x(df)  
        xs.append(x)  

    vals['N'] = 1
    nprocs = mp.cpu_count()

    dfs = []
    seed_count = 1
    while True:
        for x,method in zip(xs, ['Surrogate', 'Meta',
                                'Simulated Annealing', 'Random Search']):
            sim = Simulation(simwrapper, simulation_qswitch)
            res = sim.run_exhaustive(x=x, vals=vals, N=nprocs, seed=seed_count)
            df = to_dataframe(res)
            df['Method'] = method
            dfs.append(df)
        seed_count += 1
        if len(dfs)*nprocs > 4000:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    df_exhaustive.to_csv(result_folder) 