"""
Script to extract best found parameter values of different optimization methods and run for large simulation-sample size N>>1.
"""

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


def read_in_surrogate(folder):
    surs = []
    for name in glob.glob(f'{folder}/SU_*.pkl'): 
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

def read_in_meta(folder):
    metas = []
    for name in glob.glob(f'{folder}/AX_*.pkl'): 
        with open(name,'rb') as file: metas.append(pickle.load(file))
    dfs = []
    for meta in metas:
        dfs.append(meta[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['evaluate']
    return df

def read_in_gridsearch(folder):
    gss = []
    for name in glob.glob(f'{folder}/GS_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for gs in gss:
        dfs.append(gs[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    return df

def read_in_sa(folder):
    gss = []
    for name in glob.glob(f'{folder}/SA_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for gs in gss:
        dfs.append(gs)
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    print(df)
    return df


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

if __name__ == '__main__':

    folder = '../../surdata/qswitch'
    result_folder = '../../surdata/qswitch/Results_qswitch_5users_T30min.csv'

    df_sur, vals = read_in_surrogate(folder)
    df_meta = read_in_meta(folder)
    df_sa = read_in_sa(folder)
    df_gs = read_in_gridsearch(folder)

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
                                'Simulated Annealing', 'Random Gridsearch']):
            sim = Simulation(simwrapper, simulation_qswitch)
            res = sim.run_exhaustive(x=x, vals=vals, N=nprocs, seed=seed_count)
            df = to_dataframe(res)
            df['Method'] = method
            dfs.append(df)
        seed_count += 1
        if len(dfs)*nprocs > 4000:
            break
    
    # df_exhaustive = pd.concat(dfs, axis=0).dropna()
    # df_exhaustive = df_exhaustive.round(6)
    # df_exhaustive.to_csv(result_folder) 