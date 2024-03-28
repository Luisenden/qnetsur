import pickle
from operator import itemgetter
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
from src.utils import *
from config import *
from simulation import *

import warnings
warnings.filterwarnings("ignore")


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
    df['Method'] = 'Surrogate Optimization'
    return df, vals

def read_pkl_meta(folder):
    metas = []
    for name in glob.glob(f'{folder}/AX_*.pkl'): 
        with open(name,'rb') as file: metas.append(pickle.load(file))
    dfs = []
    for i, meta in enumerate(metas):
        data = meta[0].get_trials_data_frame()
        data['Trial'] = i
        dfs.append(data)
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['mean']
    df['Method'] = 'Meta'
    return df

def read_pkl_gridsearch(folder):
    gss = []
    for name in glob.glob(f'{folder}/GS_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for i, gs in enumerate(gss):
        gs[0]['Trial'] = i
        dfs.append(gs[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    df['Method'] = 'Random Gridsearch'
    return df

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
    return df


def to_dataframe(res):
    df = pd.DataFrame.from_records(res)
    df_raw = df[2].transform({i: itemgetter(i) for i in range(9)}) # get mean per node
    df_raw = df_raw.add_prefix('Node')
    df_raw['Aggregated Completed Requests'] = df_raw.sum(axis=1)
    return df_raw

def get_best_x(df):
    return df.iloc[df['Utility'].idxmax()][df.columns.str.contains('mem_size')]


if __name__ == '__main__':

    folder = '../../surdata/rb'
    result_folder = '../../surdata/rb/Results_starlight_compare.csv'

    df_sur, vals = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_gridsearch(folder)

    xs = []
    for df in [df_sur, df_meta, df_sa, df_gs]:
        x = get_best_x(df)  
        xs.append(x)  

    # even distribution
    even = dict()
    for i in range(9):
        even[f'mem_size_node_{i}'] = 50
    xs.append(even)
    # weighted distribution according to Wu X. et al., 2021
    xs.append({'mem_size_node_0': 25, 'mem_size_node_1': 91, 'mem_size_node_2': 67,
               'mem_size_node_3': 24, 'mem_size_node_4': 67, 'mem_size_node_5': 24, 
               'mem_size_node_6': 103, 'mem_size_node_7': 25, 'mem_size_node_8':24})

    vals['N'] = 1
    nprocs = mp.cpu_count()

    dfs = []
    seed_count = 1
    while True:
        for x, method in zip(xs[-2:], ['Surrogate', 'Meta',
                                'Simulated Annealing', 'Random Gridsearch', 'Budget 450'][-2:]):
            sim = Simulation(simwrapper, simulation_rb)
            res = sim.run_exhaustive(x=x, vals=vals, N=nprocs, seed=seed_count)

            df = to_dataframe(res)
            print(df)
            df['Method'] = method
            dfs.append(df)
        seed_count += 1
        if len(dfs)*nprocs >= 2e3:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    df_exhaustive.to_csv(result_folder) 