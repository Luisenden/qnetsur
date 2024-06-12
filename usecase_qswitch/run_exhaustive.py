import glob
import argparse
import pandas as pd
import torch.multiprocessing as mp
import time
import sys

from config import Config
sys.path.append('../')
from src.utils import Simulation

def get_solution(folder):
    dfs = []
    for i,name in enumerate(glob.glob(folder + f'/{mapping[args.method]}_*.pkl')): 
        with open(name,'rb') as file: dfs.append(pd.read_pickle(file))
        dfs[i]['Trial'] = i
    df = pd.concat(dfs, axis=0).reset_index()
    cols = df.columns[df.columns.astype('str').str.contains('bright_state')]
    x = df.iloc[df['objective'].idxmax()][cols]
    print(df['objective'].max())
    return x

def get_values(folder):
    filename = glob.glob(folder+'*_INPUT_VALUES.pkl')[0]
    vals = pd.read_pickle(filename).to_dict()[0]
    vals['N'] = 1
    return vals

def to_dataframe(res):
    df = pd.DataFrame.from_records(res)
    df = df[0].apply(pd.Series)
    df = df.add_prefix('User')
    df['Utility'] = df.sum(axis=1)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="set method")
    parser.add_argument("--method", type=str, default='Surrogate', 
                        help="Choose one of the following methods: 'Surrogate', 'Meta', 'Simulated Annealing', 'Random Search'")
    parser.add_argument("--lim", type=str, default='30min', 
                    help="Folder name.")
    args, _ = parser.parse_known_args()
    mapping = {'Surrogate':'SU', 'Meta':'AX', 'Simulated Annealing':'SA', 'Random Search':'RS'}

    folder = f'../../surdata/qswitch_{args.lim}/'
    vals = get_values(folder)
    x = get_solution(folder)

    conf = Config()
    nprocs = mp.cpu_count()
    print('Number of processes: ', nprocs)

    dfs = []
    seed_count = 1
    while True:
        sim = Simulation(conf.simobjective, conf.sim, conf.rng, values=vals, variables=None)
        start = time.time()
        res = sim.run_exhaustive(x=x, N=nprocs, seed=seed_count)
        print(time.time()-start)

        df = to_dataframe(res)
        df['Method'] = args.method
        dfs.append(df)

        seed_count += 1
        if len(dfs)*nprocs >= 10:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    result_folder = folder+f'Results_cd_compare_{args.method}.csv'
    df_exhaustive.to_csv(result_folder) 