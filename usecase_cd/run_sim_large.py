import glob
import argparse
import pandas as pd
import multiprocessing as mp
import time
import sys
import numpy as np

from config import Config
sys.path.append('../')
from qnetsur.utils import Simulation

def get_solution(folder):
    dfs = []
    for i,name in enumerate(glob.glob(folder + f'{mapping[args.method]}_randtree-100_{args.hour}.0hours_*.csv')): 
        dfs.append(pd.read_csv(name, index_col=0))
        dfs[i]['Trial'] = i
    df = pd.concat(dfs, axis=0).reset_index()
    cols = df.columns[df.columns.astype('str').str.contains('q_swap')]
    sol = df.iloc[df['objective'].idxmax()][cols]
    return sol

def to_dataframe(res):
    df = pd.DataFrame.from_records(res)
    df = df[0].apply(pd.Series)
    df = df.add_prefix('User')
    df['Aggregated Number of Virtual Neighbors'] = df.sum(axis=1)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="set method")
    parser.add_argument("--method", type=str, default='Surrogate', 
                        help="Choose one of the following methods: 'Surrogate', 'Meta', 'Simulated Annealing', 'Random Search'")
    parser.add_argument("--hour", type=int, default=1, 
                    help="Choose folder.")
    args, _ = parser.parse_known_args()
    mapping = {'Surrogate':'SU', 'Meta':'AX', 'Simulated Annealing':'SA', 'Random Search':'RS'}

    conf = Config()
    vals = conf.vals
    x = get_solution(conf.folder)

    #user_indices = np.where(vals['A'].sum(axis=1) == 1)
    #x_vals = np.random.random_sample(len(vals['A']))
    #x_vals[user_indices] = 0
    # x = {}
    # for i,val in enumerate(x_vals):
    #     x[f'q_swap{i}'] = val
    
    vals['N_samples'] = 1
    nprocs = mp.cpu_count()
    print('Number of processes: ', nprocs)

    dfs = []
    seed_count = 1
    while True:
        sim = Simulation(conf.simobjective, conf.sim, conf.rng, values=vals, variables=conf.vars)
        start = time.time()
        res = sim.run_exhaustive(x=x, N=nprocs, seed=seed_count)
        print(time.time()-start)

        df = to_dataframe(res)
        df['Method'] = args.method
        dfs.append(df)

        seed_count += 1
        if len(dfs)*nprocs >= 1000:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    result_file = conf.folder+f'Results_cd_compare_{args.method}_{args.hour}.csv'
    df_exhaustive.to_csv(result_file) 
