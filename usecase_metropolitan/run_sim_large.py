"""Script to execute simulation n=1000 times using different policies."""

import glob
import argparse
import pandas as pd
import multiprocessing as mp
import time
import sys
from plotting import get_policies

from config import Config
sys.path.append('../')
from qnetsur.utils import Simulation

import warnings
warnings.filterwarnings("ignore")

def get_solution(folder):
    dfs = []
    for i,name in enumerate(glob.glob(folder + f'{mapping[args.method]}*.csv')): 
        with open(name,'rb') as file: dfs.append(pd.read_csv(file, index_col=0))
        dfs[i]['Trial'] = i
    df = pd.concat(dfs, axis=0).reset_index()
    cols = df.columns[df.columns.astype('str').str.contains('mem_size')]
    df = df.iloc[df['objective'].idxmax()][cols]
    return df

def to_dataframe(res):
    df = pd.DataFrame.from_records(res)
    df = df[0].apply(pd.Series)
    df = df.add_prefix('User')
    df['Aggregated Completed Requests'] = df.sum(axis=1)
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="set method")
    parser.add_argument("--method", type=str, default='Surrogate', 
                        help="Choose one of the following methods: 'Surrogate', 'Meta', 'Simulated Annealing', 'Random Search', 'Wu et. al, 2021', 'Even' ")
    parser.add_argument("--hour", type=int, default=1, 
                    help="Choose folder.")
    args, _ = parser.parse_known_args()
    mapping = {'Surrogate':'SU', 'Meta':'AX', 'Simulated Annealing':'SA', 'Random Search':'RS'}

    folder = '../../surdata/QENTSUR-DATA/metropolitan_network/rb_25h/'

    conf = Config()
    conf.vals['N'] = 1

    nprocs = mp.cpu_count()

    # x = get_solution(folder)
    # print(x)

    x = pd.read_csv(folder+'Best_found_solutions.csv', index_col=0 )
    x = x.drop('Total Number of Allocated Memories', axis=1)
    x = x.loc[args.method]

    dfs = []
    seed_count = 1
    while True:
        sim = Simulation(conf.simobjective,  conf.sim, conf.rng, values=conf.vals, variables=conf.vars)
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
    print(df_exhaustive)

    result_folder = folder+f'Results_starlight_compare{args.method}.csv'
    df_exhaustive.to_csv(result_folder) 