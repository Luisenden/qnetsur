import glob
import argparse
import pandas as pd
import multiprocessing as mp
import time
import sys

from config import Config
sys.path.append('../')
from qnetsur.utils import Simulation

def get_solution(folder):
    dfs = []
    for i,name in enumerate(glob.glob(folder + f'{mapping[args.method]}_randtree-100_{args.hour}.0hours_*.pkl')): 
        with open(name,'rb') as file: dfs.append(pd.read_pickle(file))
        dfs[i]['Trial'] = i
    df = pd.concat(dfs, axis=0).reset_index()
    cols = df.columns[df.columns.astype('str').str.contains('q_swap')]
    df = df.iloc[df['objective'].idxmax()][cols]
    return df

def get_values(folder):
    filename = glob.glob(folder+'*_INPUT_VALUES.pkl')[0]
    vals = pd.read_pickle(filename).to_dict()[0]
    users = vals.pop('user')
    vals['N_samples'] = 1
    return vals, users[0]

def to_dataframe(res, users):
    df = pd.DataFrame.from_records(res)
    df = df[0].apply(pd.Series)
    df.columns = users
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

    folder = f'../../surdata/cd_{args.hour}h/'
    vals, users = get_values(folder)
    x = get_solution(folder)

    conf = Config()
    nprocs = mp.cpu_count()
    print('Number of processes: ', nprocs)

    dfs = []
    seed_count = 1
    while True:
        sim = Simulation(conf.simobjective, conf.sim, conf.rng, values=vals, variables=conf.vars)
        start = time.time()
        res = sim.run_exhaustive(x=x, N=nprocs, seed=seed_count)
        print(time.time()-start)

        df = to_dataframe(res, users=users)
        df['Method'] = args.method
        dfs.append(df)

        seed_count += 1
        if len(dfs)*nprocs >= 1000:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    result_folder = folder+f'Results_cd_compare_{args.method}_{args.hour}.csv'
    df_exhaustive.to_csv(result_folder) 
