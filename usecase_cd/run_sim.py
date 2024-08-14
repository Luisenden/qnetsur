import pandas as pd
import multiprocessing as mp
import time
import sys
import numpy as np

from config import Config
sys.path.append('../')
from qnetsur.utils import Simulation

def to_dataframe(res):
    df = pd.DataFrame.from_records(res)
    df = df[0].apply(pd.Series)
    df = df.add_prefix('User')
    df['Aggregated Number of Virtual Neighbors'] = df.sum(axis=1)
    return df

if __name__ == '__main__':

    conf = Config()
    nprocs = mp.cpu_count()
    print('Number of processes: ', nprocs)
    
    x_vals = np.zeros(len(conf.A))

    x = {}
    for i,val in enumerate(x_vals):
        x[f'q_swap{i}'] = val

    print(x)
    dfs = []
    seed_count = 1
    while True:
        sim = Simulation(conf.simobjective, conf.sim, conf.rng, values=conf.vals, variables=conf.vars)
        start = time.time()
        res = sim.run_exhaustive(x=x, N=nprocs, seed=seed_count)
        print(time.time()-start)

        df = to_dataframe(res)
        dfs.append(df)

        seed_count += 1
        if len(dfs)*nprocs >= 10:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    result_folder = conf.folder+f'{conf.name}_sanitycheck_user0.csv'
    df_exhaustive.to_csv(result_folder) 
