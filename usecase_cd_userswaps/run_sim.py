import pandas as pd
import multiprocessing as mp
import time
import sys
import numpy as np

from config import Config
sys.path.append('../')
from qnetsur.utils import Simulation

def to_dataframe(res, users):
    df = pd.DataFrame.from_records(res)
    df = df[0].apply(pd.Series)
    df.columns = users
    df = df.add_prefix('User')
    df['Aggregated Number of Virtual Neighbors'] = df.sum(axis=1)
    return df

if __name__ == '__main__':

    conf = Config()
    nprocs = mp.cpu_count()
    print('Number of processes: ', nprocs)

    df=pd.read_csv('/Users/localadmin/Documents/git/surdata/cdtest/SU_randtree-100_1.0hours_SEED42_08-10-2024_13:11:36.csv', index_col=0)

    x_vals = np.ones(len(conf.A))
    x_vals[conf.user_indices] = df.iloc[df['objective'].argmax()][:39]
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

        df = to_dataframe(res, users=conf.user_indices)
        dfs.append(df)

        seed_count += 1
        if len(dfs)*nprocs >= 10:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    result_folder = conf.folder+f'sanitychecks_nodes1_user0.csv'
    df_exhaustive.to_csv(result_folder) 
