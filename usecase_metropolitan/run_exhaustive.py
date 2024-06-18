# used parameter settings and program function of https://github.com/sequence-toolbox/Chicago-metropolitan-quantum-network/blob/master/sec5.4-two-memory-distribution-policies/run.py

import time
import pandas as pd
import torch.multiprocessing as mp
import sys
sys.path.append('../')
sys.path.append('../src')
from qnetsur.utils import Simulation
from simulation import simulation_rb
from usecase_metropolitan.plottingtools import get_best_parameters, to_dataframe
from config import Config

import warnings
warnings.filterwarnings("ignore")

# parser.add_argument("--method", type=str, default='Surrogate', 
#                     help="Choose one of the following methods: 'Surrogate', 'Meta', 'Simulated Annealing', 'Random Search', 'Even', 'Wu et. al, 2021'")
# args, _ = parser.parse_known_args()
# METHOD = args.method


if __name__ == '__main__':

    folder = '../../surdata/rb_budget_25h'

    policies = pd.read_csv(folder+'/Best_found_solutions.csv', index_col=0)

    conf = Config()
    nprocs = mp.cpu_count()

    x = policies.loc["Random Search"].drop(['Total Number of Allocated Memories'])
    conf.vals['N'] = 1

    dfs = []
    seed_count = 1
    while True:
        sim = Simulation(conf.simobjective,  conf.sim, conf.rng, values=conf.vals, variables=conf.vars)
        start = time.time()
        res = sim.run_exhaustive(x=x, N=nprocs, seed=seed_count)
        print(time.time()-start)

        df = to_dataframe(res)
        df['Method'] = 'test'

        dfs.append(df)

        seed_count += 1
        if len(dfs)*nprocs >= 10:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    print(df_exhaustive)


    # result_folder = f'../../surdata/rb_budget/Results_starlight_compare{METHOD}.csv'
    # df_exhaustive.to_csv(result_folder) 