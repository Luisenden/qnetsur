import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from config import Config
from src.utils import Simulation

from optimizingcd import main_cd as simulation
 

if __name__ == '__main__':

    # load configuration
    conf = Config(initial_model_size=5)
    limit = conf.args.time
    evals = [] # storage for results
    times_tracked = []
    time_tracker = 0
    delta = 0

    sim = Simulation(conf.simobjective, simulation.simulation_cd, rng=conf.rng, values=conf.vals, variables=conf.vars)

    while time_tracker + delta < limit * 3600:
        start = time.time()
        x = sim.get_random_x(1)
        print(x)
        eval = sim.run_sim(x)
        evalset = x.copy()
        evalset['objective'], evalset['std'], evalset['raw'] = eval
        evals.append(evalset)

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    randomsearch = pd.DataFrame.from_records(evals)
    limit_kind = 'hours' if isinstance(limit, float) else 'cycles'
    with open(conf.args.folder+f'RS_{conf.name}_{limit}{limit_kind}_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([randomsearch,time_tracker,conf.vals], file)