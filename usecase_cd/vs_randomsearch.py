import time
import numpy as np
import pandas as pd
from datetime import datetime

from config import Config
from qnetsur.utils import Simulation
 

if __name__ == '__main__':

    # load configuration
    conf = Config()
    limit = conf.args.time
    path = conf.args.folder+f'RS_{conf.name}_{limit}hours_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.csv'
    evals = [] # storage for results
    times_tracked = []
    time_tracker = 0
    delta = 0

    sim = Simulation(conf.simobjective, conf.sim, rng=conf.rng, values=conf.vals, variables=conf.vars)

    while time_tracker + delta < limit * 3600:
        start = time.time()
        x = sim.get_random_x(1)
        print(x)
        eval = sim.run_sim(x)
        evalset = x.copy()
        evalset['objective'] = np.sum(eval[0])
        evals.append(evalset)

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    randomsearch = pd.DataFrame.from_records(evals)
    randomsearch.to_csv(path)