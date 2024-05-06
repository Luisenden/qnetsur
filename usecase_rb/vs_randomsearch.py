from datetime import datetime
import numpy as np
import pandas as pd
import time
import pickle
from config import MAX_TIME, simwrapper, vals, vars, SEED
from src.utils import Simulation
from simulation import simulation_rb

if __name__ == '__main__':

    # user input:
    max_time= MAX_TIME * 3600 # in sec

    evals = [] # storage for results

    times_tracked = []
    time_tracker = 0
    delta = 0

    sim = Simulation(simwrapper, simulation_rb, vals=vals, vars=vars)

    while time_tracker + delta < max_time:
        start = time.time()

        x = sim.get_random_x(1)
        eval = sim.run_sim(x=x, vals=vals)
        
        evalset = x.copy()
        evalset['Utility'], evalset['std'], evalset['raw'] = eval
        evals.append(evalset)

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    randomsearch = pd.DataFrame.from_records(evals)
    with open(f'../../surdata/rb_budget/RS_starlight_{MAX_TIME:.1f}h_objective-budget_SEED{SEED}_'
              +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([randomsearch, times_tracked, vals], file)