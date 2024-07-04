import time
import numpy as np
import pandas as pd
from datetime import datetime

from config import Config
from qnetsur.utils import Simulation
 

if __name__ == '__main__':

    # Load configuration
    conf = Config()
    limit = conf.args.time
    # Define the path for saving the results
    path = conf.args.folder+f'RS_{conf.name}_{limit}hours_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.csv'
    
    # Initialize storage for results and time tracking
    evals = []
    times_tracked = []
    time_tracker = 0
    delta = 0

    # Initialize the simulation with configuration settings
    sim = Simulation(conf.simobjective, conf.sim, rng=conf.rng, values=conf.vals, variables=conf.vars)

    # Run the random search loop until the time limit is reached
    while time_tracker + delta < limit * 3600:
        start = time.time()
        # Generate a random set of parameters
        x = sim.get_random_x(1)
        
        # Run the simulation with the generated parameters
        eval = sim.run_sim(x)
        evalset = x.copy()
        evalset['objective'] = np.sum(eval[0])
        evals.append(evalset)

        # Track the elapsed time for each trial
        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    # Save results to a CSV file
    randomsearch = pd.DataFrame.from_records(evals)
    randomsearch.to_csv(path)