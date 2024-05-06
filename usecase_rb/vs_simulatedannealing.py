import pickle
import pandas as pd
from datetime import datetime
from config import MAX_TIME, simwrapper, vals, vars, SEED
from src.utils import Simulation
from simulation import simulation_rb

from src.simulatedannealing import simulated_annealing


if __name__ == '__main__':

        # user input:
        max_time = MAX_TIME * 3600 # in sec

        # baseline simulated annealing
        si = Simulation(simwrapper, simulation_rb, vals=vals, vars=vars)
        result = simulated_annealing(sim=si, MAX_TIME=max_time, seed=SEED)
        result = pd.DataFrame.from_records(result)

        with open(f'../../surdata/rb_budget/SA_starlight_{MAX_TIME:.1f}h_objective-budget_SEED{SEED}_'
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump([result, vals], file)
