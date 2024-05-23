
import pandas as pd
from datetime import datetime
import pickle

from config import Config
from src.utils import Simulation

from optimizingcd import main_cd as simulation
from src.simulatedannealing import simulated_annealing 


if __name__ == '__main__':

        # load configuration
        conf = Config(initial_model_size=5)
        limit = conf.args.time

        # baseline simulated annealing
        sim = Simulation(conf.simobjective, simulation.simulation_cd, values=conf.vals, variables=conf.vars, rng=conf.rng)
        
        result = simulated_annealing(sim, limit=limit)
        result = pd.DataFrame.from_records(result)

        limit_kind = 'hours' if isinstance(limit, float) else 'cycles'
        with open(conf.args.folder+f'SA_{conf.name}_{limit}{limit_kind}_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(result, file)
