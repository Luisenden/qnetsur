
import pandas as pd
from datetime import datetime

from config import Config
from qnetsur.utils import Simulation
from qnetsur.simulatedannealing import simulated_annealing 


if __name__ == '__main__':

        # load configuration
        conf = Config()
        limit = conf.args.time
        path = conf.args.folder+f'SA_{conf.name}_{limit}hours_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.csv'

        # baseline simulated annealing
        sim = Simulation(conf.simobjective, conf.sim, values=conf.vals, variables=conf.vars, rng=conf.rng)
        
        result = simulated_annealing(sim, limit=limit)
        result = pd.DataFrame.from_records(result)
        result.to_csv(path)
