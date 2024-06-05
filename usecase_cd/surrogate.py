
import numpy as np
from datetime import datetime
import pandas as pd

from config import Config
from src.datacollectors import SurrogateCollector
from src.utils import Surrogate


if __name__ == '__main__':
        
        # load configuration
        conf = Config()
        limit = conf.args.time
        limit_kind = 'hours' if isinstance(limit, float) else 'cycles'
        storage_path = conf.args.folder+f'SU_{conf.name}_{limit}{limit_kind}_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl'

        # instatiate surrogate model
        sim = Surrogate(conf.simobjective, conf.sim, rng=conf.rng, values=conf.vals,\
                        variables=conf.vars, sample_size=conf.initial_model_size)
        
        # run optimization
        sim.optimize(limit=limit, verbose=True)

        # store user nodes
        if conf.kind == 'randtree':
                sim.vals['user'] = np.where(sim.vals['A'].sum(axis=1) == 1)
        pd.DataFrame.from_dict(sim.vals, 'index').to_pickle(conf.args.folder+'SIMULATION_INPUT_VALUES.pkl')
        
        # collect and store results
        coll = SurrogateCollector(sim=sim)
        result = coll.get_total()
        result.to_pickle(storage_path)



        