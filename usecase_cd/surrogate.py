
import numpy as np
from functools import partial
import pickle
from datetime import datetime

from config import Config
from src.utils import Surrogate
from optimizingcd import main_cd as simulation


if __name__ == '__main__':
        
        # load configuration
        conf = Config(initial_model_size=5)
        limit = conf.args.time

        # instatiate surrogate model and run optimization
        sim = Surrogate(partial(conf.simobjective), simulation.simulation_cd, rng=conf.rng, values=conf.vals,\
                        variables=conf.vars, sample_size=conf.initial_model_size)
        sim.optimize(limit=limit, verbose=True)

        if conf.kind == 'randtree':
                sim.vals['user'] = np.where(sim.vals['A'].sum(axis=1) == 1)

        limit_kind = 'hours' if isinstance(limit, float) else 'cycles'
        with open(conf.args.folder+f'SU_{conf.name}_{limit}{limit_kind}_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sim, file)
