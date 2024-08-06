from datetime import datetime
import pandas as pd

from config import Config
from qnetsur.datacollector import SurrogateCollector
from qnetsur.utils import Surrogate


if __name__ == '__main__':
        
        # load configuration
        conf = Config()
        conf.set_default_values()
        limit = conf.args.time
        limit_kind = 'hours' if isinstance(limit, float) else 'cycles'
        storage_path = conf.args.folder+f'SU_{conf.name}_{limit}{limit_kind}_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.csv'

        # instatiate surrogate model
        sim = Surrogate(conf.simobjective, conf.sim, rng=conf.rng, values=conf.vals,\
                        variables=conf.vars, initial_training_size=conf.evaluation_sample_size, ntop=conf.evaluation_sample_size)
        
        # run optimization
        sim.optimize(limit=limit, isscore=conf.args.score, verbose=True)

        # store fixed input values used
        pd.DataFrame.from_dict(sim.vals, 'index').to_csv(conf.args.folder+'SIMULATION_INPUT_VALUES.csv')
        
        # collect and store results
        coll = SurrogateCollector(sim=sim)
        result = coll.get_total()
        result.to_csv(storage_path)
