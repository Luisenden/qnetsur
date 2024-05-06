import pickle
from datetime import datetime
from config import MAX_TIME, simwrapper, sample_size, vals, vars, SEED
from src.utils import Surrogate

from simulation import simulation_rb

# user input:
max_time= MAX_TIME * 3600 # in sec
output_folder = '../../surdata/rb_budget/'

# instatiate surrogate model and run optimization
sur = Surrogate(simwrapper, simulation_rb, sample_size=sample_size, vals=vals, vars=vars, k=6)
sur.optimize(limit=max_time, verbose=False)


with open(output_folder+f'SU_starlight_{MAX_TIME:.1f}h_objective-budget_SEED{SEED}_'
          +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
        pickle.dump(sur, file)