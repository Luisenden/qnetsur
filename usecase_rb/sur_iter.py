from config import *
from src.utils import *

from simulation import *

if __name__ == '__main__':

        # user input:
        max_iteration =[50, 0]  # maximum allowed optimization time in seconds [*, 1] or number of iterations [*, 0]

        # instatiate surrogate model and run optimization
        sur = Surrogate(simwrapper, simulation_rb, sample_size=sample_size, vals=vals, vars=vars, k=6)
        sur.optimize(max_time=max_iteration, verbose=False)

        
        with open(f'../../surdata/rb_budget/SU_starlight_{max_iteration[0]}iter_objective-budget_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sur, file)