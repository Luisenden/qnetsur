from config import *
from src.utils import *

from simulation import *

if __name__ == '__main__':

        # user input:
        max_time= MAX_TIME * 3600 # in sec

        # instatiate surrogate model and run optimization
        sur = Surrogate(simwrapper, simulation_rb, sample_size=sample_size, vals=vals, vars=vars, k=6)
        sur.optimize(max_time=max_time, verbose=False)

        
        with open(f'../../surdata/rb/SU_starlight_{MAX_TIME:.1f}h_objective-penal_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sur, file)