from config import *
from src.utils import *

from simulation import *

if __name__ == '__main__':

        # user input:
        max_time= MAX_TIME * 3600 # in sec

        # instatiate surrogate model and run optimization
        sim = Surrogate(simulation_rb, vals=vals, vars=vars, sample_size=sample_size)
        sim.optimize(max_time=max_time, verbose=False)
        
        with open(f'../../surdata/Sur_starlight_{MAX_TIME:.0f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sim, file)