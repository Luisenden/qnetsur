
from config import *
from src.utils import *

from simulation import simulation_qswitch

from src.simulatedannealing import * 




if __name__ == '__main__':
        
        # user input:
        max_time= MAX_TIME * 3600 # in sec

        # instatiate surrogate model and run optimization
        sim = Surrogate(simulation_qswitch, vals=vals, vars=vars, sample_size=initial_model_size)
        sim.optimize(max_time=max_time, verbose=True)

        with open(f'../../surdata/Sur_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sim, file)
