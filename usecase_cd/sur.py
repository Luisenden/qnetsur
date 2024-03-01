
from config import *
from src.utils import *
from optimizingcd import main_cd as simulation
from src.simulatedannealing import * 


if __name__ == '__main__':
        
        # user input:
        max_time= MAX_TIME * 3600 # in sec

        # instatiate surrogate model and run optimization
        sim = Surrogate(simwrapper, simulation.simulation_cd, vals=vals, vars=vars, sample_size=initial_model_size)
        sim.optimize(max_time=max_time, verbose=False)

        with open(f'../../surdata/SU_{topo.name}{TOPO}_{MAX_TIME:.2f}h_objective-meanopt_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sim, file)
