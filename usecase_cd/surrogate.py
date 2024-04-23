
from config import *
from src.utils import *
from optimizingcd import main_cd as simulation
from src.simulatedannealing import * 


if __name__ == '__main__':
        
        # user input:
        max_time= [MAX_TIME * 3600, "timer"] # in sec

        # instatiate surrogate model and run optimization
        sim = Surrogate(simwrapper, simulation.simulation_cd, vals=vals, vars=vars, sample_size=initial_model_size)
        sim.optimize(limit=max_time, verbose=True)

        if topo.name == 'randtree':
                sim.vals['user'] = np.where(sim.vals['A'].sum(axis=1) == 1)

        with open(f'../../surdata/cd/SU_{topo.raw}_{MAX_TIME}{max_time[1]}_objective-meanopt_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sim, file)
