from config import *
from src.utils import *

from optimizingcd import main_cd as simulation

from src.simulatedannealing import simulated_annealing 


if __name__ == '__main__':

        # user input:
        max_time = MAX_TIME * 3600 # in sec

        # baseline simulated annealing
        si = Simulation(simulation.simulation_cd, vals, vars)
        simaneal = partial(simulated_annealing, MAX_TIME=max_time)
        
        result = simaneal(si, seed=SEED_OPT)
        result = pd.DataFrame.from_records(result)

        with open(f'../../surdata/SA_ND_{topo.name}_{MAX_TIME:.1f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(result, file)
