from config import *
from src.utils import *
from simulation import *

from src.simulatedannealing import * 


if __name__ == '__main__':

        # user input:
        max_time = MAX_TIME * 3600 # in sec

        # baseline simulated annealing
        si = Simulation(simulation_rb, vals, vars)
        simaneal = partial(simulated_annealing, MAX_TIME=max_time)
        
        result = simaneal(si, seed=SEED_OPT)
        result = pd.DataFrame.from_records(result)

        with open(f'../../surdata/SA_starlight_{MAX_TIME:.1f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(result, file)
