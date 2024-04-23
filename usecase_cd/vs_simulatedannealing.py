from config import *
from src.utils import Simulation

from optimizingcd import main_cd as simulation

from src.simulatedannealing import simulated_annealing 


if __name__ == '__main__':

        # user input:
        max_time= [MAX_TIME * 3600, "timer"]

        # baseline simulated annealing
        si = Simulation(simwrapper, simulation.simulation_cd, vals=vals, vars=vars)
        simaneal = partial(simulated_annealing, MAX_TIME=max_time[0], seed=SEED)
        
        result = simaneal(si)
        result = pd.DataFrame.from_records(result)

        with open(f'../../surdata/cd/SA_{topo.raw}_{MAX_TIME}{max_time[1]}_objective-meanopt_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(result, file)
