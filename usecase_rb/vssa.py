from config import *
from src.utils import *
from simulation import *

from src.simulatedannealing import * 


if __name__ == '__main__':

        # user input:
        MAXTIME= float(sys.argv[1]) * 3600 # in sec

        # user input: number of trials
        ntrials = int(sys.argv[2]) 

        # baseline simulated annealing
        si = Simulation(simulation_rb, vals, vars)
        simaneal = partial(simulated_annealing, MAXTIME=MAXTIME)
        with Pool(processes=ntrials) as pool:
            result = pool.map(simaneal, [si]*ntrials) 
        
        result = pd.DataFrame.from_records(result)

        with open('../../surdata/SA_starlight_'+str(MAXTIME)+'h_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(result, file)
