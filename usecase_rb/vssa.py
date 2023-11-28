from config import *
from src.utils import *
from simulation import *

from src.simulatedannealing import * 


if __name__ == '__main__':

        # user input: number of maximum iterations optimization
        start = time.time()
        MAXITER = int(sys.argv[1]) 

        # user input: number of trials
        ntrials = int(sys.argv[2]) 

        # baseline simulated annealing
        start = time.time()
        si = Simulation(simulation_rb, vals, vars)
        simaneal = partial(simulated_annealing, MAXITER=MAXITER)
        with Pool(processes=ntrials) as pool:
            result = pool.map(simaneal, [si]*ntrials) 
        
        total_timeSA = time.time()-start
        result = pd.DataFrame.from_records(result)

        with open('../../surdata/SA_starlight_iter-'+str(MAXITER)+'_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M")+'.pkl', 'wb') as file:
                pickle.dump([result,[total_timeSA]], file)
        print('time:', time.time()-start)
