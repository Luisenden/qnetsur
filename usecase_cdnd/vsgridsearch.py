from config import *
from src.utils import *

from optimizingcd import main_cd as simulation
 

if __name__ == '__main__':

    # user input:
    max_time= MAX_TIME * 3600 # in sec

    evals = [] # storage for results

    times_tracked = []
    time_tracker = 0
    delta = 0
    while time_tracker + delta < max_time:
        start = time.time()

        sim = Simulation(simulation.simulation_cd, vals, vars)
        x = sim.get_random_x(1)
        eval = sim.run_sim(x)
        evalset = x.copy()
        evalset['objective'], evalset['std'], evalset['raw'] = eval
        evals.append(evalset)

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    gridsearch = pd.DataFrame.from_records(evals)
    with open(f'../../surdata/GS_ND_{topo.name}{TOPO}_{MAX_TIME:.2f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([gridsearch,time_tracker,vals], file)