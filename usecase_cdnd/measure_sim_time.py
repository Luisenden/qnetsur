from config import *
from src.utils import *

from optimizingcd import main_cd as simulation

def set_q_swap_for_nd(**kwargs):
    series = pd.Series(kwargs)
    d = series[~series.index.str.contains('q_swap')] # filter all vars not containing 'q_swap'
    kwargs = pd.concat([d, pd.Series([series[series.index.str.contains('q_swap')].values], index=['q_swap'])]) # concatenate with 'q_swap' which is now a vector
    kwargs.to_dict()
    return kwargs
    

if __name__ == '__main__':

    tracked_sim_time = []
    
    for _ in range(10):
        sim = Simulation(simulation.simulation_cd, vals, vars)
        x = sim.get_random_x(1)
        x = {**x, **vals}
        x = set_q_swap_for_nd(**x)

        start = time.time()
        simulation.simulation_cd(**x)
        tracked_sim_time.append(time.time()-start)
    
    print(tracked_sim_time)

    with open(f'../../surdata/measured_sim_time_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(tracked_sim_time, file)
    