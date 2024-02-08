from config import *
from src.utils import *

from simulation import *

if __name__ == '__main__':

        # # user input:
        # max_time= MAX_TIME * 3600 # in sec

        # # instatiate surrogate model and run optimization
        # sim = Surrogate(simulation, vals=vals, vars=vars, sample_size=sample_size)
        # sim.optimize(max_time=max_time, verbose=False)
        
        # with open(f'../../surdata/test_{max_time}h_objective-meanopt_SEED{SEED}'+'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")'+'.pkl', 'wb') as file:
        #         pickle.dump(sim, file)

        sim = Simulation(simulation, vals, vars)
        x = sim.get_random_x(1)
        print(x)
        print(sim.run_sim(x))