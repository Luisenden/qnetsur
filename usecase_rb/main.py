from config import *
from src.utils import *

from simulation import *


if __name__ == '__main__':

        # user input:
        max_time= float(sys.argv[1]) * 3600 # in sec

        # instatiate surrogate model and run optimization
        sim = Surrogate(simulation_rb, vals=vals, vars=vars, sample_size=sample_size)
        sim.optimize(max_time=max_time, verbose=False)
        
        with open('../../surdata/Sur_starlight_'+str(max_time)+'h_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sim, file)