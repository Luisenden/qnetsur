import sys
import pickle
from utils import *
from datetime import datetime
from time import time

from rb_simulation import *
from rb_specifications import *


if __name__ == '__main__':

        # user input: number of maximum iterations optimization
        MAXITER = int(sys.argv[1]) 

        # user input: number of trials
        ntrials = int(sys.argv[2]) 

        # instatiate surrogate model and run optimization
        total_time = []
        sims = []
        for _ in range(ntrials):
                start = time()
                s = Surrogate(simulation_rb, vals=vals, vars=vars, initial_model_size=initial_model_size)
                s.optimize(MAXITER=MAXITER, verbose=False)
                sims.append(s)
                total_time.append(time()-start)
        

        with open('../../surdata/Sur_starlight_iter-'+str(MAXITER)+'_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M")+'.pkl', 'wb') as file:
                pickle.dump([sims,total_time], file)
