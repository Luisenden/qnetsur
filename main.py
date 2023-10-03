import sys
import pickle
from utils import *
from datetime import datetime

from optimizingcd import main_cd
from specifications import *

from simulatedannealing import *


if __name__ == '__main__':

        # user input: network topology type
        vv = sys.argv[1]
        v = vv.split(',') 

        assert(len(v) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'
        topo = NetworkTopology((int(v[0]), ), 'square') if len(v)==1 else NetworkTopology((int(v[0]), int(v[1])), 'tree')

        # user input: number of maximum iterations optimiztion
        MAXITER = int(sys.argv[2]) 
    
        # instatiate surrogate model and run optimization
        start = time.time()
        s = Surrogate(main_cd.simulation_cd, topo, vals=vals, vars=vars, initial_model_size=initial_model_size)
        s.optimize(MAXITER=MAXITER, verbose=False)
        total_time = time.time()-start

        # baseline 
        start = time.time()
        MAXITERSA = s.procs*MAXITER + initial_model_size
        si = Simulation(main_cd.simulation_cd, topo, vals, vars)
        xSA, ySA = simulated_annealing(si, MAXITER=MAXITERSA)
        total_timeSA = time.time()-start

        # print('optimization:', np.mean(s.y[-1]))
        # print('time total', total_time)

        # reference model
        # initial_ref_size = initial_model_size+MAXITER # reference model
        # sref = Surrogate(main_cd.simulation_cd, topo, vals=vals, vars=vars, initial_model_size=initial_ref_size)

        with open('../surdata/'+topo.name+vv.replace(',','')+'_iter-'+str(MAXITER)+'_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M")+'.pkl', 'wb') as file:
                pickle.dump([s,total_time, xSA, ySA, total_timeSA], file)