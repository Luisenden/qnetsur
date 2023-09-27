import sys
import pickle
from utils import *

if __name__ == '__main__':

    # network topology type
    vv = sys.argv[1]
    v = vv.split(',') 
    assert(len(v) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'
    A = main.adjacency_squared(int(v[0])) if len(v)==1 else main.adjacency_tree(int(v[0]), int(v[1]))
    topo = 'square' if len(v)==1 else 'tree'

    
    # number of maximum iterations during optimiztion
    MAXITER = int(sys.argv[2]) 
    
    n = 50 # size of initial training set
    nref = n+MAXITER # reference model

    vals = {
            'A': A,
            'protocol':'srs', 
            'p_cons': 0.225, 
            'p_gen': 0.9, 
            'p_swap':1,  
            'return_data':'avg', 
            'progress_bar': None,
            'cutoff': 50,
            'total_time': 1000,
            'N_samples' : 10,
            }
    vars = {
            'M': [1, 10],
            'qbits_per_channel': [3,50],
            'q_swap': [0., 1.],
            } 
    
    s = Surrogate(main.simulation_cd, A=A, vals=vals, vars=vars, n=n)
    surrogate_optimize(s, MAXITER=MAXITER, verbose=True)

    sref = Surrogate(main.simulation_cd, A=A, vals=vals, vars=vars, n=nref)

    with open('../surdata/'+topo+vv.replace(',','')+'_iter-'+str(MAXITER)+'_objective-meanopt.pkl', 'wb') as file:
        pickle.dump([s,sref], file)