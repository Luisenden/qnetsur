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

    
    s = Surrogate(main.simulation_cd, A=A, n=n)
    surrogate_optimize(s, MAXITER=MAXITER, verbose=True)

    sref = Surrogate(main.simulation_cd, A=A, n=nref)

    with open('../surdata/'+topo+vv.replace(',','')+'_iter-'+str(MAXITER)+'_objective-meanopt.pkl', 'wb') as file:
        pickle.dump([s,sref], file)