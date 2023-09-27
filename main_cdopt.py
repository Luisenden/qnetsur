import sys
import pickle
from utils import *

from optimizingcd import main_cd
from specifications import vars,vals


if __name__ == '__main__':

        # network topology type
        vv = sys.argv[1]
        v = vv.split(',') 
        assert(len(v) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'
        A = main_cd.adjacency_squared(int(v[0])) if len(v)==1 else main_cd.adjacency_tree(int(v[0]), int(v[1]))
        vals['A'] = A

        topo = 'square' if len(v)==1 else 'tree'

        # number of maximum iterations during optimiztion
        MAXITER = int(sys.argv[2]) 

        n = 50 # size of initial training set
        nref = n+MAXITER # reference model

    
        s = Surrogate(main_cd.simulation_cd, vals=vals, vars=vars, n=n)
        surrogate_optimize(s, MAXITER=MAXITER, verbose=True)

        sref = Surrogate(main_cd.simulation_cd, vals=vals, vars=vars, n=nref)

        with open('../surdata/'+topo+vv.replace(',','')+'_iter-'+str(MAXITER)+'_objective-meanopt.pkl', 'wb') as file:
                pickle.dump([s,sref], file)