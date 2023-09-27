import sys
import pickle
from utils import *

if __name__ == '__main__':
    
        # Optimization using SA
        def objective_function(x,y):
                return  [(1-x)**2+100*(y-x**2)**22, (1-x), (y-x**2)**22] # Rosenbrock function with optimal at (1,1) https://irene.readthedocs.io/en/latest/benchmarks.html
    
        # number of maximum iterations during optimiztion
        MAXITER = int(sys.argv[1]) 

        n = 5 # size of initial training set
        nref = n+MAXITER # reference model

        vars = {'x':[-3.,3.], }
        vals = {'y': 1}

        sim = Simulation(objective_function, vals, vars)
        sim.run_sim({'x': 3})
 
        # s = Surrogate(objective_function, vals=vals, vars=vars, n=n)
        # surrogate_optimize(s, MAXITER=MAXITER, verbose=True)

        # sref = Surrogate(objective_function, vals=vals, vars=vars, n=nref)

        # with open('../surdata/rosenbrock_iter-'+str(MAXITER)+'_objective-meanopt.pkl', 'wb') as file:
        #         pickle.dump([s,sref], file)