"""
Script to compare Simulated Annealing to scipy optimizers for several test functions.
"""

import numpy as np
import time, pickle
from simulatedannealing import *
from scipy.optimize import minimize
from multiprocessing import Pool

# test functions
rosenbrock = lambda x,y,z,w: sum([(1-[x,y][i])**2+100*([x,y][i+1]-[x,y][i]**2)**2 for i in range(1)]) + 0*(z+w) # Continious Rosenbrock function with optimal at (1,1) https://irene.readthedocs.io/en/latest/benchmarks.html
bartelsconn = lambda x,y,z,w: abs(x**2+y**2+x*y) + abs(np.sin(x)) + abs(np.cos(y)) + 0*(z+w) # nr.9 continuous, differentiable https://arxiv.org/pdf/1308.4008.pdf
schaffer = lambda x,y,z,w: 0.5+ (np.sin(x**2-y**2)**2 - 0.5) / (1+0.001*(y**2+x**2))**2 + 0*(z+w)# continuous, non-diff.
step = lambda x,y,z,w: sum([np.floor(i+0.5)**2 for i in [x,y,z,w]]) # nr.139  discont.
sign = lambda x,y,z,w: sum([np.sign(i) for i in [x,y,z,w]]) # discont.

funs = [rosenbrock, bartelsconn, schaffer, step, sign]

if __name__ == '__main__':

    vars = {}
    dims = 'xyzw'
    for i in dims:
        vars[i] =  [-100.,100.]

    vals = {}

    sims = [Simulation(fun, vals=vals, vars=vars) for fun in funs]
            
    # Optimization using simulated annealing
    simannealsols = [simulated_annealing(si, MAX_TIME=5)[1][-1] for si in sims]

    # Optimization using SciPy
    scipysols = []
    bounds = list(vars.values())
    for fun in funs:    
        scipysols.append( minimize(lambda input: fun(*input),
                                   x0=np.random.uniform(bounds[0][0], bounds[0][1], size=len(dims)), bounds=bounds, method='L-BFGS-B').fun )

    # store
    with open('../surdata/scipy_vs_simanneal.pkl', 'wb') as file:
        pickle.dump(zip(simannealsols, scipysols), file)