"""
Script to compare simulated annealing to scipy optimizers for different test functions.
"""
import sys
sys.path.append('../')
import numpy as np
from src.simulatedannealing import *
from scipy.optimize import minimize
from config import *

# test functions
rosenbrock = lambda x,y,z,w: sum([(1-[x,y,z,w][i])**2+100*([x,y,z,w][i+1]-[x,y,z,w][i]**2)**2 for i in range(3)]) # Continious Rosenbrock function with optimal at (1,1,1,1) https://irene.readthedocs.io/en/latest/benchmarks.html
bartelsconn = lambda x,y,z,w: abs(x**2+y**2+x*y) + abs(np.sin(x)) + abs(np.cos(y)) + 0*(z+w) # nr.9 continuous, differentiable https://arxiv.org/pdf/1308.4008.pdf
schaffer = lambda x,y,z,w: 0.5+ (np.sin(x**2-y**2)**2 - 0.5) / (1+0.001*(y**2+x**2))**2 + 0*(z+w)# nr. 136 continuous, non-diff.
step = lambda x,y,z,w: sum([np.floor(i+0.5)**2 for i in [x,y,z,w]]) # nr.139  discont.
sign = lambda x,y,z,w: sum(np.sign([x,y,z,w])) # discont.

funs = [rosenbrock, bartelsconn, schaffer, step, sign]

if __name__ == '__main__':

    results = dict()
    
    # set variable ranges
    dims = 'xyzw'
    for i in dims:
        vars['range'][i] =  ([-5., 5.], 'float')
            
    # Optimization using simulated annealing
    sims = [Simulation(sim_wrapper=simwrapper, sim=fun, vals=vals, vars=vars) for fun in funs]
    simannealsols = [simulated_annealing(si, MAX_TIME=20)[-1]['objective'] for si in sims]
    results['Simulated Annealing'] = np.array(simannealsols)*(-1)

    # Optimization using SciPy
    scipysols = []
    bounds = [value[0] for _,value in vars['range'].items()]
    for fun in funs:    
        scipysols.append( minimize(lambda input: fun(*input),
                                   x0=np.random.uniform(bounds[0][0], bounds[0][1], size=len(dims)), bounds=bounds, method='L-BFGS-B').fun )
    results['Scipy'] = scipysols

    results['Optimal'] = [0, 1, 0, 0, -4] # references 

    # store
    df = pd.DataFrame.from_dict(results)
    print(df)
    df.to_csv('scipy_vs_simulatedannealing.csv')
    
    