import sys
import numpy as np
from scipy.stats import truncnorm

from optimizingcd import main_cd as sim
from utils import Simulation
from specifications import *

def objective(s, x :dict) -> float:

    eval = s.run_sim(x)#[0]
    return eval#-np.mean(eval)


def get_neighbour(s, x :dict) -> dict:
    """
    Generates random parameters for the simulation.

    Args:
        n (int): Number of random parameter sets to generate.

    Returns:
        dict: Randomly generated parameters.
    """
    
    assert all(isinstance(val, list) for val in s.vars.values()), f"Dimension types must be list!"

    x_n = {}
    for dim, vals in s.vars.items():
        if len(vals) > 2:
            x_n[dim] = np.random.choice(vals)
        elif all(isinstance(x, int) for x in vals):
            std = (vals[1] - vals[0])/2
            x_n[dim] = int(truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=1)[0])
        elif all(isinstance(x, float) for x in vals):
            std = (vals[1] - vals[0])/2
            x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=1)[0]         

    return x_n


def simulated_annealing(s, temp :int = 10, beta_schedule :int = 100, MAXITER = 5, seed=42):

    np.random.seed(seed)

    y = []
    
    # generate an initial point
    current = s.get_random_x(1)

    # evaluate the initial point
    current_eval = objective(s,current)
    # print('initial value %.2f' % current_eval)

    # optimize
    count = 0
    t = temp 
    while t > 1e-5 and count < MAXITER:

        # cooling 
        t = temp / (count + 1)
        y.append(current_eval)
        # repeat
        for _ in range(beta_schedule):

            # choose a different point and evaluate
            candidate = get_neighbour(s, current)
            candidate_eval = objective(s,candidate)

            # check for new best solution
            if candidate_eval < current_eval:
                # store new best point
                current, current_eval = candidate, candidate_eval

            # calculate metropolis acceptance criterion
            diff = candidate_eval - current_eval
            metropolis = np.exp(-diff / t)

            # keep if improvement or metropolis criterion satisfied
            if diff < 0 or np.random.random() < metropolis:
                current, current_eval = candidate, candidate_eval
        
    y.append(current_eval)
    # print('result value %.2f' % current_eval)
    return current, y

if __name__ == '__main__':

        # user input: network topology type
        vv = sys.argv[1]
        v = vv.split(',') 

        # user input: number of maximum iterations during optimiztion
        MAXITER = int(sys.argv[2]) 

        assert(len(v) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'
        topo = NetworkTopology((int(v[0]), ), 'square') if len(v)==1 else NetworkTopology((int(v[0]), int(v[1])), 'tree')

        start = time.time()
        s = Simulation(sim.simulation_cd, topo, vals, vars)
        simulated_annealing(s, MAXITER=MAXITER)
        print('total time', time.time()-start)
