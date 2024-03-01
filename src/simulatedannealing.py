import sys, time
import numpy as np
from scipy.stats import truncnorm

from src.utils import *


def objective(s, x :dict) -> float:
    eval = s.run_sim(x)[0] 
    return -np.sum(eval)


def get_neighbour(s, x :dict) -> dict:
    """
    Generates random parameters for the simulation.

    Args:
        n (int): Number of random parameter sets to generate.

    Returns:
        dict: Randomly generated parameters.
    """
    x_n = {}
    for dim, par in s.vars['range'].items():
            vals = par[0]
            if par[1] == 'int':
                std = (vals[1] - vals[0])/2
                x_n[dim] = int(truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=1)[0])
            elif par[1] == 'float':
                std = (vals[1] - vals[0])/2
                x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=1)[0]  
            else:
                raise Exception('Datatype must be "int" or "float".')
                
    for dim, vals in s.vars['choice'].items():
            x_n[dim] = np.random.choice(vals)        

    return x_n


def simulated_annealing(s, MAX_TIME, temp :int = 10, beta_schedule :int = 5, seed=42):

    
    np.random.seed(seed)
    
    # generate an initial point
    current = s.get_random_x(1)

    # evaluate the initial point
    current_eval = objective(s,current)
    current_set = current.copy()
    current_set['objective'] = -current_eval

    sets = []
    sets.append(current_set)

    # optimize
    count = 0
    time_tracker = 0
    t = temp 
    while t > 1e-5 and time_tracker < MAX_TIME:

        # cooling 
        t = temp / (count + 1)
        
        start_outer = time.time()
        # repeat
        for _ in range(beta_schedule):
            start = time.time()

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
            r = np.random.random()
            if diff < 0 or r < metropolis:
                current, current_eval = candidate, candidate_eval
            
            time_tracker += time.time()-start
            if time_tracker >= MAX_TIME:
                break
        
        current_set = current.copy()
        current_set['objective'] = -current_eval
        current_set['time'] = time.time()-start_outer
        sets.append(current_set)

        count += 1 

    return sets
