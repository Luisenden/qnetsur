"""
Global optimizing method Simulated Annealing (Kirkpatrick et al., 1983)
"""

import time
import numpy as np
from scipy.stats import truncnorm

from src.utils import *


def objective(s, x :dict) -> float:
    eval = s.run_sim(x)[0] 
    return -np.sum(eval)

def get_random_x(self, n, rng) -> dict:
    """
    Generates random parameters for the simulation.

    Args:
        n (int): Number of random parameter sets to generate.

    Returns:
        dict: Randomly generated parameters.
    """
    assert all(isinstance(val, tuple) for val in self.vars['range'].values()) and n > 0,\
        f"Dimension types must be a tuple (sample-list, dataype) and n must be greater zero."

    x = {}
    for dim, par in self.vars['range'].items():
            vals = par[0]
            if par[1] == 'int':
                x[dim] = rng.integers(vals[0], vals[1], n) if n > 1\
                    else rng.integers(vals[0], vals[1])
            elif par[1] == 'float':
                x[dim] = rng.uniform(vals[0], vals[1], n) if n > 1\
                    else rng.uniform(vals[0], vals[1])
            else:
                raise Exception('Datatype must be "int" or "float".')

    for dim, vals in self.vars['ordinal'].items():
            x[dim] = rng.choice(vals, size=n) if n > 1\
                else rng.choice(vals)
                
    for dim, vals in self.vars['choice'].items():
            x[dim] = rng.choice(vals, n) if n > 1\
                else rng.choice(vals)       

    return x

def get_neighbour(s, x :dict, rng) -> dict:
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
                x_n[dim] = int(truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=1, random_state=rng)[0])
            elif par[1] == 'float':
                std = (vals[1] - vals[0])/2
                x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=1, random_state=rng)[0]  
            else:
                raise Exception('Datatype must be "int" or "float".')
                
    for dim, vals in s.vars['choice'].items():
            x_n[dim] = rng.choice(vals)        

    return x_n


def simulated_annealing(s, MAX_TIME, temp :int = 10, beta_schedule :int = 5, seed=42):

    start_initial = time.time()
    rng = np.random.default_rng(seed=seed)
    
    # generate an initial point
    current = get_random_x(s, 1, rng)

    # evaluate the initial point
    current_eval = objective(s,current)
    current_set = current.copy()
    current_set['objective'] = -current_eval
    dt_init = time.time()-start_initial
    current_set['time'] = dt_init
    
    sets = []
    sets.append(current_set)

    # optimize
    count = 0
    time_tracker = 0
    t = temp 
    while t > 1e-5 and time_tracker < MAX_TIME:

        # cooling 
        t = temp / (count + 1)
        
        start = time.time()
        # repeat
        for _ in range(beta_schedule):
            start_beta = time.time()
            # choose a different point and evaluate
            candidate = get_neighbour(s, current, rng)
            candidate_eval = objective(s, candidate)

            # check for new best solution
            if candidate_eval < current_eval:
                # store new best point
                current, current_eval = candidate, candidate_eval
            dt_beta = time.time() - start_beta
            time_tracker += dt_beta

            if time_tracker >= MAX_TIME:
                break

        # calculate metropolis acceptance criterion
        diff = candidate_eval - current_eval
        metropolis = np.exp(-diff / t)
        if metropolis > 1:
            print(metropolis) 

        # keep if improvement or metropolis criterion satisfied
        r = rng.random()
        if diff < 0 or r < metropolis:
            current, current_eval = candidate, candidate_eval
        
        dt = time.time()-start

        current_set = current.copy()
        current_set['objective'] = -current_eval
        current_set['time'] = dt
        sets.append(current_set)

        if time_tracker >= MAX_TIME:
            break

        count += 1 

    return sets
