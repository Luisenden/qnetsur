import time
import numpy as np
from scipy.stats import truncnorm


def objective(sim, x :dict) -> float:
    """Evaluate the objective function at given parameter settings.

    Args:
        sim (Simulation): The simulation environment.
        x (dict): The parameter settings at which to evaluate.

    Returns:
        float: The negative sum of evaluations (to be minimized).
    """
    eval = sim.run_sim(x)[0] 
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
        f"Dimension types must be a tuple (sample list, dataype) and n must be greater zero."

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

def get_neighbour(sim, x :dict, rng) -> dict:
    """
    Generates random parameters for the simulation from neighborhood of `x`.

    Args:
        n (int): Number of random parameter sets to generate.

    Returns:
        dict: Randomly generated parameters.
    """
    x_n = {}
    for dim, par in sim.vars['range'].items():
            vals = par[0]
            if par[1] == 'int':
                std = (vals[1] - vals[0])/2
                x_n[dim] = int(truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=1, random_state=rng)[0])
            elif par[1] == 'float':
                std = (vals[1] - vals[0])/2
                x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=1, random_state=rng)[0]  
            else:
                raise Exception('Datatype must be "int" or "float".')
            
                
    for dim, vals in sim.vars['choice'].items():
            x_n[dim] = rng.choice(vals)        

    return x_n


def simulated_annealing(sim, limit, temp :int = 10, beta_schedule :int =5) -> list:
    """
    Global optimizing method Simulated Annealing (Kirkpatrick et al., 1983)

    Args:
        sim (Simulation): An instance of the Simulation class that provides methods 
                        and properties necessary for the simulation environment.
        limit (float): The maximum time allowed for the optimization process in seconds.
        temp (int, optional): The initial temperature for the annealing process. Default is 10.
        beta_schedule (int, optional): The number of iterations for each temperature level. Default is 5.
        seed (int, optional): The seed for the random number generator to ensure reproducibility. Default is 42.

    Returns:
        list: A list of dictionaries, where each dictionary represents the state of the system at each step
            of the optimization. Each dictionary contains keys for the parameter values, their corresponding 
            utility, and the computational time taken for that step.
    """
    start_initial = time.time()
    rng = sim.rng
    
    # generate an initial point
    current = get_random_x(sim, 1, rng)

    # evaluate the initial point
    current_eval = objective(sim, current)
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
    dt_beta = 0
    while t > 1e-20 and time_tracker+dt_beta < limit *3600:

        # cooling 
        t = temp / (count + 1)
        
        start = time.time()
        # repeat
        for _ in range(beta_schedule):
            start_beta = time.time()
            # choose a different point and evaluate
            candidate = get_neighbour(sim, current, rng)
            candidate_eval = objective(sim, candidate)

            # compare candidate to current solution
            if candidate_eval < current_eval:
                # select candidate over current solution 
                current, current_eval = candidate, candidate_eval
            dt_beta = time.time() - start_beta
            time_tracker += dt_beta

            if time_tracker+dt_beta >= limit:
                break

        # calculate metropolis acceptance criterion
        diff = candidate_eval - current_eval
        metropolis = np.exp(-diff / t)

        # replace if improvement or metropolis satisfied
        r = rng.random()
        if diff < 0 or r < metropolis:
            current, current_eval = candidate, candidate_eval
        
        dt = time.time()-start

        current_set = current.copy()
        current_set['objective'] = -current_eval
        current_set['time'] = dt
        sets.append(current_set)
        count += 1 

    return sets
