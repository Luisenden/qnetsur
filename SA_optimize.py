import numpy as np
from scipy.stats import truncnorm

def objective(s, x :dict) -> float:
    """
    Computes the objective function value for a set of parameters.

    Args:
        x (dict): Parameters for the simulation.

    Returns:
        float: Objective function value.
    """
    x_list = list(x.values())
    eval = s.mmodel.predict([x_list])
    return -np.mean(eval)

def get_candidate(s) -> dict:

    x = {}
    cols = s.X_df.select_dtypes(int)
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.randint(col.min(), col.max())
    cols = s.X_df.select_dtypes(float)
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.uniform(col.min(), col.max())
    cols = s.X_df.select_dtypes(exclude=(int,float))
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.choice(col)
  
    return x 

def get_neighbour(s, temp, temp_init, x :dict) -> dict:

    x_n = {}

    cols = s.X_df.select_dtypes(int)
    f = (1-np.log(1+temp/temp_init))**2
    for key in cols.keys():
        col = cols[key]
        std = f * (col.max() - col.min())
        x_n[key] = int(truncnorm.rvs((col.min() - x[key]) / std, (col.max() - x[key]) / std, loc=x[key], scale=std, size=1)[0]) #*temp/temp_init
    cols = s.X_df.select_dtypes(float)
    for key in cols.keys():
        col = cols[key]
        std =  f * (col.max() - col.min())
        x_n[key] = truncnorm.rvs((col.min() - x[key]) / std, (col.max() - x[key]) / std, loc=x[key], scale=std, size=1)[0]
    cols = s.X_df.select_dtypes(exclude=(int,float))
    for key in cols.keys():
        col = cols[key]
        x_n[key] = np.random.choice(col)

    return x_n




def simulated_annealing(s, temp :int = 10, beta_schedule :int = 1, MAXITER = 5, seed=42):

    np.random.seed(seed)
    temp_init = temp
    
    # generate an initial point
    current = get_candidate(s)

    # evaluate the initial point
    current_eval = objective(s,current)
    #print('initial value of surrogate optimization is %.2f' % current_eval)

    # optimize
    count = 0
    t = temp 
    while t > 1e-5 and count < MAXITER:

        # cooling 
        t = temp / (count + 1)

        # repeat
        for _ in range(beta_schedule):

            # choose a different point and evaluate
            candidate = get_neighbour(s, t, temp_init, current)
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
        
        count += 1 
    
    #print('result value of surrogate optimization is %.2f' % current_eval)
    return current, current_eval