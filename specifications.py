import functools
import numpy as np
import time

def simwrap(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        result = func(*args,**kwargs)
        return [[node[-1] for node in result[1]], [node[-1] for node in result[3]]] # mean and std according to simulation function
    return wrapper

def objective(s, X) -> tuple:

    start = time.time()
    y = s.mmodel.predict(X)
    s.predict_time.append(time.time()-start)
    # print('predict ', s.predict_time)

    start = time.time()
    y = np.array(y)
    y_mean = y.mean(axis=1)
    y_mean = np.array(y_mean)
    index = y_mean.argmax()
    s.findmax_time.append(time.time()-start)

    return X[index]

class NetworkTopology:
    def __init__(self, size: tuple, name: str):
            self.size = size
            self.name = name

vals = { # define fixed parameters for your simulation function
        'protocol':'srs', 
        'p_cons': 0.1, 
        'p_gen': 0.9, 
        'p_swap':1,  
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 1000,
        'N_samples' : 10,
        }

vars = { # define variables and bounds for your simulation function
        'M': [1, 10],
        'qbits_per_channel': [1,50],
        'cutoff':[1.,30.],
        'q_swap': [0., 1.],
        'p_cons':[0.01, 0.2]
        } 

nodes_to_optimize = [] # specify nodes for objective, use [] in case all nodes
initial_model_size = 30