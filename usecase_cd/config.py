import sys
sys.path.append('../')

import pickle, time
from datetime import datetime
from functools import partial, wraps
import numpy as np
import pandas as pd

print('importet config of cd')

def simwrap(ax=False): # simulation wrapper: define processing of a given simulation function
    def decorator(func): 
        @wraps(func)
        def wrapper(*args):
            result = func(*args)
            if ax:
                return result
            else:
                mean_per_node, std_per_node = [node[-1] for node in result[1]], [node[-1] for node in result[3]]
                return mean_per_node, std_per_node # mean and std according to simulation function
        return wrapper
    return decorator


class NetworkTopology: # use case specific topology class defined for convenience
    def __init__(self, size:tuple = None, name:str = None):
            self.size = size
            self.name = name

vals = { # define fixed parameters for given simulation function
        'protocol':'srs', 
        'p_gen': 0.9, 
        'p_swap':1,  
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 1000,
        'N_samples' : 10,
        }

vars = { # define variables and bounds for given simulation function
        'range': {
            'M': ([1, 10],'int'), 
            'qbits_per_channel': ([1,50], 'int'),
            'cutoff': ([1.,10.], 'float'),
            'q_swap': ([0., 1.], 'float'),
            'p_cons': ([0.01, 0.2],'float')
            },
        'choice':{}
        } 

initial_model_size = 10 # number of samples used for the initial training of the surrogate model