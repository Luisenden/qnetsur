import functools
import numpy as np
import time

def simwrap(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        result = func(*args,**kwargs)
        return [[node[-1] for node in result[1]], [node[-1] for node in result[3]]] # mean and std according to simulation function
    return wrapper


class NetworkTopology:
    def __init__(self, size:tuple = None, name:str = None):
            self.size = size
            self.name = name

vals = { # define fixed parameters for your simulation function
        'protocol':'srs', 
        'p_gen': 0.9, 
        'p_swap':1,  
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 300,
        'N_samples' : 10,
        }

vars = { # define variables and bounds for your simulation function
        'M': [1, 10],
        'qbits_per_channel': [1,50],
        'cutoff':[1.,10.],
        'q_swap': [0., 1.],
        'p_cons':[0.01, 0.2]
        } 

nodes_to_optimize = [] # specify nodes for objective, use [] in case all nodes
initial_model_size = 10