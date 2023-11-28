import pickle, time
from functools import partial, wraps
import numpy as np
from datetime import datetime

import sys
sys.path.append('../')

n = 9 # number of nodes
m_max = 106 # maximum number of memory qubits in a node
initial_model_size = 10 # number of samples used for the initial training of the surrogate model

def simwrap(func): # simulation wrapper: define processing of a given simulation function
    @wraps(func)
    def wrapper(*args,**kwargs):
        mean, std = func(*args,**kwargs)
        return mean-sum(kwargs.values())/(n*m_max), std # number of completed requests per node (nodes sorted alphabetically)
    return wrapper

vals = { # specify fixed parameters of quantum network simulation
        'cavity': 500, 
        'network_config_file': 'starlight.json',
        'N': 5,
        'total_time': 1e14
        }

# specify variables and bounds of quantum network simulation
vars = { 
        'range':{},
        'choice':{}
        } 

for i in range(n):
    vars['range'][f'mem_size_node_{i}'] = ([4,m_max], 'int')