import pickle, time
import pandas as pd
from functools import partial, wraps
import numpy as np
from datetime import datetime

import sys
sys.path.append('../')

n = 9 # number of nodes
m_max = 110 # maximum number of memory qubits in a node
initial_model_size = 10 # number of samples used for the initial training of the surrogate model


def simwrap(func): # simulation wrapper: define processing of a given simulation function
    @wraps(func)
    def wrapper(*args):
        mems = pd.Series(args[-1]).values
        mean, std = func(*args)
        mean_per_node = mean-sum(mems)/(n*m_max)
        return mean_per_node, std # number of completed requests per node (nodes sorted alphabetically)
    return wrapper


vals = { # specify fixed parameters of quantum network simulation
        'cavity': 500, 
        'network_config_file': 'starlight.json',
        'N': 1,
        'total_time': 2e13
        }

# specify variables and bounds of quantum network simulation
vars = { 
        'range':{},
        'choice':{}
        } 

for i in range(n):
    vars['range'][f'mem_size_node_{i}'] = ([5,m_max], 'int')