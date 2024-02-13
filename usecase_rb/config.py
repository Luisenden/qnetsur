import pickle, time
import pandas as pd
from functools import partial, wraps
import numpy as np
from datetime import datetime
import argparse
import sys
sys.path.append('../')

# get the globals
parser = argparse.ArgumentParser(description="Import globals")
parser.add_argument("--time", type=float, default=1, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args = parser.parse_args()
MAX_TIME = args.time
SEED = args.seed

np.random.seed(SEED) # set seed for optimization 
nnodes = 9 # number of nodes
m_max = 110 # maximum number of memory qubits per node
sample_size = 5 # number of samples used for the initial training of the surrogate model

def simwrapper(simulation, kwargs: dict):
    mem_size = []
    for key, value in list(kwargs.items()):
        if 'size' in key:
            mem_size.append(value)
            kwargs.pop(key)
    kwargs['mem_size'] = mem_size
    mean, std = simulation(**kwargs)
    kwargs['mem_size'] = np.array(mem_size)
    objectives = mean-np.array(mem_size)/m_max
    objectives_std = std
    return objectives, objectives_std

# specify fixed parameters of quantum network simulation
vals = {
        'cavity': 500,
        'network_config_file': 'starlight.json',
        'N': 1,
        'total_time': 2e13
        }

# specify variables and bounds of quantum network simulation
vars = {
        'range':{},
        'choice':{},
        'ordinal':{}
        } 

for i in range(nnodes):
    vars['range'][f'mem_size_node_{i}'] = ([5, m_max], 'int')