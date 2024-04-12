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
parser.add_argument("--time", type=float, default=0.1, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args, _ = parser.parse_known_args()
MAX_TIME = args.time
SEED = args.seed


rng_sur = np.random.default_rng(seed=SEED) # set rng for optimization 
nnodes = 9 # number of nodes
m_max = 105 # maximum number of memory qubits per node
sample_size = 5 # number of samples used for the initial training of the surrogate model

def simwrapper(simulation, kwargs: dict):
    mem_size = []
    for key, value in list(kwargs.items()):
        if 'size' in key:
            mem_size.append(value)
            kwargs.pop(key)
    kwargs['mem_size'] = mem_size

    
    #slackbudget = kwargs.pop('slackbudget')
    slackbudget = 0

    mean, std = simulation(**kwargs)
    kwargs['mem_size'] = np.array(mem_size)
    objectives = mean - (np.sum(mem_size) - slackbudget)**2/len(mem_size)
    objectives_std = std
    raw = mean
    return objectives, objectives_std, raw

# specify fixed parameters of quantum network simulation
vals = {
        'cavity': 500,
        'network_config_file': 'starlight.json',
        'N': 5,
        'total_time': 2e13
        }

# specify variables and bounds of quantum network simulation
vars = {
        'range':{},
        'choice':{},
        'ordinal':{}
        } 

vars['range']['slackbudget'] = ([225, 450], 'int')
for i in range(nnodes):
    vars['range'][f'mem_size_node_{i}'] = ([25, m_max], 'int')