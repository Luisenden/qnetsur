import sys
sys.path.append('../')

import pickle, time
from datetime import datetime
from functools import partial, wraps
import numpy as np
import pandas as pd
import argparse

from optimizingcd import main_cd as simulation

# get the globals
parser = argparse.ArgumentParser(description="Import globals")
parser.add_argument("--topo", type=str, default='2,3', help="Network topology")
parser.add_argument("--time", type=float, default=0.05, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args = parser.parse_args()

TOPO = args.topo
MAX_TIME = args.time
SEED = args.seed

np.random.seed(SEED) # set seed for optimization 

class NetworkTopology: # use case specific topology class defined for convenience
    def __init__(self, size:tuple = None, name:str = None):
            self.size = size
            self.name = name 

vals = { # define fixed parameters for given simulation function 
        'protocol':'ndsrs', 
        'p_gen': 0.9, 
        'p_swap': 1,
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 100,
        'N_samples' : 100,
        'p_cons': 0.1,
        'qbits_per_channel': 50,
        'cutoff': 20,
        }

input_topo = TOPO.split(',') 
assert(len(input_topo) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'

topo = NetworkTopology((int(input_topo[0]), ), 'square') if len(input_topo)==1 else NetworkTopology((int(input_topo[0]), int(input_topo[1])), 'tree')
size = topo.size
vals['A'] = simulation.adjacency_squared(size[0]) if topo.name == 'square' else simulation.adjacency_tree(size[0], size[1])

vars = { # define variables and bounds for given simulation function 
    'range': {
        'M': ([1, 10],'int')
        },
    'choice':{},
    'ordinal':{}
} 
for i in range(np.shape(vals['A'])[0]):
    vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')

initial_model_size = 5 # number of samples used for the initial training of the surrogate model

def simwrapper(simulation, kwargs: dict):
     
    q_swap = []
    for key,value in list(kwargs.items()):
        if 'q_swap' in key:
            q_swap.append(value)
            kwargs.pop(key)
    kwargs['q_swap'] = q_swap
    # run simulation
    result = simulation(**kwargs)
    mean_per_node, std_per_node = [node for node in result[1]][-1], [node for node in result[3]][-1] 
    # get user nodes
    user_indices = np.where(kwargs['A'].sum(axis=1) == min(kwargs['A'].sum(axis=1)))[0]
    objectives = [mean_per_node[index] for index in user_indices]
    objectives_std = [std_per_node[index] for index in user_indices]
    raw = mean_per_node
    return objectives, objectives_std, raw