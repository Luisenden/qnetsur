import sys
sys.path.append('../')

import pickle, time
from datetime import datetime
from functools import partial, wraps
import numpy as np
import pandas as pd
import argparse
import re

from optimizingcd import main_cd as simulation

# get the globals
parser = argparse.ArgumentParser(description="Import globals")
parser.add_argument("--topo", type=str, default='2,3', help="Network topology")
parser.add_argument("--time", type=float, default=0.05, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args, _ = parser.parse_known_args()

TOPO = args.topo
MAX_TIME = args.time
SEED = args.seed

rng_sur = np.random.default_rng(seed=SEED) # set rng for optimization 

class NetworkTopology: # use case specific topology class defined for convenience
    def __init__(self, size:tuple = None, name:str = None):
            self.size = size
            self.name = name 

vals = { # define fixed parameters for given simulation function 
        'protocol':'ndsrs', 
        'p_gen': 0.9,  # generation rate
        'p_swap': 1,  # success probability
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 1000,
        'N_samples' : 1000,
        'p_cons': 0.9/4,  # consumption rate
        'qbits_per_channel': 5,
        'cutoff': 28,
        'M': 2
        }

input_topo = TOPO.split(',') 
assert(len(input_topo) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'

topo = NetworkTopology((int(input_topo[0]), ), 'square') if len(input_topo)==1 else NetworkTopology((int(input_topo[0]), int(input_topo[1])), 'tree')
size = topo.size
vals['A'] = simulation.adjacency_squared(size[0]) if topo.name == 'square' else simulation.adjacency_tree(size[0], size[1])

vars = { # define variables and bounds for given simulation function 
    'range': {},
    'choice':{},
    'ordinal':{}
} 

for i in range(topo.size[1]+1):
    vars['range'][f'q_swap_level{i}'] = ([0., 1.], 'float')

initial_model_size = 5 # number of samples used for the initial training of the surrogate model

def simwrapper(simulation, kwargs: dict):
    q_swap = []
    for key,value in list(kwargs.items()):  # assign q_swap to level nodes
        if 'q_swap' in key:
            exp = int(re.findall('\d', key)[0])
            q_swap_level = [value]*2**exp
            q_swap += q_swap_level
            kwargs.pop(key)
        
    kwargs['q_swap'] = q_swap
    
    # run simulation and retrieve number of virtual neighbors per node
    result = simulation(**kwargs)
    mean_per_node, std_per_node = [node[-1] for node in result[1]], [np.sqrt(node[-1]) for node in result[3]]

    # get user nodes
    # user_indices = np.where(kwargs['A'].sum(axis=1) == min(kwargs['A'].sum(axis=1)))[0]

    objectives = mean_per_node
    objectives_std = std_per_node
    raw = mean_per_node
    return objectives, objectives_std, raw