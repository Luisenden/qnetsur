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

# get/set the globals
parser = argparse.ArgumentParser(description="Import globals")
parser.add_argument("--topo", type=str, default='tree-2-3', help="Network topology; \
                    Use 'tree-i-j' or 'randtree-i' or 'square-i', where i,j are integers. Type: str")
parser.add_argument("--time", type=float, default=0.05,
                    help="Maximum time allowed for optimization (in hours). Type: int")
parser.add_argument("--seed", type=int, default=42,
                    help="Global seed for random number generation for the optimizer. Type: int")
args, _ = parser.parse_known_args()

MAX_TIME = args.time
SEED = args.seed
rng_sur = np.random.default_rng(seed=SEED) # set rng for optimization 

class NetworkTopology: # use case specific topology class defined for convenience
    def __init__(self, topo_input:str):

        self.raw = topo_input    
        self.topo_input = topo_input.split('-')
        self.name = self.topo_input[0]

        if self.name == 'square':
            A = simulation.adjacency_squared(int(self.topo_input[1]))
        elif self.name == 'tree':
            A = simulation.adjacency_tree(int(self.topo_input[1]), int(self.topo_input[2]))
        elif self.name == 'randtree':
            A = simulation.adjacency_random_tree(int(self.topo_input[1]))
        else:
            raise Exception(f'Make sure topology input is set correctly! Check help via "python surrogate.py -h".')
        
        self.A = A
    
    def set_variables(self):

        vars = { # define variables and bounds for given simulation function 
            'range': {},
            'choice':{},
            'ordinal':{}
        }
        if self.name == 'tree':
            for i in range(int(self.topo_input[2])+1):
                vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')
        else:
            for i in range(int(self.topo_input[1])):
                vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')
        return vars
    
def simwrapper_levels(simulation, kwargs: dict):
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
    objectives = mean_per_node
    objectives_std = std_per_node
    raw = mean_per_node
    return objectives, objectives_std, raw

def simwrapper(simulation, kwargs: dict):
    q_swap = []
    for key,value in list(kwargs.items()): # to list as copy since kwargs is changed over the course of the loop
        if 'q_swap' in key:
            q_swap.append(value)
            kwargs.pop(key)
        
    kwargs['q_swap'] = q_swap
    
    # run simulation and retrieve number of virtual neighbors per node
    result = simulation(**kwargs)
    mean_per_node = np.array([node[-1] for node in result[1]])
    std_per_node = np.array([np.sqrt(node[-1]) for node in result[3]])

    # get user nodes
    user_indices = np.where(kwargs['A'].sum(axis=1) == 1)
    objectives = mean_per_node[user_indices]
    objectives_std = std_per_node[user_indices]
    raw = mean_per_node
    return objectives, objectives_std, raw

topo = NetworkTopology(topo_input=args.topo)
vals = { # define fixed parameters for given simulation function 
        'protocol':'ndsrs', 
        'p_gen': 0.9,  # generation rate
        'p_swap': 1,  # success probability
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 1000,
        'N_samples' : 20,
        'p_cons': 0.9/4,  # consumption rate
        'qbits_per_channel': 5,
        'cutoff': 28,
        'M': 10,
        }
vals['A'] = topo.A
vars = topo.set_variables()
initial_model_size = 5 # number of samples used for the initial training of the surrogate model