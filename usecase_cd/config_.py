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
parser.add_argument("--nnodes", type=int, default=3, help="Network topology")
parser.add_argument("--time", type=float, default=0.05, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args, _ = parser.parse_known_args()

NNODES = args.nnodes
MAX_TIME = args.time
SEED = args.seed
folder = '../../surdata/cd/'

rng_sur = np.random.default_rng(seed=SEED) # set rng for optimization 

vals = { # define fixed parameters for given simulation function 
        'protocol':'ndsrs', 
        'p_gen': 0.9,  # generation rate
        'p_swap': 1,  # success probability
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 1000,
        'N_samples' : 1,
        'p_cons': 0.9/4,  # consumption rate
        'qbits_per_channel': 5,
        'cutoff': 28,
        'M': 2
        }

vals['A'] = simulation.adjacency_random_tree(NNODES, folder=folder)


vars = { # define variables and bounds for given simulation function 
    'range': {},
    'choice':{},
    'ordinal':{}
} 

for i in range(NNODES):
    vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')

initial_model_size = 5 # number of samples used for the initial training of the surrogate model

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
    user_indices = np.where(vals['A'].sum(axis=1) == 1)

    objectives = mean_per_node[user_indices]
    objectives_std = std_per_node[user_indices]
    raw = mean_per_node
    return objectives, objectives_std, raw