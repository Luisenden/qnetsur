import sys
sys.path.append('../')

import pickle, time
import copy
from datetime import datetime
from functools import partial, wraps
import numpy as np
import pandas as pd

import argparse
import torch.multiprocessing as mp

from simulation import simulation_qswitch

# get the globals
parser = argparse.ArgumentParser(description="Import globals")
parser.add_argument("--nleaf", type=int, default=10, help="Number of leaf nodes")
parser.add_argument("--time", type=float, default=1, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args = parser.parse_args()

NLEAF_NODES = args.nleaf
MAX_TIME = args.time
SEED_OPT = args.seed

np.random.seed(SEED_OPT) # set seed for optimization 

def simwrapper(simulation, kwargs: dict):

    buffer_size = []
    for key,value in list(kwargs.items()):
        if 'buffer_node' in key:
            buffer_size.append(value)
            kwargs.pop(key)

    buffer_size = np.array(buffer_size)
    print(f'{mp.current_process().name} buffer initial: {buffer_size}')
    buffer_size = buffer_size[buffer_size.nonzero()]

    surplus = kwargs['connect_size'] - len(buffer_size) # if length of buffer size is n smaller than connect size, then add n nodes with smallest buffer size
    if surplus > 0: 
        print(f'{mp.current_process().name} adds {surplus} nodes with buffer size 1') 
        buffer_size = np.append(buffer_size,[1] * surplus)

    kwargs['buffer_size'] = buffer_size
    print(f'{mp.current_process().name} buffer actual: {buffer_size}')
    
    assert kwargs['nnodes'] - len(buffer_size) >= 0, "Too few nodes specified."
    
    spfl = kwargs['nnodes'] - len(buffer_size) # remove superfluous rates 
    print(f'{mp.current_process().name} thus removes {spfl} rate entries')
    kwargs['rates'] = kwargs['rates_initial'][:-spfl] if spfl > 0 else kwargs['rates_initial']
    kwargs.pop('rates_initial')

    kwargs['nnodes'] = len(buffer_size) # number of nodes is reduced to the buffer size 

    # run simulation
    print('PARAMETERS: ' , kwargs)
    states_per_node, capacities = simulation(**kwargs)
    capacities = np.array(capacities) / 1e6 # MHz

    print('STATES: ', states_per_node)

    # define objectives
    share_of_server_node = kwargs['connect_size'] * (1 + states_per_node['leaf_node_0'] - np.sum(states_per_node.drop(['leaf_node_0'], axis=1), axis=1))
    
    obj1 = share_of_server_node
    obj2 = 0.1 * np.array(capacities) 

    mean_obj1 = np.mean(obj1)
    mean_obj2 = np.mean(obj2)

    std_obj1 = np.std(obj1)
    std_obj2 = np.std(obj2)

    objectives = [mean_obj1, mean_obj2]
    objectives_std = [std_obj1, std_obj2]

    print('OBJECTIVES', objectives)
    raw = [np.mean(share_of_server_node), np.mean(capacities)]
    return objectives, objectives_std, raw


vals = { # define fixed parameters for given simulation function
            'nnodes': NLEAF_NODES,
            'total_runtime_in_seconds': 10 ** -4, # simulation time of 1ms
            'decoherence_rate': 0,
            'connect_size': 5,
            'T2': 10 ** (-7),
            'include_classical_comm': False,
            'rates_initial': [2 * 10 ** 6] * NLEAF_NODES, 
            'num_positions': 1000,
            'buffer_node0': 1024,
            'N': 10
        }

vars = { # define variables and bounds for given simulation function
    'range': {},
    'choice':{},
    'ordinal':{}
} 
for i in range(1,NLEAF_NODES):
    vars['ordinal'][f'buffer_node{i}'] = [0]+[2 ** i for i in range(11)]

initial_model_size = 5 # number of samples used for initial training 