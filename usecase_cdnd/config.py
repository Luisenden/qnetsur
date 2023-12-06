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
parser.add_argument("--topo", type=str, help="Network topology")
parser.add_argument("--time", type=float, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, help="Global seed for random number generation for the optimizer")
args = parser.parse_args()

TOPO = args.topo
if TOPO is None:
    raise ValueError("Please provide a network topology using --topo argument, e.g. '11' yields 11x11 square lattice and '2,3' yields 2,3-tree network.")
MAX_TIME = args.time
if MAX_TIME is None:
    raise ValueError("Please provide a maximum number of hours (float) using --time argument.")
SEED_OPT = args.seed
if SEED_OPT is None:
    print(f"Warning: No global seed for optimization used. The results might not be reproduceable.")


def simwrap(func): 

    def set_q_swap_for_nd(args):
        l = list(args).copy()
        l.pop()
        series = pd.Series(args[1])
        x = series[~series.index.str.contains('q_swap')] # filter all vars not containing 'q_swap'
        x = pd.concat([x, pd.Series([series[series.index.str.contains('q_swap')].values], index=['q_swap'])]) # concatenate with 'q_swap' which is now a vector
        l.append(x)
        args = tuple(l)
        return args

    @wraps(func)
    def wrapper(*args):
        args = set_q_swap_for_nd(args=args) 
        result = func(*args)
        mean_per_node, std_per_node = [node[-1] for node in result[1]], [node[-1] for node in result[3]]
        return mean_per_node, std_per_node # mean and std according to simulation function
    return wrapper


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
        'total_time': 1000,
        'N_samples' : 10,
        'p_cons': 0.1,
        'qbits_per_channel': 50,
        'cutoff': 20
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
    'choice':{}
} 
for i in range(np.shape(vals['A'])[0]):
    vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')

initial_model_size = 10 # number of samples used for the initial training of the surrogate model