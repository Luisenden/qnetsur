import sys
sys.path.append('../')

import pickle, time
from datetime import datetime
from functools import partial, wraps
import numpy as np
import pandas as pd
import argparse

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


vals = { # define fixed parameters for given simulation function
                'total_runtime_in_seconds': 100 * 10 ** (-6),
                'connect_size': NLEAF_NODES, 
                'rates': [1.9 * 1e6] * NLEAF_NODES,
                'buffer_size': [np.inf] * NLEAF_NODES,
                'decoherence_rate': 0,
                'T2': 10 ** (-7),
                'include_classical_comm':False,
                'N': 1
        }

vars = { # define variables and bounds for given simulation function
    'range': {
        'num_positions': ([1000, 10000],'int')
        },
    'choice':{}
} 

initial_model_size = 10 # number of samples used for the initial training of the surrogate model

def simwrap(func): 
    @wraps(func)
    def wrapper(*args):
        mean, std = func(*args)
        return mean, std
    return wrapper