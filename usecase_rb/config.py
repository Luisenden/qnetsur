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
if MAX_TIME is None:
    raise ValueError("Please provide a maximum number of hours (float) using --time argument.")
SEED_OPT = args.seed
if SEED_OPT is None:
    print(f"Warning: No global seed for optimization used. The results might not be reproduceable.")

nnodes = 9 # number of nodes
m_max = 110 # maximum number of memory qubits per node
sample_size = 5 # number of samples used for the initial training of the surrogate model


def simwrap(func): # simulation wrapper: define processing of a given simulation function
    @wraps(func)
    def wrapper(*args):
        vars_temp = pd.Series(args[-1])
        mems = vars_temp[vars_temp.index.str.contains('size')].values
        mean, std = func(*args)
        wrapped = mean-mems/m_max
        return wrapped, std, mean # number of completed requests per node (nodes sorted alphabetically)
    return wrapper


vals = { # specify fixed parameters of quantum network simulation
        'cavity': 500, 
        'network_config_file': 'starlight.json',
        'N': 5,
        'total_time': 2e13
        }

# specify variables and bounds of quantum network simulation
vars = { 
        'range':{},
        'choice':{}
        } 

for i in range(nnodes):
    vars['range'][f'mem_size_node_{i}'] = ([5,m_max], 'int')