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
parser.add_argument("--time", type=float, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, help="Global seed for random number generation for the optimizer")
args = parser.parse_args()

MAX_TIME = args.time
if MAX_TIME is None:
    raise ValueError("Please provide a maximum number of hours (float) using --time argument.")
SEED_OPT = args.seed
if SEED_OPT is None:
    print(f"Warning: No global seed for optimization used. The results might not be reproduceable.")

sample_size = 10

def simwrap(func): # simulation wrapper: define processing of a given simulation function
    @wraps(func)
    def wrapper(*args):
        mean, std = func(*args)
        return mean, std # number of completed requests per node (nodes sorted alphabetically)
    return wrapper


vals = { # specify fixed parameters of quantum network simulation
        }

# specify variables and bounds of quantum network simulation
vars = { 
        'range':{},
        'choice':{}
        } 

for i in range(100):
    vars['range'][f'x{i}'] = ([-500,500], 'int')