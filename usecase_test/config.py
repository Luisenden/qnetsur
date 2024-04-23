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
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args, _ = parser.parse_known_args()
SEED = args.seed

if SEED is None:
    print(f"Warning: No global seed for optimization used. The results might not be reproduceable.")

sample_size = 10

def simwrapper(simulation, kwargs: dict):
    mean = simulation(**kwargs)
    return -mean, 0

vals = { # specify fixed parameters of quantum network simulation
        }

# specify variables and bounds of quantum network simulation
vars = { 
        'range':{},
        'choice':{}, 
        'ordinal':{}
        } 

