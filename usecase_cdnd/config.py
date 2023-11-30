import sys
sys.path.append('../')

import pickle, time
from datetime import datetime
from functools import partial, wraps
import numpy as np
import pandas as pd


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

initial_model_size = 10 # number of samples used for the initial training of the surrogate model