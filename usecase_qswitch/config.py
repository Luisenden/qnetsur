"""
Basis script to simulate 
* optimziation of two-user-one-server scenario of Vadoyan et al., 2023 with 'sur_vardoyan_netsquid_comparison.py'
* optimization for more complex scenarios with scripts 'sur.py' and comparison with 'vsax.py', 'vsgridsearch.py', 'vssa.py'

The goal is to find the optimal bright-state population (alpha=1-Fidelity) for all links and buffer sizes to maximize
the given utility function, Distillable Entanglement as defined in the paper.
"""
import sys
sys.path.append('../')

import pickle, time
import copy
import re
from datetime import datetime
from functools import partial, wraps
import numpy as np
import pandas as pd

import argparse
import torch.multiprocessing as mp

from simulation import simulation_qswitch
from netsquid_qswitch.aux_functions import distance_to_rate, VARDOYAN_LOSS_COEFFICIENT, VARDOYAN_LOSS_PARAMETER 


# get the globals
parser = argparse.ArgumentParser(description="Import globals")
parser.add_argument("--nleaf", type=int, default=3, help="Number of leaf nodes")
parser.add_argument("--time", type=float, default=0.1, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args = parser.parse_args()

NLEAF_NODES = args.nleaf
MAX_TIME = args.time
SEED = args.seed

np.random.seed(SEED) # set seed for optimization 
initial_model_size = 5 # number of samples used for initial training 

h = lambda p: -p * np.log2(p) - (1-p) * np.log2(1-p)\
    if p > 0 and p < 1 else 0 # binary entropy
D_H = lambda F: 1 + F*np.log2(F) + (1-F) * np.log2((1-F)/3)\
    if F > 0.81 else 1e-4 # yield of the so-called “hashing” protocol

rep_times = [10 ** -3, 10 ** -3] # repetition time in [s]

 # define variables and bounds skeleton (used in executables sur.py, vsax.py etc.)
vars = {
    'range': {},
    'choice':{},
    'ordinal':{}
} 

def simwrapper(simulation, kwargs: dict):
    """
    Wraps a simulation function to adjust its parameters based on the presence of bright states in the keyword arguments,
    then runs the simulation with adjusted distances, repetition times, and bright state populations for each node. The 
    wrapper calculates rates, fidelities, and shares per node, and then computes the utility of the network based on 
    Distillable Entanglement.

    Parameters
    ----------
    simulation : function
        The simulation function to be wrapped.
    kwargs : dict
        A dictionary of keyword arguments for the simulation function. Expected to include 'initial_distances', 
        'repetition_times', and any number of 'bright_state' entries.

    Returns
    -------
    U_D : numpy.ndarray
        The mean utility of the network across all routes, calculated as the mean of the logarithm of the product of 
        route rates and fidelities, representing Distillable Entanglement.
    U_D_std : numpy.ndarray
        The standard deviation of the utility across all routes, indicating the variability in the utility.
    raw : list
        A list containing the mean of the rates and the mean of the fidelities across all simulations."""

    bright_states = []
    for key,value in list(kwargs.items()):
        if 'bright_state' in key:
            bright_states.append(value)
            kwargs.pop(key)

    buffer_size = []
    for key,value in list(kwargs.items()):
        if 'buffer' in key:
            buffer_size.append(value)
            kwargs.pop(key)

    kwargs['bright_state_population'] = [bright_states[0]] + [bright_states[1]] * (NLEAF_NODES-1)\
        if len(bright_states)==2 else bright_states
    
    kwargs['buffer_size'] = buffer_size if len(buffer_size)>1 else buffer_size[0]
    
    # run simulation
    rates, fidelities, shares_per_node = simulation(**kwargs)
    rates = np.array(rates)
    rates_per_node = ((shares_per_node.T * rates).T) * kwargs['connect_size']
    rates_per_node.columns = [''.join(re.findall(r'\d+', s)) for s in rates_per_node.columns]
    rates_per_node = rates_per_node.add_prefix('R_')
    route_rates = rates_per_node.add(rates_per_node['R_0']/
                                     kwargs['connect_size'], axis=0).drop(['R_0'], axis=1)

    # Distillable Entanglement (see definition vardoyan et al.)
    Ds = fidelities.applymap(D_H)
    U_Ds = pd.DataFrame(route_rates.values*Ds.values, columns=fidelities.columns,
                        index=fidelities.index).applymap(np.log)

    # Set NaN for nodes that are not involved
    bool_involved = np.array([any(U_Ds.columns.str.contains(str(node))) for node in range(1,NLEAF_NODES)])
    for node_not_involved in np.array(range(1,NLEAF_NODES))[~bool_involved]:
        U_Ds[f'F_leaf_node_{node_not_involved}'] = np.nan

    U_D = U_Ds.mean(axis=0).values
    U_D_std = U_Ds.std(axis=0).values

    raw = [np.mean(rates), np.mean(fidelities)]    
    return U_D, U_D_std, raw