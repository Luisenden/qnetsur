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
from netsquid_qswitch.aux_functions import distance_to_rate, VARDOYAN_LOSS_COEFFICIENT, VARDOYAN_LOSS_PARAMETER # beta = 0.2  [dB/km] and  c=0.1 on p. 10 of paper

#"On the stochastic analysis of quantum entanglement switch ",
# section VII 'Numerical Observations'.
# (https://arxiv.org/abs/1903.04420).

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

h = lambda p: -p * np.log2(p) - (1-p) * np.log2(1-p) if p > 0 and p < 1 else 0 # binary entropy
D_H = lambda F: 1 + F*np.log2(F) + (1-F) * np.log2((1-F)/3) if F > 0.81 else 1e-4 # yield of the so-called “hashing” protocol


rep_times = [10**-3, 10**-3] # in [s]

vals = { # define fixed parameters for given simulation function
            'nnodes': NLEAF_NODES,
            'decoherence_rate': 0,
            'connect_size': 2,
            'server_node_name': 'leaf_node_0',
            'T2': 0,
            'eta': 0.2, # link efficiency
            'loss': 1, # link efficiency
            'buffer_size': 1,
            'include_classical_comm': False,
            'num_positions': 10000,
            'N': 2 # batch size 
        }

vars = { # define variables and bounds for given simulation function
    'range': {},
    'choice':{},
    'ordinal':{}
} 
vars['range'][f'bright_state_server'] = ([0.001, .1], 'float') # in [ms]
vars['range'][f'bright_state_user'] = ([0.001, .1], 'float') # in [ms]


def simwrapper(simulation, kwargs: dict):

    bright_states = []
    for key,value in list(kwargs.items()):
        if 'bright_state' in key:
            bright_states.append(value)
            kwargs.pop(key)

    kwargs['distances'] = [kwargs['initial_distances'][0]] + [kwargs['initial_distances'][1]] * (NLEAF_NODES-1)
    kwargs['repetition_times'] = [rep_times[0]] + [rep_times[1]] * (NLEAF_NODES-1)
    kwargs['bright_state_population'] = [bright_states[0]] + [bright_states[1]] * (NLEAF_NODES-1)

    kwargs.pop('initial_distances')
    
    # run simulation
    rates, fidelities, shares_per_node = simulation(**kwargs)

    rates = np.array(rates)
    rates_per_node = ((shares_per_node.T * rates).T) * kwargs['connect_size']
    rates_per_node.columns = pd.Series(range(0,NLEAF_NODES))
    rates_per_node = rates_per_node.add_prefix('R_')


    route_rates = rates_per_node.add(rates_per_node['R_0']/kwargs['connect_size'], axis=0).drop(['R_0'], axis=1)

    # define utility

    ## Secret Key Fraction
    # werner = (4*fidelities-1)/3
    # p = (1-werner)/2
    # h_values = np.array([h(pi) for pi in p])
    # U_S = np.log((rates_per_node.T * (1-2*h_values)).T)

    # Distillable Entanglement 
    Ds = fidelities.applymap(D_H)
    U_Ds = pd.DataFrame(route_rates.values*Ds.values, columns=fidelities.columns, index=fidelities.index).applymap(np.log)


    U_D = U_Ds.mean(axis=0).values
    U_D_std = U_Ds.std(axis=0).values
    
    print("UTILITY: ", round(sum(U_D),2))
    raw = [rates.mean(),fidelities.mean()]
    print(round(raw[0],2))

    return U_D, U_D_std, raw

initial_model_size = 5 # number of samples used for initial training 