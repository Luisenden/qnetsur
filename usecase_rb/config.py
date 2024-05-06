import numpy as np
import argparse
import sys
sys.path.append('../')

# get the globals
parser = argparse.ArgumentParser(description="Import globals")
parser.add_argument("--time", type=float, default=0.1, help="Maximum time allowed for optimization (in hours)")
parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
args, _ = parser.parse_known_args()
MAX_TIME = args.time
SEED = args.seed


rng_sur = np.random.default_rng(seed=SEED) # set rng for surrogate optimization 
nnodes = 9 # number of nodes
m_max = 105 # maximum number of memory qubits per node
sample_size = 10 # number of samples used for the initial training of the surrogate model
budget = 450

def simwrapper(simulation, kwargs: dict):
    mem_size = []
    for key, value in list(kwargs.items()):
        if 'size' in key:
            mem_size.append(value)
            kwargs.pop(key)
    kwargs['mem_size'] = mem_size

    mean, std = simulation(**kwargs)
    kwargs['mem_size'] = np.array(mem_size)
    objectives = mean - max(np.sum(mem_size)-budget, 0)
    objectives_std = std
    raw = mean
    return objectives, objectives_std, raw

# specify fixed parameters of quantum network simulation
vals = {
        'network_config_file': 'starlight.json', # network configuration file
        'N': 5, # simulation sample size
        'total_time': 2e13, # simulation time
        'cavity': 500 # experimental parameter for atom-cavity cooperativity
        }

# specify variables and bounds of quantum network simulation
vars = {
        'range':{},
        'choice':{},
        'ordinal':{}
        } 

for i in range(nnodes):
    vars['range'][f'mem_size_node_{i}'] = ([10, m_max], 'int')