import functools
import numpy as np

def simwrap(func): # simulation wrapper: any postprocessing of the simulation function
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        mean, std = func(*args,**kwargs)
        return mean-sum(kwargs.values())/450, std # number of completed requests per node (nodes sorted alphabetically)
    return wrapper

vals = { # specify fixed parameters of quantum network simulation
        'cavity': 500, 
        'network_config_file': 'starlight.json',
        'N': 10,
        'total_time': 1e14
        }


vars = { # specify variables and bounds of quantum network simulation
        'range':{},
        'choice':{}
        } 

for i in range(9):
    vars['range'][f'mem_size_node_{i}'] = ([4,50], 'int')


initial_model_size = 10 # number of samples used for the initial training of the surrogate model