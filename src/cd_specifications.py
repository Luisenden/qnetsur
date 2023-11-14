import functools

def simwrap(func): # simulation wrapper: any postprocessing of the simulation function
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        result = func(*args,**kwargs)
        return [[node[-1] for node in result[1]], [node[-1] for node in result[3]]] # mean and std according to simulation function
    return wrapper


class NetworkTopology:
    def __init__(self, size:tuple = None, name:str = None):
            self.size = size
            self.name = name

vals = { # define fixed parameters for your simulation function
        'protocol':'srs', 
        'p_gen': 0.9, 
        'p_swap':1,  
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 1000,
        'N_samples' : 10,
        }

vars = { # define variables and bounds for your simulation function
        'range': {
            'M': ([1, 10],'int'), 
            'qbits_per_channel': ([1,50], 'int'),
            'cutoff': ([1.,10.], 'float'),
            'q_swap': ([0., 1.], 'float'),
            'p_cons': ([0.01, 0.2],'float')
            },
        'choice':{}
        } 

initial_model_size = 10 # number of samples used for the initial training of the surrogate model