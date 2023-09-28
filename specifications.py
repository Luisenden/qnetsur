import functools

def postprocess(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        result = func(*args,**kwargs)
        return [node[-1] for node in result[1]]
    return wrapper

vals = {
        'protocol':'srs', 
        'p_cons': 0.1, 
        'p_gen': 0.9, 
        'p_swap':1,  
        'return_data':'avg', 
        'progress_bar': None,
        'cutoff': 20,
        'total_time': 500,
        'N_samples' : 10,
        }

vars = {
        'M': [1, 10],
        'qbits_per_channel': [3,100],
        'q_swap': [0., 1.],
        } 