import numpy as np
from functools import partial
from optimizingcd import main_cd as simulation
import time
from datetime import datetime

import pickle
import sys 
from ax.service.ax_client import AxClient, ObjectiveProperties

class NetworkTopology:
    def __init__(self, size:tuple = None, name:str = None):
            self.size = size
            self.name = name

def evaluate(parameters) -> float:
    x = {**parameters, **vals}
    result = simulation.simulation_cd(**x)
    res = dict()
    i = 0
    for mean, std in zip(result[1], result[3]):
        res[f"n{i}"] = (mean[-1],std[-1])
        i += 1
    return res


if __name__ == '__main__':

    # user input: network topology
    vv = sys.argv[1]
    v = vv.split(',') 

    # user input: number of maximum iterations optimiztion
    MAXITER = int(sys.argv[2]) 

    assert(len(v) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'
    topo = NetworkTopology((int(v[0]), ), 'square') if len(v)==1 else NetworkTopology((int(v[0]), int(v[1])), 'tree')
    size = topo.size

    vals = { # define fixed parameters for your simulation function
        'A': simulation.adjacency_squared(size[0]) if topo.name == 'square' else simulation.adjacency_tree(size[0], size[1]),
        'protocol':'srs', 
        'p_gen': 0.9, 
        'p_swap':1,  
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 300,
        'N_samples' : 10,
        } 

    objectives = dict()
    for i in range(vals['A'].shape[0]):
        objectives[f"n{i}"] = ObjectiveProperties(minimize=False)

    ax_client = AxClient()
    ax_client.create_experiment(
        name="simulation_test_experiment",
        parameters=[
            {
                "name": "M",
                "type": "range",
                "bounds": [0, 10],
            },
            {
                "name": "qbits_per_channel",
                "type": "range",
                "bounds": [1, 50],
            },
            {
                "name": "cutoff",
                "type": "range",
                "bounds": [1., 10.],
            },
            {
                "name": "q_swap",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "p_cons",
                "type": "range",
                "bounds": [0.01, 0.2],
            },
        ],
        objectives=objectives,
    )

    start = time.time()
    raw_data_vec = []
    for i in range(MAXITER):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
    total_time = time.time()-start

    
    with open('../surdata/Ax_'+topo.name+vv.replace(',','')+'_iter-'+str(MAXITER)+'_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M")+'.pkl', 'wb') as file:
            pickle.dump([ax_client,total_time,vals], file)