import numpy as np
from functools import partial
from optimizingcd import main_cd as simulation
import time
from datetime import datetime

import pickle
import sys 
from ax.service.ax_client import AxClient, ObjectiveProperties

from cd_specifications import *

def evaluate(parameters) -> float:
    x = {**parameters, **vals}
    result = simulation.simulation_cd(**x)
    mean_all_nodes, std_all_nodes = np.mean([node[-1] for node in result[1]]), np.mean([node[-1] for node in result[3]])
    res = {"mean" : (mean_all_nodes,std_all_nodes)}
    return res

def evaluate_multiple(parameters) -> float:
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

    # user input: number of maximum iterations optimiztion
    MAXITER = int(sys.argv[2]) 

    # user input: number of trials
    ntrials = int(sys.argv[3]) 

    # instantiate network topology
    v = vv.split(',')
    assert(len(v) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'
    topo = NetworkTopology((int(v[0]), ), 'square') if len(v)==1 else NetworkTopology((int(v[0]), int(v[1])), 'tree')
    size = topo.size

    vals = { # define fixed parameters for simulation function
        'A': simulation.adjacency_squared(size[0]) if topo.name == 'square' else simulation.adjacency_tree(size[0], size[1]),
        'protocol':'srs', 
        'p_gen': 0.9, 
        'p_swap':1,  
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 1000,
        'N_samples' : 10,
        } 

    objectives = dict()
    objectives["mean"] = ObjectiveProperties(minimize=False)

    total_time = []
    ax_clients = []
    for _ in range(ntrials):
        ax_client = AxClient(verbose_logging=False)
        ax_client.create_experiment( # define variable parameters for simulation function
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
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
        total_time.append(time.time()-start)

        ax_clients.append(ax_client)

    
    with open('../../surdata/Ax_'+topo.name+vv.replace(',','')+'_iter-'+str(MAXITER)+'_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M")+'.pkl', 'wb') as file:
            pickle.dump([ax_clients,total_time,vals], file)