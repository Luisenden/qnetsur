import numpy as np
from functools import partial
from datetime import datetime

import pickle
import sys 
from ax.service.ax_client import AxClient, ObjectiveProperties

from rb_simulation import *

def evaluate(parameters) -> float:
    x = {**parameters, **vals}
    mean_per_node, std_per_node = simulation_rb(**x)
    mean_all_nodes, std_all_nodes = np.mean(mean_per_node), np.mean(std_per_node)
    res = {"mean" : (mean_all_nodes,std_all_nodes)}
    return res

def evaluate_multiple(parameters) -> float:
    x = {**parameters, **vals}
    result = simulation_rb(**x)
    res = dict()
    i = 0
    for mean, std in zip(result[1], result[3]):
        res[f"n{i}"] = (mean[-1],std[-1])
        i += 1
    return res


if __name__ == '__main__':

    # user input: number of maximum iterations optimiztion
    MAXITER = int(sys.argv[1]) 

    # user input: number of trials
    ntrials = int(sys.argv[2]) 

    objectives = dict()
    objectives["mean"] = ObjectiveProperties(minimize=False)

    # specify fixed parameters of quantum network simulation
    vals = { 
        'cavity': 500, 
        'network_config_file': 'starlight.json',
        'N': 1,
        'total_time': 1e11
        }

    total_time = []
    ax_clients = []
    for _ in range(ntrials):
        ax_client = AxClient(verbose_logging=False)
        ax_client.create_experiment( # define variable parameters of quantum network simulation
            name="simulation_test_experiment",
            parameters=[
                {
                    "name": f'mem_size_node_{i}',
                    "type": "range",
                    "bounds": [4, 50],
                } for i in range(9)
            ],
            objectives=objectives,
        )

        start = time()
        raw_data_vec = []
        for i in range(MAXITER):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
        total_time.append(time()-start)

        ax_clients.append(ax_client)

    
    with open('../../surdata/Ax_starlight_iter-'+str(MAXITER)+'_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M")+'.pkl', 'wb') as file:
            pickle.dump([ax_clients,total_time,vals], file)