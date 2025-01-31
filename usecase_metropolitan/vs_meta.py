"""
Optimize using BoTorch service loop according to best practices (see https://ax.dev/tutorials/gpei_hartmann_service.html)
"""

import numpy as np
import time
from datetime import datetime

from config import Config
from qnetsur.datacollector import get_parameters
from ax.service.ax_client import AxClient, ObjectiveProperties

def evaluate(parameters) -> float:
    """
    Evaluate the simulation objective function using the given parameters.

    Parameters:
    parameters (dict): A dictionary of parameter values for the simulation.

    Returns:
    dict: A dictionary containing the objective value and its standard deviation.
    """
    # Combine fixed configuration values with dynamic parameters
    x = {**conf.vals, **parameters}
    # Evaluate the simulation objective function
    mean_per_user_node, std_per_user_node, _ = conf.simobjective(simulation=conf.sim, kwargs=x)
    return {"objective" : (np.sum(mean_per_user_node), np.sqrt(np.sum(np.square(std_per_user_node))))}
          
if __name__ == '__main__':

    # Load configuration
    conf = Config()
    limit = conf.args.time
    # Define the path for saving the results
    path = conf.args.folder+f'AX_{conf.name}_{limit}hours_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.csv'

    # Define the objective properties
    objectives = dict()
    objectives["objective"] = ObjectiveProperties(minimize=False)

    # Initialize the AxClient 
    ax_client = AxClient(verbose_logging=False, random_seed=conf.args.seed)
    ax_client.create_experiment( # define variable parameters for simulation function
        name=f"on-demand-metropolitan-simulation-seed{conf.args.seed}",
        parameters=get_parameters(conf.vars),
        objectives=objectives,
    )
    print('MAX PARALLEL: ', ax_client.get_max_parallelism())

    times_tracked = []
    time_tracker = 0
    delta = 0
    # Run the optimization loop until the time limit is reached
    while time_tracker + delta < limit * 3600:
        start = time.time()
        # Get the next set of parameters and trial index
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        # Track the elapsed time for each trial
        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)

    # Save the results to a DataFrame
    df = ax_client.get_trials_data_frame()
    df.to_csv(path)
    
    # Uncomment the line below to print the maximum number of parallel trials
    # print(ax_client.get_max_parallelism()) 