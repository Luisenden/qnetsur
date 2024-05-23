import numpy as np
import time
from datetime import datetime
import pickle


from config import Config
from src.utils import get_parameters

from optimizingcd import main_cd as simulation
from ax.service.ax_client import AxClient, ObjectiveProperties

def evaluate(parameters) -> float:
    x = {**conf.vals, **parameters}
    mean_per_user_node, std_per_user_node, _ = conf.simobjective(simulation=simulation.simulation_cd, kwargs=x)
    return {"evaluate" : (np.sum(mean_per_user_node), np.sqrt(np.sum(np.square(std_per_user_node))))}
          
if __name__ == '__main__':

    # load configuration
    conf = Config(initial_model_size=5)
    limit = conf.args.time

    objectives = dict()
    objectives["evaluate"] = ObjectiveProperties(minimize=False)

    ax_client = AxClient(verbose_logging=False, random_seed=conf.args.seed)
    ax_client.create_experiment( # define variable parameters for simulation function
        name=f"continious-node-dependent-simulation-seed{conf.args.seed}",
        parameters=get_parameters(conf.vars),
        objectives=objectives,
    )

    times_tracked = []
    time_tracker = 0
    delta = 0
    while time_tracker + delta < limit * 3600:
        start = time.time()

        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)

        df = ax_client.get_trials_data_frame()

    limit_kind = 'hours' if isinstance(limit, float) else 'cycles'
    with open(conf.args.folder+f'AX_{conf.name}_{limit}{limit_kind}_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([df,time_tracker,conf.vals], file)