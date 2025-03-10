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
    x = {**conf.vals, **parameters}
    mean_obj, std_obj, _ = conf.simobjective(simulation=conf.sim, kwargs=x)
    mean_obj = np.nan_to_num(mean_obj, copy=True, nan=0)
    std_obj = np.nan_to_num(std_obj, copy=True, nan=0)
    return (np.sum(mean_obj), np.sum(std_obj))
          

if __name__ == '__main__':

    # load configuration
    conf = Config()
    limit = conf.args.time
    conf.set_default_values()
    path = conf.args.folder+f'AX_{conf.name}_{limit}hours_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.csv'

    objectives = dict()
    objectives["objective"] = ObjectiveProperties(minimize=False)

    ax_client = AxClient(verbose_logging=False, random_seed=conf.args.seed)
    ax_client.create_experiment( # define variable parameters for simulation function
        name=f"on-demand-protocol-seed{conf.args.seed}",
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
    df.to_csv(path)