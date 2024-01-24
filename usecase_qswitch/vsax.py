from config import *
from src.utils import get_parameters
from ax.service.ax_client import AxClient

def evaluate(parameters) -> float:
    x = {**vals, **parameters}
    mean_obj, std_obj, _ = simwrapper(simulation=simulation_qswitch, kwargs=x)
    return (np.sum(mean_obj), np.sum(std_obj))
          

if __name__ == '__main__':

    max_time= MAX_TIME * 3600 # in sec

    ax_client = AxClient(verbose_logging=False, random_seed=SEED_OPT)
    ax_client.create_experiment( # define variable parameters for simulation function
        name="qswitch-simulation-seed{SEED_OPT}",
        parameters=get_parameters(vars),
        minimize=False,
        objective_name="evaluate",
    )

    times_tracked = []
    time_tracker = 0
    delta = 0
    while time_tracker + delta < max_time:
        start = time.time()

        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    result = ax_client.get_trials_data_frame()
    best_parameters, metrics = ax_client.get_best_parameters()
    with open(f'../../surdata/qswitch/Ax_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-servernode_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([result,time_tracker,vals], file)
    
    print(result)
    print(best_parameters, metrics)