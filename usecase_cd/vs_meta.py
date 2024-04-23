from config import *
from src.utils import get_parameters

from ax.service.ax_client import AxClient, ObjectiveProperties

def evaluate(parameters) -> float:
    x = {**vals, **parameters}
    mean_per_user_node, std_per_user_node, _ = simwrapper(simulation=simulation.simulation_cd, kwargs=x)
    return {"evaluate" : (np.sum(mean_per_user_node), np.sqrt(np.sum(np.square(std_per_user_node))))}
          
if __name__ == '__main__':

    # user input:
    max_time= [MAX_TIME * 3600, "timer"]

    objectives = dict()
    objectives["evaluate"] = ObjectiveProperties(minimize=False)

    ax_client = AxClient(verbose_logging=False, random_seed=SEED)
    ax_client.create_experiment( # define variable parameters for simulation function
        name="continious-node-dependent-simulation-seed{SEED}",
        parameters=get_parameters(vars),
        objectives=objectives,
    )

    times_tracked = []
    time_tracker = 0
    delta = 0
    while time_tracker + delta < max_time[0]:
        start = time.time()

        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)

        df = ax_client.get_trials_data_frame()
    
    with open(f'../../surdata/cd/AX_{topo.raw}_{MAX_TIME}{max_time[1]}_objective-meanopt_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([df,time_tracker,vals], file)