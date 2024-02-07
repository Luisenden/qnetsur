from config import *
from src.utils import get_parameters

from ax.service.ax_client import AxClient, ObjectiveProperties

def evaluate(parameters) -> float:
    x = {**vals, **parameters}
    mean_per_user_node, std_per_user_node, _ = simwrapper(simulation=simulation.simulation_cd, kwargs=x)
    return {"mean" : (np.mean(mean_per_user_node), np.mean(std_per_user_node))}
          
if __name__ == '__main__':

    # user input:
    max_time= MAX_TIME * 3600 # in sec

    objectives = dict()
    objectives["mean"] = ObjectiveProperties(minimize=False)

    ax_client = AxClient(verbose_logging=False, random_seed=SEED_OPT)
    ax_client.create_experiment( # define variable parameters for simulation function
        name="continious-based-node-dependent-simulation-seed{SEED_OPT}",
        parameters=get_parameters(vars),
        objectives=objectives,
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
    
    with open(f'../../surdata/cd/AX_{topo.name}{TOPO}_{MAX_TIME:.2f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([ax_client,time_tracker,vals], file)