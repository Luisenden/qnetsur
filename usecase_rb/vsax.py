from config import *
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.scheduler import SchedulerOptions

from simulation import *


def evaluate(parameters) -> float:
    x = {**parameters, **vals}
    mean_per_node, std_per_node = simulation_rb(**x) 
    return np.mean(mean_per_node), np.mean(std_per_node)

def evaluate_multiple(parameters) -> float:
    x = {**parameters, **vals}
    result = simulation_rb(**x)
    res = dict()
    i = 0
    for mean, std in zip(result[1], result[3]):
        res[f"n{i}"] = (mean[-1],std[-1])
        i += 1
    return res

def get_parameters(vars:dict):
    parameters = []
    for k in vars:
        for key,value in vars[k].items():
            parameters.append(
                {
                "name": str(key),
                "type": k,
                "bounds": value[0],
                }
            )
    return parameters


if __name__ == '__main__':

    # user input:
    max_time = MAX_TIME * 3600 # in sec

    objectives = dict()
    objectives["mean"] = ObjectiveProperties(minimize=False)

    ax_client = AxClient(verbose_logging=False, random_seed=SEED_OPT)
    ax_client.create_experiment( # define variable parameters of quantum network simulation
        name=f"request-based-simulation-seed{SEED_OPT}",
        parameters=get_parameters(vars),
        objectives=objectives,
    )


    times_tracked = []
    time_tracker = 0
    while time_tracker < max_time:
        start = time.time()

        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        times_tracked.append(time.time()-start)
        time_tracker = sum(times_tracked)
    
    with open(f'../../surdata/Ax_starlight_{MAX_TIME:.0f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([ax_client,time_tracker,vals], file)