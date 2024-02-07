from config import *
from src.utils import get_parameters
from ax.service.ax_client import AxClient, ObjectiveProperties

from simulation import *


def evaluate(parameters) -> float:
    x = {**vals, **parameters}
    mean_obj, std_obj= simwrapper(simulation=simulation_rb, kwargs=x)
    return (np.sum(mean_obj), np.sum(std_obj))


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
    delta = 0
    while time_tracker + delta < max_time:
        start = time.time()

        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    with open(f'../../surdata/rb/AX_starlight_{MAX_TIME:.1f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([ax_client,time_tracker,vals], file)