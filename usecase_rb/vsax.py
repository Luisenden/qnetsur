from config import *
from ax.service.ax_client import AxClient, ObjectiveProperties

from simulation import *

@simwrap(ax=True)
def evaluate(parameters) -> float:
    x = {**parameters, **vals}
    mean_all_nodes, std_all_nodes = simulation_rb(**x)
    return mean_all_nodes, std_all_nodes

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

    start = time.time()
    # user input: number of maximum iterations optimiztion
    MAXITER = int(sys.argv[1]) 

    # user input: number of trials
    ntrials = int(sys.argv[2]) 

    objectives = dict()
    objectives["mean"] = ObjectiveProperties(minimize=False)

    total_time = []
    ax_clients = []
    for _ in range(ntrials):
        ax_client = AxClient(verbose_logging=False)
        ax_client.create_experiment( # define variable parameters of quantum network simulation
            name="simulation_test_experiment",
            parameters=get_parameters(vars),
            objectives=objectives,
        )

        start = time.time()
        raw_data_vec = []
        for i in range(MAXITER):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
        total_time.append(time.time()-start)

        ax_clients.append(ax_client)

    
    with open('../../surdata/Ax_starlight_iter-'+str(MAXITER)+'_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M")+'.pkl', 'wb') as file:
            pickle.dump([ax_clients,total_time,vals], file)
    
    print('time:', time.time()-start)