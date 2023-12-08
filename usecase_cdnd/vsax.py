from config import *

from ax.service.ax_client import AxClient, ObjectiveProperties

def set_q_swap_for_nd(args):
    series = pd.Series(args)
    x = series[~series.index.str.contains('q_swap')] # filter all vars not containing 'q_swap'
    x = pd.concat([x, pd.Series([series[series.index.str.contains('q_swap')].values], index=['q_swap'])]) # concatenate with 'q_swap' which is now a vector
    x = x.to_dict()
    return x

def evaluate(parameters) -> float:
    parameters = set_q_swap_for_nd(args=parameters)
    x = {**parameters, **vals}
    result = simulation.simulation_cd(**x)
    mean_per_node, std_per_node = [node[-1] for node in result[1]], [node[-1] for node in result[3]]

    A = vals['A']
    user_indices = np.where(A.sum(axis=1) == min(A.sum(axis=1)))[0]
    mean_per_node = [mean_per_node[index] for index in user_indices]
    return {"mean" : (np.mean(mean_per_node), np.mean(std_per_node))}

def evaluate_multiple(parameters) -> float:
    x = {**parameters, **vals}
    result = simulation.simulation_cd(**x)
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
    
    with open(f'../../surdata/Ax_ND_{topo.name}{TOPO}_{MAX_TIME:.1f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([ax_client,time_tracker,vals], file)