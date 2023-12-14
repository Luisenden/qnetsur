from config import *

from ax.service.ax_client import AxClient, ObjectiveProperties



def evaluate(parameters) -> float:
    x = {**parameters, **vals}
    result = simulation_qswitch(**x)
    mean_cap_fid, std_cap_fid = result
    return {"mean" : (np.mean(mean_cap_fid), np.mean(std_cap_fid))}

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


    ax_client = AxClient(verbose_logging=False, random_seed=SEED_OPT)
    ax_client.create_experiment( # define variable parameters for simulation function
        name="qswitch-simulation-seed{SEED_OPT}",
        parameters=[{
            "name": "num_positions",
            "type": "range",
            "bounds": [500, 1000],
        },],
        objective_name = "mean",
        minimize = False,
    )

    times_tracked = []
    time_tracker = 0
    delta = 0
    while time_tracker + delta < max_time:
        start = time.time()

        parameters, trial_index = ax_client.get_next_trial()
        raw_data = evaluate(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    with open(f'../../surdata/Ax_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-meanopt_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([ax_client,time_tracker,vals], file)