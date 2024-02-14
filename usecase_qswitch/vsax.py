from config import *
from src.utils import get_parameters
from ax.service.ax_client import AxClient

def evaluate(parameters) -> float:
    x = {**vals, **parameters}
    mean_obj, std_obj, _ = simwrapper(simulation=simulation_qswitch, kwargs=x)
    mean_obj = np.nan_to_num(mean_obj, copy=True, nan=0)
    std_obj = np.nan_to_num(std_obj, copy=True, nan=0)
    return (np.sum(mean_obj), np.sum(std_obj))
          

if __name__ == '__main__':

    # define fixed parameters and set variable ranges
    vals = {  
        'nnodes': NLEAF_NODES,
        'total_runtime_in_seconds': 300,  # simulation time [s]
        'connect_size': 2,
        'server_node_name': 'leaf_node_0',
        'distances': np.array([42, 7, 13, 30, 17, 24, 11, 10, 39, 43])[:NLEAF_NODES],
        'repetition_times': [10 ** -3] * NLEAF_NODES,  # time between generation attempts
        'beta': 0.2, # link efficiency coefficient
        'loss': 1, # loss parameter
        'T2': 0,
        'include_classical_comm': False,
        'num_positions': 300,
        'decoherence_rate': 0,
        'N': 10, # batch size
        'buffer_size': 1
    }
    for node in range(NLEAF_NODES):
        vars['range'][f'bright_state_{node}'] = ([0.001, .1], 'float') 
    # for node in range(NLEAF_NODES):
    #     vars['range'][f'buffer_{node}'] = ([1, 15], 'int') 

    max_time= MAX_TIME * 3600 # in sec

    # create instance of AxClient and set objective
    ax_client = AxClient(verbose_logging=False, random_seed=SEED)
    ax_client.create_experiment( 
        name="qswitch-simulation-seed{SEED}",
        parameters=get_parameters(vars),
        minimize=False,
        objective_name="evaluate",
    )
    # optimize according to best practices (see https://ax.dev/tutorials/gpei_hartmann_service.html)
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
    with open(f'../../surdata/qswitch/AX_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-servernode_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([result,time_tracker,vals], file)