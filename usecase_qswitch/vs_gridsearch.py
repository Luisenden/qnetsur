from config import *
from src.utils import *

if __name__ == '__main__':

    # define fixed parameters and set variable ranges
    vals = {  
        'nnodes': NLEAF_NODES,
        'total_runtime_in_seconds': 5,  # simulation time [s]
        'connect_size': 2,
        'server_node_name': 'leaf_node_0',
        'distances': np.array([5, 10, 20, 30, 40, 50])[:NLEAF_NODES],
        'repetition_times': [10 ** -3] * NLEAF_NODES,  # time between generation attempts
        'beta': 0.2, # link efficiency coefficient
        'loss': 1, # loss parameter
        'buffer_size': 20,
        'T2': 0,
        'include_classical_comm': False,
        'num_positions': 200,
        'decoherence_rate': 0,
        'N': 20, # batch size
    }
    for node in range(NLEAF_NODES):
        vars['range'][f'bright_state_{node}'] = ([1e-12, .1], 'float') 

    # user input:
    max_time= MAX_TIME * 3600 # in sec

    evals = [] # storage for results
    times_tracked = [] # time tracking
    time_tracker = 0
    delta = 0

    # do grid search until maximum time is reached
    sim = Simulation(simwrapper, simulation_qswitch, vals=vals, vars=vars)
    while time_tracker + delta < max_time:
        start = time.time()
        x = sim.get_random_x(1)
        eval = sim.run_sim(x)
        evalset = x.copy()
        evalset['objective'], evalset['std'], evalset['raw'] = eval
        evals.append(evalset)
        times_tracked.append(time.time()-start)
        time_tracker = np.sum(times_tracked)
        delta = np.mean(times_tracked)
    
    gridsearch = pd.DataFrame.from_records(evals)
    with open(f'../../surdata/qswitch/GS_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-servernode_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump([gridsearch,time_tracker,vals], file)