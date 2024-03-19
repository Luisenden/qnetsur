
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
        vars['range'][f'bright_state_{node}'] = ([.0, .1], 'float') 

    # user input:
    max_time= MAX_TIME * 3600 # in sec

    # create an instance of the surrogate model and run optimization
    sim = Surrogate(simwrapper, simulation_qswitch, vals=vals, vars=vars, sample_size=initial_model_size)
    sim.optimize(max_time=max_time, verbose=True)

    with open(f'../../surdata/qswitch/SU_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-servernode_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
            pickle.dump(sim, file)
