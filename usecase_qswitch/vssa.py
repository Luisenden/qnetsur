from config import *
from src.utils import *

from src.simulatedannealing import simulated_annealing 


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

        # user input:
        max_time = MAX_TIME * 3600 # in sec

        # baseline simulated annealing
        si = Simulation(simwrapper, simulation_qswitch, vals=vals, vars=vars )
        simanneal = partial(simulated_annealing, MAX_TIME=max_time, seed=SEED)
        
        result = simanneal(si)
        result = pd.DataFrame.from_records(result)

        with open(f'../../surdata/qswitch/SA_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-servernode_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(result, file)
