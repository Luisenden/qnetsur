
"""
Configurtaion to simulate optimziation of two-user-one-server scenario of 
'Vardoyan, Gayane, and Stephanie Wehner. "Quantum network utility maximization." 
2023 IEEE International Conference on 
Quantum Computing and Engineering (QCE). Vol. 1. IEEE, 2023.'

The goal is to find the optimal link bright-state population values (alpha=1-Fidelity) to maximize
the given utility function, Distillable Entanglement as defined in the paper.
"""

from config import *
from src.utils import *


def run(vars:dict, vals:dict, max_optimize_time:float, path:str, bottleneck_length:int, n=20):
        """
        Executes a series of optimizations over a range of server distances from 1-100km to evaluate the performance of a quantum switch network.
        This function iterates through specified server distances, instantiates a surrogate model for each, and runs optimization 
        to find the best network parameters based on utility, which is a function of the network's end-to-end rate and fidelity.

        Parameters
        ----------
        max_optimize_time : float
        The maximum amount of time allowed for each optimization process.
        vals : dict
        A dictionary of initial values and parameters to be passed to the simulation function.
        path : str
        The directory path where the results of the optimizations will be saved as a pickle file.

        Returns
        -------
        df : pandas.DataFrame
        A DataFrame containing the best parameters found for each server distance, alongside the corresponding utility, 
        utility standard deviation, rate, fidelity, and server distance.
        """
        result = {'server_distance':[], 'Utility': [], 'Utility_std':[], 'Rate':[], 'Fidelity':[]}
        best_params = []
        for server_distance in np.linspace(1, bottleneck_length, n):
                #instatiante surrogate model and run optimization
                try:
                        vals['distances'] = [server_distance, 2, 2]
                        sim = Surrogate(simwrapper, simulation_qswitch, vals=vals, vars=vars, sample_size=initial_model_size)

                        sim.optimize(max_time=max_optimize_time, verbose=True)

                        obj_sums = np.sum(sim.y, axis=1)
                        best = np.max(obj_sums)
                        best_index = np.argmax(obj_sums)
                        best_std = np.sum(sim.y_std, axis=1)[best_index]
                        best_e2e_rate = sim.y_raw[best_index][0]
                        best_e2e_fidel = sim.y_raw[best_index][1]

                        result['Utility'].append(best)
                        result['Utility_std'].append(best_std)
                        result['Rate'].append(best_e2e_rate)
                        result['Fidelity'].append(best_e2e_fidel)
                        result['server_distance'].append(server_distance)

                        best_params.append(sim.X_df.iloc[best_index])
                except:
                        print(f'An exception occurred at server distance {server_distance}')
        
        df_params = pd.DataFrame.from_records(best_params)
        df_result = pd.DataFrame.from_records(result)
        df = df_params.join(df_result, how='left')
        with open(path+
                  f'Sur_df_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.3f}h_bottleneck-link_SEED{SEED_OPT}_'
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(df, file)
        return df


if __name__ == '__main__':

        storage_path='../../surdata/qswitch/'  # storage path
        max_time=MAX_TIME * 3600  # maximum allowed optimization time in [s]

        vals = {  # define fixed parameters for given simulation function
            'nnodes': NLEAF_NODES,
            'total_runtime_in_seconds': 50,  # in [s]
            'decoherence_rate': 0,
            'connect_size': 2,
            'server_node_name': 'leaf_node_0',
            'T2': 0,
            'beta': 0.2, # link efficiency coefficient
            'loss': 1, # loss parameter
            'buffer_size': 1,
            'include_classical_comm': False,
            'num_positions': 10,
            'repetition_times': [10 ** -3] * NLEAF_NODES, # repetition time in [s]
            'N': 5 # batch size 
        }
        vars['range']['bright_state_server'] = ([0.001, .1], 'float') 
        vars['range']['bright_state_user'] = ([0.001, .1], 'float')
        # run the optimizations over different bottleneck link lengths (1-100km)
        df = run(vars=vars, vals=vals, max_optimize_time=max_time, path=storage_path, bottleneck_length=100, n=2)
        print(df)



