
"""
Configurtaion to simulate optimziation of two-user-one-server scenario of 
'Vardoyan, Gayane, and Stephanie Wehner. "Quantum network utility maximization." 
2023 IEEE International Conference on 
Quantum Computing and Engineering (QCE). Vol. 1. IEEE, 2023.'

The goal is to find the optimal link bright-state population values (alpha=1-Fidelity) to maximize
the given utility function, Distillable Entanglement as defined in the paper.
"""

from datetime import datetime
import pandas as pd
import numpy as np 

from config import Config
from src.utils import Surrogate, Simulation


def get_results(result:pd.DataFrame, params:list, sim:Simulation) -> None:
        obj_sums = np.sum(sim.y, axis=1)
        best_index = np.argmax(obj_sums)

        best = obj_sums[best_index]
        best_std = np.sqrt(np.sum(np.square(sim.y_std), axis=1))[best_index] # calc standard deviation
        
        best_e2e_rate = np.mean(sim.y_raw[best_index][0][1:]) # server_node is left out
        best_e2e_rate_std = np.mean(sim.y_raw[best_index][1][1:])
        best_e2e_fidel = np.mean(sim.y_raw[best_index][2][1:])
        best_e2e_fidel_std = np.mean(sim.y_raw[best_index][3][1:])

        result['Utility'].append(best)
        result['Utility_std'].append(best_std)
        result['Rate'].append(best_e2e_rate)
        result['Rate_std'].append(best_e2e_rate_std)
        result['Fidelity'].append(best_e2e_fidel)
        result['Fidelity_std'].append(best_e2e_fidel_std)

        params.append(sim.X_df.iloc[best_index])


def run(bottleneck_length:int = 100, n:int =20):
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
        utility standard deviation, rate, fidelity, the respective standard deviations and server distance.
        """
        result = {'server_distance':[], 'Utility': [], 'Utility_std':[], 'Rate':[], 'Rate_std':[], 'Fidelity':[], 'Fidelity_std':[]}
        best_params = []
        for server_distance in np.linspace(1.5, bottleneck_length, n):
                #instatiante surrogate model and run optimization
                # try:
                vals['distances'] = [server_distance, 2, 2]
                sim = Surrogate(conf.simobjective, conf.sim, values=vals, variables=vars, sample_size=conf.initial_model_size, rng=conf.rng)
                sim.optimize(limit=limit, verbose=True)

                get_results(result, best_params, sim)
                result['server_distance'].append(server_distance)
                print(f'done server distance {server_distance}')
                # except:
                        # print(f'An exception occurred at server distance {server_distance}')
        
        df_params = pd.DataFrame.from_records(best_params)
        df_result = pd.DataFrame.from_records(result)
        df = df_params.join(df_result, how='left')

        df.to_csv(storage_path)
        return df


if __name__ == '__main__':

        # load configuration
        conf = Config()
        limit = conf.args.time
        limit_kind = 'hours' if isinstance(limit, float) else 'cycles'
        storage_path = conf.args.folder+f'SU_{conf.name}_{limit}{limit_kind}_SEED{conf.args.seed}_'\
                  +datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.csv'

        vals = {  # define fixed parameters for given simulation function
            'nnodes': conf.args.nleaf,
            'total_runtime_in_seconds': 5,  # in [s]
            'decoherence_rate': 0,
            'connect_size': 2,
            'server_node_name': 'leaf_node_0',
            'T2': 0,
            'beta': 0.2, # link efficiency coefficient
            'loss': 1, # loss parameter
            'buffer_size': 20,
            'include_classical_comm': False,
            'num_positions': 200,
            'repetition_times': [10 ** -3] * conf.args.nleaf, # repetition time in [s]
            'N': 20 # sample size 
        }

        vars = {
            'range': {},
            'choice':{},
            'ordinal':{}
        } 
        vars['range']['bright_state_server'] = ([.0, .1], 'float') 
        vars['range']['bright_state_user'] = ([.0, .1], 'float')

        # optimize at different bottleneck-link lengths (1-100km)
        df = run(n=1)