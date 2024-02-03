import numpy as np
import pandas as pd
import re
from netsquid_qswitch.runtools import Scenario, Simulation, SimulationMultiple
import time
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

def simulation_qswitch(nnodes, total_runtime_in_seconds, connect_size, server_node_name, num_positions, 
                       buffer_size, bright_state_population, decoherence_rate, eta, loss, T2, include_classical_comm, distances, repetition_times, N):


    scenario = Scenario(total_runtime_in_seconds=total_runtime_in_seconds, 
                        connect_size=connect_size, 
                        server_node_name=server_node_name,
                        num_positions=num_positions, 
                        buffer_size=buffer_size, 
                        bright_state_population=bright_state_population,
                        T2=T2,
                        eta=eta,
                        loss=loss,
                        decoherence_rate=decoherence_rate, 
                        include_classical_comm=include_classical_comm)
    #print("SCENARIO: ", scenario)
    simulation = Simulation(scenario=scenario, distances=distances, repetition_times=repetition_times)

    sm = SimulationMultiple(simulation=simulation, number_of_runs=N)
    sm.run()
    
    share_per_node = []
    rates = []
    fidels_per_node = []
    for result in sm.results:
        nodes_involved_per_run = pd.Series([x for nodes in result.nodes_involved for x in nodes])
        node_names_involved = nodes_involved_per_run.unique().tolist()
        node_names_involved.remove(server_node_name)
        share_involved_per_run = nodes_involved_per_run.value_counts() / total_runtime_in_seconds / result.capacity / scenario.connect_size 
        share_per_node.append( share_involved_per_run )
        rates.append(result.capacity)

        fidels = {}
        for name in node_names_involved:
            fidels['F_'+name] = np.mean([fidel for fidel, nodes_involved in zip(result.fidelities, result.nodes_involved) if name in nodes_involved]) 
        
        fidels_per_node.append(fidels)
    
    share_per_node = pd.DataFrame.from_records(share_per_node)
    share_per_node = share_per_node.reindex(sorted(share_per_node.columns, key=lambda x: int(re.search('_([0-9]+)', x).group(1))), axis=1)
    share_per_node = share_per_node.add_prefix('% ')

    fidelities = pd.DataFrame.from_records(fidels_per_node)

    res = pd.DataFrame({'Rate [Hz]':rates})
    res['B'] = buffer_size
    res[r'$\alpha$'] = bright_state_population[0]
    #res = pd.concat([res, fidelities, share_per_node], axis=1)
    
    return np.array(rates), fidelities, share_per_node
    
    


if __name__ == "__main__":

    distances = [0.001, 0.001] # in [km]
    rep_times = [0.1, 0.1] # in [s] (=10 Hz clock)

    vals = { # define fixed parameters for given simulation function
                'nnodes': 2,
                'total_runtime_in_seconds': 5, # simulation time [s]
                'decoherence_rate': 0,
                'connect_size': 2,
                'server_node_name': 'leaf_node_0',
                'T2': 0,
                'include_classical_comm': False,
                'num_positions': 3000,
                'distances': distances,
                'repetition_times':rep_times,
                'N': 5, # batch size 
                # 'eta':8
            }
    
    dfs = []
    for alpha in np.linspace(0.01, 0.49, 20):
        for buffer_size in range(1,5):

            vals['bright_state_population'] = [alpha, alpha]
            vals['buffer_size'] = buffer_size

            df = simulation_qswitch(**vals)
            dfs.append(df)
            print(f'done {alpha}')
    
    df_result = pd.concat(dfs, axis=0)
    print(df_result)

    fig, ax = plt.subplots(2, figsize=(10, 6))
    sns.lineplot(x=r'$\alpha$', y='F_leaf_node_1', data=df_result, hue='B', color='b', ax=ax[0], marker='^')
    ax[0].set_ylabel('Fidelity', color='b')

    sns.lineplot(x=r'$\alpha$', y='Rate [Hz]', data=df_result, hue='B', color='r', ax=ax[1], marker='^')
    ax[1].set_ylabel('Rate [Hz]', color='r')

    plt.tight_layout()
    plt.show()