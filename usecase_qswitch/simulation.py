import numpy as np
import pandas as pd
import re
from netsquid_qswitch.runtools import Scenario, Simulation, SimulationMultiple
import time
import itertools

def simulation_qswitch(nnodes, total_runtime_in_seconds, connect_size, rates, num_positions, buffer_size, decoherence_rate, T2, include_classical_comm, N):


    scenario = Scenario(total_runtime_in_seconds, 
                        connect_size, rates, 
                        num_positions, 
                        buffer_size, 
                        T2, 
                        decoherence_rate, 
                        include_classical_comm)
    simulation = Simulation(scenario=scenario)

    sm = SimulationMultiple(simulation=simulation, number_of_runs=N)
    sm.run()

    # fidel = [result.mean_fidelity for result in sm.results]
    # mean_fidelities = np.mean(fidel)
    # std_fidelity = np.std(fidel)
    
    share_per_node = []
    capacities = []
    for result in sm.results:
        nodes_involved_per_run = [x for nodes in result.nodes_involved for x in nodes] 
        share_involved_per_run = pd.Series(nodes_involved_per_run).value_counts() / total_runtime_in_seconds / result.capacity / scenario.connect_size
        share_per_node.append( share_involved_per_run )
        capacities.append(result.capacity)
    
    share_per_node = pd.DataFrame.from_records(share_per_node)
    share_per_node = share_per_node.reindex(sorted(share_per_node.columns, key=lambda x: int(re.search('_([0-9]+)', x).group(1))), axis=1)
    
    return share_per_node, capacities
    
    


if __name__ == "__main__":

    start = time.time()
    N = 10
    nnodes = 3

    for i in range(2,11):
        scenario = Scenario(total_runtime_in_seconds=10 ** -4,
                    connect_size=2, 
                    rates=[(i-0.5) * 10 ** 7] + [i * 10 ** 7] * (nnodes-1),
                    num_positions=1000,
                    buffer_size= [0] * (nnodes-2) + [1000, 1],
                    decoherence_rate=0,
                    T2=10 ** (-7),
                    include_classical_comm=False)
        simulation = Simulation(scenario=scenario)
        sm = SimulationMultiple(simulation=simulation, number_of_runs=N)
        sm.run()

        # fidel = [result.mean_fidelity for result in sm.results]
        # mean_fidelities = np.mean(fidel)
        # std_fidelity = np.std(fidel)

        capas = [result.capacity for result in sm.results]
        mean_capacities = np.mean(capas)
        std_capacities = np.std(capas)


        states_per_node = []
        for result in sm.results:
            nodes_involved_per_run = [x for nodes in result.nodes_involved for x in nodes] 
            nodes_involved_per_run = pd.Series(nodes_involved_per_run).value_counts() / scenario.total_runtime_in_seconds
            states_per_node.append( nodes_involved_per_run )
    
        states_per_node = pd.DataFrame.from_records(states_per_node)
        states_per_node = states_per_node.reindex(sorted(states_per_node.columns, key=lambda x: int(re.search('_([0-9]+)', x).group(1))), axis=1)

        mean_share_per_node = states_per_node.mean(axis=0).values/result.capacity/scenario.connect_size
        print(mean_share_per_node)
        print('time', time.time()-start)