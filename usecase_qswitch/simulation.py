import numpy as np
from netsquid_qswitch.runtools import Scenario, Simulation, SimulationMultiple

def simulation_qswitch(total_runtime_in_seconds, connect_size, rates, num_positions, buffer_size, decoherence_rate, T2, include_classical_comm, N):

    scenario = Scenario(total_runtime_in_seconds, connect_size, rates, num_positions, buffer_size, decoherence_rate, T2, include_classical_comm)
    simulation = Simulation(scenario=scenario)
    sm = SimulationMultiple(simulation=simulation, number_of_runs=N)
    sm.run()

    fidel = [result.mean_fidelity for result in sm.results]
    mean_fidelities = np.mean(fidel)
    std_fidelity = np.std(fidel)
    
    capas = [result.capacity for result in sm.results]
    mean_capacities = np.mean(capas)
    std_capacities = np.std(capas)

    return [mean_capacities, mean_fidelities], [std_capacities, std_fidelity]
    
    


if __name__ == "__main__":

    N = 10
    nnodes = 4

    scenario = Scenario(total_runtime_in_seconds=100 * 10 ** (-6),
                connect_size=4, 
                rates=[1.9 * 1e6] * nnodes,
                num_positions=1000,
                buffer_size=[np.inf]*4,
                decoherence_rate=0,
                T2=10 ** (-7),
                include_classical_comm=False)
    simulation = Simulation(scenario=scenario)
    sm = SimulationMultiple(simulation=simulation, number_of_runs=N)
    sm.run()

    fidel = [result.mean_fidelity for result in sm.results]
    mean_fidelities = np.mean(fidel)
    std_fidelity = np.std(fidel)
    
    capas = [result.capacity for result in sm.results]
    mean_capacities = np.mean(capas)
    std_capacities = np.std(capas)
