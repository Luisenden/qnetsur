import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from netsquid_qswitch.runtools import Scenario, Simulation, SimulationMultiple


def simulation_qswitch(nnodes, total_runtime_in_seconds, connect_size, server_node_name, num_positions, 
                       buffer_size, bright_state_population, decoherence_rate, beta, loss, T2,
                       include_classical_comm, distances, repetition_times, N, seed=42):
    """
    Simulates a quantum switch network to evaluate its performance over a specified runtime. The simulation
    accounts for node connectivity, quantum buffer sizes, state decoherence, and other quantum mechanical properties
    to assess network efficiency and quantum state fidelity.

    Parameters
    ----------
    nnodes : int
        Number of nodes in the network.
    total_runtime_in_seconds : float
        Total simulation duration in seconds.
    connect_size : int
        The number of qubits or quantum channels constituting the connection size.
    server_node_name : str
        Identifier for the server node within the network.
    num_positions : int
        Number of positions or locations in the network for node placement.
    buffer_size : int
        Quantum memory buffer size at each node.
    bright_state_population : float
        Population of the bright state in the quantum nodes, affecting the initial state quality.
    decoherence_rate : float
        Rate at which quantum states undergo decoherence, impacting fidelity.
    beta : float
        Efficiency parameter for quantum state transfer or entanglement swapping.
    loss : float
        Quantum channel loss rate, influencing communication success.
    T2 : float
        Coherence time of the qubits, indicating how long they maintain their quantum state.
    include_classical_comm : bool
        Indicates whether the simulation includes classical communication overhead.
    distances : list of float
        Physical distances between nodes, affecting quantum communication quality.
    repetition_times : list of float
        Intervals at which quantum operations are repeated to ensure reliability.
    N : int
        Number of simulation runs to average out stochastic effects.
    ret : str, optional
        Determines the format of the return value. The default 'default' option returns rates, fidelities, and share per node.

    Returns
    -------
    If `ret` is 'default', the function returns a tuple of:
        numpy.array
            Array of capacity rates achieved in the simulation.
        pandas.DataFrame
            Average fidelity values for quantum states transferred between nodes.
        pandas.DataFrame
            Proportional share of network resources utilized by each node.
    Otherwise, if `ret` is not 'default', it returns a pandas.DataFrame containing rates, buffer size, the first element of 
    bright state population, and fidelities per route.
    """
    scenario = Scenario(total_runtime_in_seconds=total_runtime_in_seconds,
                        connect_size=connect_size,
                        server_node_name=server_node_name,
                        num_positions=num_positions,
                        buffer_size=buffer_size,
                        bright_state_population=bright_state_population,
                        T2=T2,
                        beta=beta,
                        loss=loss,
                        decoherence_rate=decoherence_rate,
                        include_classical_comm=include_classical_comm)

    simulation = Simulation(scenario=scenario, distances=distances, repetition_times=repetition_times, seed=seed)
    sm = SimulationMultiple(simulation=simulation, number_of_runs=N)
    sm.run()
    # get results
    rates, fidelities_per_route = collect_results(sm=sm, scenario=scenario)
    return rates, fidelities_per_route
    
def collect_results(sm, scenario): 
    """
    Helper function to collect results.
    """   
    rates_per_node = []
    fidels_per_node = []
    for result in sm.results:
        nodes_involved_per_run = pd.Series([x for nodes in result.nodes_involved for x in nodes])
        node_names_involved = nodes_involved_per_run.unique().tolist()
        rate_per_node = (nodes_involved_per_run.value_counts()\
            / scenario.total_runtime_in_seconds ).to_dict() 
        rates_per_node.append(rate_per_node)
        fidels = {}
        for name in node_names_involved:
            fidelities = [fidel for fidel, nodes_involved in
                                         zip(result.fidelities, result.nodes_involved) if name in nodes_involved]
            fidels['F_'+name] = np.mean(fidelities)

        fidels_per_node.append(fidels)

    fidelities_per_node = pd.DataFrame.from_records(fidels_per_node)
    rates_per_node = pd.DataFrame.from_records(rates_per_node)
    return rates_per_node, fidelities_per_node