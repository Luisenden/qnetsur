import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from netsquid_qswitch.runtools import Scenario, Simulation, SimulationMultiple
def simulation_qswitch(nnodes, total_runtime_in_seconds, connect_size, server_node_name, num_positions, 
                       buffer_size, bright_state_population, decoherence_rate, beta, loss, T2,
                       include_classical_comm, distances, repetition_times, N, ret='default', seed=42):
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
    rates, fidelities_per_route, share_per_node = collect_results(sm=sm, scenario=scenario)
    if ret is 'default':
        return np.array(rates), fidelities_per_route, share_per_node
    else:
        res = pd.DataFrame({'Rate [Hz]':rates})
        res['Number of nodes'] = nnodes
        res['B'] = buffer_size
        res[r'$\alpha$'] = bright_state_population[0]
        res = pd.concat([res, fidelities_per_route], axis=1)
        return res
    
def collect_results(sm, scenario, nnodes): 
    """
    Helper function to collect results.
    """   
    share_per_node = []
    rates = []
    fidels_per_route = []
    for result in sm.results:
        nodes_involved_per_run = pd.Series([x for nodes in result.nodes_involved for x in nodes])
        node_names_involved = nodes_involved_per_run.unique().tolist()
        node_names_involved.remove(scenario.server_node_name)
        share_involved_per_run = nodes_involved_per_run.value_counts()\
            / scenario.total_runtime_in_seconds / result.capacity / scenario.connect_size
        share_per_node.append( share_involved_per_run )
        rates.append(result.capacity)
        fidels = {}
        for name in node_names_involved:
            fidels['F_'+name] = np.mean([fidel for fidel, nodes_involved in
                                         zip(result.fidelities, result.nodes_involved) if name in nodes_involved])
        fidels_per_route.append(fidels)
    share_per_node = pd.DataFrame.from_records(share_per_node)
    share_per_node = share_per_node.reindex(sorted(share_per_node.columns,
                                                   key=lambda x: int(re.search('_([0-9]+)', x).group(1))), axis=1)
    bool_involved = np.array([any(share_per_node.columns.str.contains(str(node))) for node in range(nnodes)])
    for node_not_involved in np.array(range(nnodes))[~bool_involved]:
        share_per_node[f'leaf_node_{node_not_involved}'] = 0
    share_per_node = share_per_node.add_prefix('% ')
    fidelities_per_route = pd.DataFrame.from_records(fidels_per_route)
    return rates, fidelities_per_route, share_per_node


if __name__ == "__main__":
    """Simulation for getting the experimental results presented in Figure 2|d-e of
        'Humphreys, Peter C., et al. "Deterministic delivery of remote 
        entanglement on a quantum network." Nature 558.7709 (2018): 268-273'.

        Paramter settings
        * two users generating bipartide entanglement (via switch in the center)
        * link efficiency (eta in Humphreys, et al.) = 8
        * clock = 10 Hz
        * distance between two leaf nodes = 2 meters,
            i.e., one meter between user node and switch node

    """
    distances = [0.001, 0.001]  # in [km]
    rep_times = [0.01, 0.01]  # in [s] 
    vals = {  # define fixed parameters for given simulation function
                'nnodes': 2,  # number of leaf nodes
                'total_runtime_in_seconds': 100,  # simulation time in [s]
                'decoherence_rate': 0, 
                'connect_size': 2,  # size of GHZ state (2=bipartide)
                'server_node_name': 'leaf_node_0',  # server node
                'T2': 0,  # T2 noise when qubits are idle in memory
                'beta': 0.2,  # loss coefficient as defined in vardoyan et al.
                'loss': 1,  # other losses of the system (1 means no loss)
                'include_classical_comm': False,
                'num_positions': 9,  # number of memory positions in all nodes
                'distances': distances,  # distances of leaf nodes to switch node
                'repetition_times':rep_times,  # link generation rate
                'N': 5,  # batch size 
                'ret': 'df',  # result format
                'seed': 42
            }
    
    dfs = []
    for alpha in np.linspace(0.001, 0.5, 20):  # different bright-state populations
        for buffer in range(1, 6):  # vary over different buffer sizes
            vals['bright_state_population'] = [alpha, alpha]
            vals['buffer_size'] = buffer
            df = simulation_qswitch(**vals)
            print(df)
            dfs.append(df)
        print(f'done {alpha:.3f}')
    df_result = pd.concat(dfs, axis=0)

    # plot
    fig, ax = plt.subplots(2, figsize=(10, 6))
    sns.lineplot(x=r'$\alpha$', y='F_leaf_node_1', data=df_result, hue='B', ax=ax[0], marker='^')
    ax[0].set_ylabel('Fidelity')
    sns.lineplot(x=r'$\alpha$', y='Rate [Hz]', data=df_result, hue='B', ax=ax[1], marker='^')
    ax[1].set_ylabel('Rate [Hz]')
    plt.tight_layout()
    plt.show()