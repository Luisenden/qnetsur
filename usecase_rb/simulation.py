import numpy as np
from datetime import datetime
import json, os
import random

import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
try:    
    set_start_method('spawn')
except RuntimeError:
     pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sequence.topology import Node

import pandas as pd
from sequence.app.random_request import RandomRequestApp
from sequence.topology.router_net_topo import RouterNetTopo

def update_memory_config(file_path, new_memo_size, total_time,seed):
    random.seed(seed)
    proc = mp.current_process().ident

    # Load JSON data from file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    data["stop_time"] = total_time

    # Call the function to update memo_size dynamically
    seeds = random.sample(range(len(data["nodes"])), len(data["nodes"])) 
    for i,node in enumerate(data["nodes"]):
            node["memo_size"] = new_memo_size[i]
            node["seed"] = seeds[i]

    # Write the updated JSON back to the file
    with open(str(proc)+'.json', 'w') as file:
        json.dump(data, file, indent=2)

def get_fidelity_by_efficiency(C: int):
    """
    C: the cavity cooperativity
    50 <= C <= 500
    """
    gama = 14
    gama_star = 32
    delta_omega = 0
    gama_prime = (C+1)*gama
    tau = gama_prime + 2*gama_star
    F_e = 0.5 * (1 + gama_prime**2 / (tau**2 + delta_omega**2))
    return F_e


def get_component(node: "Node", component_type: str):
    for comp in node.components.values():
        if type(comp).__name__ == component_type:
            return comp

    raise ValueError("No component of type {} on node {}".format(component_type, node.name))

def set_parameters(cavity:int, network_topo):

    C = cavity
    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    bsm_nodes = network_topo.get_nodes_by_type(RouterNetTopo.BSM_NODE)

    # set memory parameters
    MEMO_FREQ = 2e3
    MEMO_EXPIRE = 1.3
    MEMO_EFFICIENCY = 0.75
    MEMO_FIDELITY = get_fidelity_by_efficiency(C) #0.9349367588934053
    for node in routers:
        memory_array = node.get_components_by_type("MemoryArray")[0]  # assume only 1 memory array
        memory_array.update_memory_params("frequency", MEMO_FREQ)
        memory_array.update_memory_params("coherence_time", MEMO_EXPIRE)
        memory_array.update_memory_params("efficiency", MEMO_EFFICIENCY)
        memory_array.update_memory_params("raw_fidelity", MEMO_FIDELITY)

    # set detector parameters
    DETECTOR_EFFICIENCY = 0.8
    DETECTOR_COUNT_RATE = 5e7
    DETECTOR_RESOLUTION = 100
    for node in bsm_nodes:
        bsm = node.get_components_by_type("SingleAtomBSM")[0]
        bsm.update_detectors_params("efficiency", DETECTOR_EFFICIENCY)
        bsm.update_detectors_params("count_rate", DETECTOR_COUNT_RATE)
        bsm.update_detectors_params("time_resolution", DETECTOR_RESOLUTION)

    # set quantum channel parameters
    ATTENUATION = 0.0002
    QC_FREQ = 1e11
    for qc in network_topo.get_qchannels():
        qc.attenuation = ATTENUATION
        qc.frequency = QC_FREQ

    # set entanglement swapping parameters
    SWAP_SUCC_PROB = 0.64
    SWAP_DEGRADATION = 0.99
    for node in routers:
        node.network_manager.protocol_stack[1].set_swapping_success_rate(SWAP_SUCC_PROB)
        node.network_manager.protocol_stack[1].set_swapping_degradation(SWAP_DEGRADATION)
    
def run(network_topo,n):
    tl = network_topo.get_timeline()
    tl.show_progress = False
    
    apps = []
    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    router_names = [node.name for node in routers]
    for i, node in enumerate(routers):
        app_node_name = node.name
        others = router_names[:]
        others.remove(app_node_name)
        max_mem_to_reserve = len(node.get_components_by_type("MemoryArray")[0])//2
        app = RandomRequestApp(node, others,
                            min_dur=1e12, max_dur=2e12, min_size=max(1,max_mem_to_reserve-max_mem_to_reserve//2),
                            max_size=max_mem_to_reserve, min_fidelity=0.8, max_fidelity=1.0, seed=i*n)
        apps.append(app)
        app.start()

    tl.init()
    tl.run()

    # log results
    initiators = []
    responders = []
    start_times = []
    end_times = []
    memory_sizes = []
    fidelities = []
    wait_times = []
    throughputs = []
    for node in routers:
        initiator = node.name
        reserves = node.app.reserves
        _wait_times = node.app.get_wait_time()
        _throughputs = node.app.get_all_throughput()
        min_size = min(len(reserves), len(_wait_times), len(_throughputs))
        reserves = reserves[:min_size]
        _wait_times = _wait_times[:min_size]
        _throughputs = _throughputs[:min_size]
        for reservation, wait_time, throughput in zip(reserves, _wait_times,
                                                    _throughputs):
            responder, s_t, e_t, size, fidelity = reservation
            initiators.append(initiator)
            responders.append(responder)
            start_times.append(s_t)
            end_times.append(e_t)
            memory_sizes.append(size)
            fidelities.append(fidelity)
            wait_times.append(wait_time)
            throughputs.append(throughput)
    log = {"Initiator": initiators, "Responder": responders,
        "Start_time": start_times, "End_time": end_times,
        "Memory_size": memory_sizes, "Fidelity": fidelities,
        "Wait_time": wait_times, "Throughput": throughputs}

    df = pd.DataFrame(log)
    return df


def simulation_rb(network_config_file, cavity, total_time, N, mem_size):
    
    results = []
    proc = mp.current_process().ident

    for n in range(N):
        update_memory_config(network_config_file, mem_size, total_time,seed=n)
        network_topo = RouterNetTopo(str(proc)+'.json')
        nodes = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)

        set_parameters(cavity=cavity, network_topo=network_topo)
        try:
            df = run(network_topo,n)

            completed_requests_per_node = df.groupby('Initiator').size()
            res = np.zeros(len(nodes))
            for i,node in enumerate(nodes):
                if node.name in completed_requests_per_node: res[i] = completed_requests_per_node[node.name]
        except:
            res = np.zeros(len(nodes))
            for i,node in enumerate(nodes):
                if node.name in completed_requests_per_node: res[i] = completed_requests_per_node[node.name]

        results.append(res)
        os.remove(str(proc)+'.json')
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    print(f'mean {mean} and std {std} from proc {proc}')
    return mean, std


if __name__ == "__main__":

    import pickle
    
    network_config_file = "starlight.json"  # network configuration 
    total_time = 2e13  # total simulation time in [ps]

    # policy 0: evenly distributed memories
    even = [50]*9 
    # policy 1: weighted policy of Wu et al. Table 3
    weighted = [25, 91, 67, 24, 67, 24, 103, 25, 24]
    # policy 2: drawn from surrogate optimization results (execute sur.py EXEC TIME = 12 hours)
    surrogate_weighted = [59, 99, 20, 19, 41, 19, 100, 65, 21]

    results = {0: [], 1: [], 2: []}
    for i, policy in enumerate([even, weighted, surrogate_weighted]):
        for j in range(10):
            proc = mp.current_process().ident
            update_memory_config(network_config_file, policy, total_time,seed=42) 
            network_topo = RouterNetTopo(str(proc)+'.json')
            os.remove(str(proc)+'.json')

            set_parameters(cavity=500, network_topo=network_topo)
            nodes = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)

            df = run(network_topo,j)
            completed_requests_per_node = df.groupby('Initiator').size()
            res = np.zeros(len(nodes))
            for k,node in enumerate(nodes):
                if node.name in completed_requests_per_node: res[k] = completed_requests_per_node[node.name]

            results[i].append(sum(res))
            print(f'done policy {i} iteration {j}')

    df = pd.DataFrame.from_records(results)
    with open(f'../../surdata/rb/sim_policy_comparison_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
        pickle.dump(df, file)

