from time import time
import numpy as np
from datetime import datetime
import json
import os
import multiprocessing as mp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sequence.topology import Node

import pandas as pd
from sequence.app.random_request import RandomRequestApp
from sequence.topology.router_net_topo import RouterNetTopo

def update_memory_config(file_path, new_memo_size, seed):
    np.random.seed(1000+seed)
    proc = mp.current_process().ident

    # Load JSON data from file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Call the function to update memo_size dynamically
    for i,node in enumerate(data["nodes"]):
            node["memo_size"] = new_memo_size[i]
            node["seed"] = np.random.randint(0,100)

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
    
def run(network_topo):
    tl = network_topo.get_timeline()
    tl.show_progress = False
    
    apps = []
    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    router_names = [node.name for node in routers]
    for i, node in enumerate(routers):
        app_node_name = node.name
        others = router_names[:]
        others.remove(app_node_name)
        app = RandomRequestApp(node, others, i,
                            min_dur=1e13, max_dur=2e13, min_size=10,
                            max_size=len(node.get_components_by_type("MemoryArray")[0]), min_fidelity=0.8, max_fidelity=1.0)
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
    #df.to_csv("request_with_perfect_network"+datetime.now().strftime("%m-%d-%Y_%H:%M")+".csv")
    return df


    # node_names = []
    # start_times = []
    # end_times = []
    # memory_sizes = []
    # for node in routers:
    #     node_name = node.name
    #     for reservation in node.network_manager.protocol_stack[1].accepted_reservation:
    #         s_t, e_t, size = reservation.start_time, reservation.end_time, reservation.memory_size
    #         if reservation.initiator != node.name and reservation.responder != node.name:
    #             size *= 2
    #         node_names.append(node_name)
    #         start_times.append(s_t)
    #         end_times.append(e_t)
    #         memory_sizes.append(size)
    # log = {"Node": node_names, "Start_time": start_times,
    #     "End_time": end_times, "Memory_size": memory_sizes}
    # df = pd.DataFrame(log)
    # print(df)

def rb_simulation(network_config_file, cavity, N, **kwargs):
    
    results = []
    proc = mp.current_process().ident
    print('list', list(kwargs.values()))
    for n in range(N):
        update_memory_config(network_config_file, list(kwargs.values()),seed=n)
        network_topo = RouterNetTopo(str(proc)+'.json')
        nodes = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)

        set_parameters(cavity=cavity, network_topo=network_topo)
        df = run(network_topo)

        completed_requests_per_node = df.groupby('Initiator').size()

        res = np.zeros(len(nodes))
        for i,node in enumerate(nodes):
            if node.name in completed_requests_per_node: res[i] = completed_requests_per_node[node.name]

        results.append(res)
    
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    
    os.remove(str(proc)+'.json')
    return mean, std



if __name__ == "__main__":

    start = time()
    network_config_file = "starlight.json"
    update_memory_config(network_config_file, [25, 91, 67, 24, 67, 24, 103, 25, 24], seed=42) #source for weighted: https://github.com/sequence-toolbox/Chicago-metropolitan-quantum-network/blob/master/sec5.4-two-memory-distribution-policies/uneven_memory.json
    network_topo = RouterNetTopo(network_config_file)

    set_parameters(cavity=500, network_topo=network_topo)
    nodes = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)

    for node in nodes:
        print(len(node.get_components_by_type("MemoryArray")[0]))

    print('time:', time()-start)
