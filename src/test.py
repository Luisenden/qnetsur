import ax
import sklearn
import pandas
import numpy as np
import scipy

from optimizingcd import main_cd as simulation


## PROTOCOL
protocol = 'ndsrs' # 'ndsrs' is Node-Dependent SRS

## TOPOLOGY
d = 2
k = 3
A = simulation.adjacency_tree(d,k)
topology = 'tree'

## HARDWARE
p_gen = 1
p_swap = 1
qbits_per_channel = 5

## SOFTWARE
q_swap = [1,0.2,1,0.5,1,0.7,1]
max_links_swapped = 4
p_cons = 0.1

## CUTOFF#
cutoff = 20

## SIMULATION
data_type = 'avg'
N_samples = 100
total_time = int(cutoff*5)
plot_nodes = [0,1,2]
randomseed = 2
np.random.seed(randomseed)
data = simulation.simulation_cd(protocol, A, p_gen, q_swap, p_swap,
                                p_cons, cutoff, max_links_swapped,
                                qbits_per_channel, N_samples,
                                total_time,
                                progress_bar='notebook',
                                return_data=data_type)
print(data)