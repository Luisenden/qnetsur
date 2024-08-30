from optimizingcd import main_cd as simulation
import numpy as np
import pickle
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="set seed")
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()


    n = 100
    np.random.seed(args.seed)
    vals = { # define fixed parameters for given simulation function 
                'protocol':'ndsrs', 
                'A': simulation.adjacency_random_tree(n),
                'p_gen': 0.9,  # generation rate
                'q_swap': np.random.random_sample(n),
                'p_swap': 1,  # success probability
                'p_cons': 0.9/4,  # consumption rate
                'cutoff': 28,
                'M': 10,
                'qbits_per_channel': 5,
                'N_samples' : 1000,
                'total_time': 1000,
                }
    sim_result = simulation.simulation_cd(**vals)

    with open(f'../../surdata/sim_random_N=1000_T=1000_seed{args.seed}.pkl', 'wb') as file:
        pickle.dump(sim_result, file)