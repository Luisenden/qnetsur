"""Central config file for all executables (surrogate.py, vs_*.py)."""
import sys
sys.path.append('../')

import numpy as np
import argparse
import re

from optimizingcd import main_cd as simulation

class Config:
    """
    Configuration class for setting up and running simulations with specific parameters.

    Attributes:
        vars (dict): Variables and their bounds used in the simulation.
        vals (dict): Fixed parameters for the simulation function including protocol, probabilities, and other settings.
        sim (func): Simulation function to be executed.
        initial_model_size (int): The initial size of the model used in simulations.
        args (Namespace): Parsed command line arguments specifying the simulation scope.
        rng (Generator): Random number generator object initialized with a specific seed.
        name (str): The topology name derived from command line arguments.
        topo_input (list): Split parts of the 'topo' argument indicating the network layout.
        kind (str): Type of topology (e.g., 'tree', 'square', 'randtree').
        A (array): Adjacency matrix representing the topology of the network.
    """
    def __init__(self, initial_model_size=5):
        self.vals = { # define fixed parameters for given simulation function 
            'protocol':'ndsrs', 
            'p_gen': 0.9,  # generation rate
            'p_swap': 1,  # success probability
            'return_data':'avg', 
            'progress_bar': None,
            'total_time': 1000,
            'N_samples' : 5,
            'p_cons': 0.9/4,  # consumption rate
            'qbits_per_channel': 5,
            'cutoff': 28,
            'M': 10,
            }
        self.sim = simulation.simulation_cd
        self.initial_model_size = initial_model_size

        # parse global params
        parser = argparse.ArgumentParser(description="Import globals")
        parser.add_argument("--topo", type=str, default='tree-2-3', help="Network topology; \
                            Use 'tree-i-j' or 'randtree-i' or 'square-i', where i,j are integers. Type: str")
        parser.add_argument('--level', action='store_true', help="If tree-i-j is used for topology \
                         level assigns the same swap probability per tree level.", default=False)
        parser.add_argument('--user', action='store_true', help="Specifies if only user nodes \
                         (min vertex degree) should be selected for optimization.", default=False)
        parser.add_argument("--time", type=float, default=0.05,
                            help="Maximum time allowed for optimization (in hours). Type: float")
        parser.add_argument("--iterator", type=int, dest='time',
                            help="Maximum optimization cycles allowed; overwrites arg 'time'. Type: int")
        parser.add_argument("--seed", type=int, default=42,
                            help="Global seed for random number generation for the optimizer. Type: int")
        parser.add_argument("--folder", type=str, default='../../surdata/cd/',
                            help="Directory to store result data. Type: str")
        self.args, _ = parser.parse_known_args()

        self.rng = np.random.default_rng(seed=self.args.seed) # set rng for optimization 

        self.name = self.args.topo   
        self.topo_input = self.name.split('-')
        self.kind = self.topo_input[0]
        if self.args.level == True:
            assert self.kind == 'tree', 'If "level" is used, topology must be "tree".'
        
        self.set_adjacency()
        self.set_variables()
    
    def set_adjacency(self) -> None:
        if self.kind == 'square':
            A = simulation.adjacency_squared(int(self.topo_input[1]))
        elif self.kind == 'tree':
            A = simulation.adjacency_tree(int(self.topo_input[1]), int(self.topo_input[2]))
        elif self.kind == 'randtree':
            A = simulation.adjacency_random_tree(int(self.topo_input[1]))
        else:
            raise Exception(f'Make sure topology input is set correctly! Check help via "python surrogate.py -h".')
        self.A = A
        self.vals['A'] = self.A


    def set_variables(self) -> None:
        """
        Defines variables and their bounds for the simulation based on the network topology.
        """
        vars = { # define variables and bounds for given simulation function 
            'range': {},
            'choice':{},
            'ordinal':{}
        }
        if self.args.level:
            for i in range(int(self.topo_input[2])+1):
                vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')
        else:
            for i in range(len(self.A)):
                vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')
        self.vars = vars
    
    def simobjective(self, simulation, kwargs: dict):
        """
        Runs the simulation with the given parameters and calculates objectives.

        Args:
            simulation (function): The simulation function to run.
            kwargs (dict): Keyword arguments for the simulation function including q_swap values.

        Returns:
            tuple: Tuple containing the objectives, their standard deviations, and raw data.
        """
        q_swap = []
        for key,value in list(kwargs.items()):  # assign q_swap to level nodes
            if 'q_swap' in key:
                if self.args.level:
                    exp = int(re.findall('\d', key)[0]) 
                    q_swap_level = [value]*int(self.topo_input[1])**exp
                    q_swap += q_swap_level
                else:
                    q_swap.append(value)
                kwargs.pop(key)
        kwargs['q_swap'] = q_swap
        
        # run simulation and retrieve number of virtual neighbors per node
        result = simulation(**kwargs)
        mean_per_node = np.array([node[-1] for node in result[1]])
        std_per_node = np.array([np.sqrt(node[-1]) for node in result[3]])
        raw = mean_per_node

        if self.args.user:
            # get user nodes
            vertex_degree_user = min(kwargs['A'].sum(axis=1))
            user_indices = np.where(kwargs['A'].sum(axis=1) == vertex_degree_user)
            objectives = mean_per_node[user_indices]
            objectives_std = std_per_node[user_indices]
        else:
            objectives = mean_per_node
            objectives_std = std_per_node

        return objectives, objectives_std, raw