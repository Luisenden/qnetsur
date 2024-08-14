"""Central config file used in `surrogate.py`, `vs_*.py`."""
import sys
# Include parent directory to use qnetsur package
sys.path.append('..')
import numpy as np
import argparse
import re

from optimizingcd import main_cd as simulation

class Config:
    """
    Configuration class for setting up and running simulations with specific parameters.
    """
    def __init__(self, ntop=10):
        self.vals = { # define fixed parameters for given simulation function 
            'protocol':'ndsrs', 
            'p_gen': 0.9,  # generation rate
            'p_swap': 1,  # success probability
            'return_data':'avg', 
            'progress_bar': None,
            'total_time': 1000,
            'N_samples' : 100,
            'p_cons': 0.9/4,  # consumption rate
            'qbits_per_channel': 5,
            'cutoff': 28,
            'M': 10,
            }
        self.sim = simulation.simulation_cd
        self.ntop = ntop # number of top configurations used to find promising neighbors

        # parse global params
        parser = argparse.ArgumentParser(description="Import globals")
        parser.add_argument("--topo", type=str, default='tree-2-1', help="Network topology; \
                            Use 'tree-i-j' (=tree-#children-#levels) or 'randtree-i' (=randtree-#nodes) or 'square-i' (=square-ixilattice), where i,j are integers. Type: str")
        parser.add_argument('--level', action='store_true', help="If tree-i-j is used for topology \
                         level assigns the same swap probability per tree level.", default=False)
        parser.add_argument('--user', action='store_true', help="Specifies if only user nodes \
                         (min vertex degree) should be selected for optimization.", default=False)
        parser.add_argument("--time", type=float, default=0.05,
                            help="Maximum time allowed for optimization (in hours). Type: float")
        parser.add_argument("--iterator", type=int, dest='time',
                            help="Maximum optimization cycles allowed; overwrites arg 'time'. Type: int")
        parser.add_argument("--score", action='store_true',
                            help="If argument is used, the ml score is used in the acquisition process. Type:bool", default=False)
        parser.add_argument("--seed", type=int, default=42,
                            help="Global seed for random number generation for the optimizer. Type: int")
        parser.add_argument("--folder", type=str, default='../../surdata/cdtest/',
                            help="Directory to store result data. Type: str")
        self.args, _ = parser.parse_known_args()

        self.rng = np.random.default_rng(seed=self.args.seed) # set rng for optimization 
        self.name = self.args.topo   
        self.topo_input = self.name.split('-')
        self.kind = self.topo_input[0]
        self.folder = self.args.folder
        if self.args.level == True:
            assert self.kind == 'tree', 'If "level" is used, topology must be "tree".'
        
        # set adjacency matrix and variables
        self.set_adjacency()
        self.set_variables()
    
    def set_adjacency(self) -> None:
        """
        Generates and sets the adjacency matrix for the network based on the specified topology.
        """
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
        self.user_indices = np.where(self.A.sum(axis=1) == 1)[0]


    def set_variables(self) -> None:
        """
        Defines variables and their bounds for the simulation based on the network topology.
        """
        vars = { # define variables and bounds for given simulation function 
            'range': {},
            'choice':{},
            'ordinal':{}
        }
    
        for i in self.user_indices:
            vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')
        self.vars = vars
    
    def simobjective(self, simulation, kwargs: dict) -> tuple:
        """
        Runs the simulation with the given parameters and calculates objectives.

        Args:
            simulation (function): The simulation function to run.
            kwargs (dict): Keyword arguments for the simulation function including q_swap values.

        Returns:
            tuple: Tuple containing the objectives, their standard deviations, and raw data.
        """
        q_swap = np.ones(len(self.A))
        for key,value in list(kwargs.items()):  # assign q_swap to level nodes
            if 'q_swap' in key:
                q_swap[int(re.findall(r'\d+', key)[0])] = value
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