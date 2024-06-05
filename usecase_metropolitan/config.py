"""Central config file for all executables (surrogate.py, vs_*.py)."""
import sys
sys.path.append('../')

import numpy as np
import argparse

import simulation

class Config:
    """
    Configuration class for setting up and running simulations with specific parameters.
    """
    def __init__(self, initial_model_size=5):
        self.vals = {
        'network_config_file': 'starlight.json', # network configuration file
        'N': 1, # simulation sample size
        'total_time': 2e12, # simulation time
        'cavity': 500 # experimental parameter for atom-cavity cooperativity
        }
        self.name = 'starlight'
        self.sim = simulation.simulation_rb
        self.initial_model_size = initial_model_size

        # parse user inputs
        parser = argparse.ArgumentParser(description="Import globals")
        parser.add_argument("--memory_max", type=int, default=105, help="Maximum number of memories allocatable per node. Type: int")
        parser.add_argument("--memory_budget", type=int, default=450, help="Budget of memory qubits. Type: int")
        parser.add_argument("--time", type=float, default=0.05,
                            help="Maximum time allowed for optimization (in hours). Type: float")
        parser.add_argument("--iterator", type=int, dest='time',
                            help="Maximum optimization cycles allowed; overwrites arg 'time'. Type: int")
        parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
        parser.add_argument("--folder", type=str, default='../../surdata/rb/',
                        help="Directory to store result data. Type: str")
        self.args, _ = parser.parse_known_args()
        self.rng = np.random.default_rng(seed=self.args.seed) # set rng for optimization 

        self.set_variables()

    def set_variables(self) -> None:
        """
        Defines variables and their bounds for the simulation based on the network topology.
        """
        vars = {
        'range':{},
        'choice':{},
        'ordinal':{}
        } 
        for i in range(9):
            vars['range'][f'mem_size_node_{i}'] = ([10, self.args.memory_max], 'int')
        self.vars = vars

    def simobjective(self, simulation, kwargs: dict):
        """
        Runs the simulation with the given parameters and calculates objectives.
        """
        mem_size = []
        for key, value in list(kwargs.items()):
            if 'size' in key:
                mem_size.append(value)
                kwargs.pop(key)
        kwargs['mem_size'] = mem_size

        mean, std = simulation(**kwargs)
        kwargs['mem_size'] = np.array(mem_size)
        objectives = mean - max(np.sum(mem_size)-self.args.memory_budget, 0) # objective
        objectives_std = std
        raw = mean
        return objectives, objectives_std, raw

