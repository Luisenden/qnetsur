"""Central configuration file (used in `surrogate.py`, `vs_*.py`) for metropolitan-network usecase. 
The goal is to maximize the number of request the network stack can fulfill by tuning the number of memory qubits 
allocated in each node. The network stack is simulated using the package `SeQUeNCe <https://github.com/sequence-toolbox/SeQUeNCe>`_ 
and an adpated run script `simulation.py` of a `sequence-toolbox script <https://github.com/sequence-toolbox/Chicago-metropolitan-quantum-network/blob/master/sec5.4-two-memory-distribution-policies/run.py>`_.
"""
import sys
sys.path.append('../')
import numpy as np
import argparse

import usecase_metropolitan.simulation as simulation

class Config:
    """
    Configuration class for setting up and running simulations with specific parameters.
    
    Attributes:
        vals (dict): A dictionary containing default simulation configuration values.
            - 'network_config_file' (str): Path to the network configuration file.
            - 'N' (int): Simulation sample size.
            - 'total_time' (float): Total simulation time.
        name (str): The name identifier for the configuration.
        sim (function): The simulation function to be used.
        args (argparse.Namespace): Parsed command-line arguments.
        rng (numpy.random.Generator): A random number generator initialized with a seed.
    """

    def __init__(self):
        """
        Initializes the configuration with default values, parses command-line arguments,
        and sets up the simulation environment.
        """

        self.vals = {
        'network_config_file': 'starlight.json', # network configuration file
        'N': 5, # simulation sample size
        'total_time': 2e13, # simulation time
        }
        self.name = 'starlight'
        # Set simulation function
        self.sim = simulation.simulation_rb

        # Parse user inputs using argparse
        parser = argparse.ArgumentParser(description="Import globals")
        
        parser.add_argument(
            "--memory_max",
            type=int,
            default=105,
            help="Maximum number of memories allocatable per node. Type: int"
        )
        parser.add_argument(
            "--memory_budget",
            type=int,
            default=450,
            help="Budget of memory qubits. Type: int"
        )
        parser.add_argument(
            "--time",
            type=float,
            default=0.05,
            help="Maximum time allowed for optimization (in hours). Type: float"
        )
        parser.add_argument(
            "--iterator",
            type=int,
            dest='time',
            help="Maximum optimization cycles allowed; overwrites arg 'time'. Type: int"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Global seed for random number generation for the optimizer"
        )
        parser.add_argument(
            "--folder",
            type=str,
            default='../../surdata/rbtest/',
            help="Directory to store result data. Type: str"
        )

        # Parse known arguments and store them in self.args
        self.args, _ = parser.parse_known_args()

        # Initialize the random number generator with the provided seed
        self.rng = np.random.default_rng(seed=self.args.seed) 

        # Set simulation variables (defined below)
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

    def simobjective(self, simulation, kwargs: dict) -> tuple:
        """
        Runs the simulation with the given parameters and calculates objectives.
        
        Args:
            simulation (function): The simulation function to be run.
            kwargs (dict): The keyword arguments to be passed to the simulation function.
        
        Returns:
            tuple: A tuple containing:
                - objectives (float): The calculated objective value.
                - objectives_std (float): The standard deviation of the objective.
                - raw (float): The raw mean result from the simulation.
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

