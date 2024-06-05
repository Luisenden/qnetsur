"""
Configuration script to run qswitch simulations (builds on `simulation.py`)
* optimziation of two-user-one-server scenario of [Vadoyan et al., 2023](https://ieeexplore.ieee.org/abstract/document/10313675) with 'sur_vardoyan_netsquid_comparison.py'
* optimization for more complex scenarios with scripts 'surrogate.py' and comparison with 'vs_xx.py'

The goal is to find the optimal bright-state population (alpha=1-Fidelity) for all links to maximize
the given utility function (based on Distillable Entanglement) as defined in the paper.
"""
import sys
sys.path.append('../')

import numpy as np
import argparse
import pandas as pd

import simulation

class Config:
    """
    Configuration class for setting up and running simulations with specific parameters.
    """
    def __init__(self, initial_model_size=5):
        self.initial_model_size = initial_model_size
        self.name = 'qswitch'
        self.sim = simulation.simulation_qswitch

        # get the globals
        parser = argparse.ArgumentParser(description="Import globals")
        parser.add_argument("--nleaf", type=int, default=3, help="Number of leaf nodes")
        parser.add_argument("--time", type=float, default=0.05,
                            help="Maximum time allowed for optimization (in hours). Type: float")
        parser.add_argument("--iterator", type=int, dest='time',
                            help="Maximum optimization cycles allowed; overwrites arg 'time'. Type: int")
        parser.add_argument("--seed", type=int, default=42, help="Global seed for random number generation for the optimizer")
        parser.add_argument("--folder", type=str, default='../../surdata/qswitchtest/',
                        help="Directory to store result data. Type: str")
        self.args, _ = parser.parse_known_args()
        self.rng = np.random.default_rng(seed=self.args.seed) # set rng for optimization 
    
    def D_H(self,F):
        return 1 + F*np.log2(F) + (1-F) * np.log2((1-F)/3) \
            if F > 0.81 else 1e-12  # yield of the so-called “hashing” protocol
    
    def set_default_values(self):
        self.vals = {  
            'nnodes': self.args.nleaf,
            'total_runtime_in_seconds': 5,  # simulation time [s]
            'connect_size': 2,
            'server_node_name': 'leaf_node_0',
            'distances': np.array([5, 10, 20, 30, 40, 50])[:self.args.nleaf], # set up to five leaf nodes
            'repetition_times': [10 ** -3] * self.args.nleaf,  # time between generation attempts
            'beta': 0.2, # link efficiency coefficient
            'loss': 1, # loss parameter
            'buffer_size': 20,
            'T2': 0,
            'include_classical_comm': False,
            'num_positions': 200,
            'decoherence_rate': 0,
            'N': 20, # batch size
        }
        vars = {
            'range': {},
            'choice':{},
            'ordinal':{}
        } 
        for node in range(self.args.nleaf):
            vars['range'][f'bright_state_{node}'] = ([1e-12, .1], 'float') 
        self.vars = vars

    def simobjective(self, simulation, kwargs: dict):
        """
        Wraps a simulation function to adjust its parameters based on the presence of bright states in the keyword arguments,
        then runs the simulation with adjusted distances, repetition times, and bright state populations for each node. The 
        wrapper calculates rates, fidelities, and shares per node, and then computes the utility of the network based on 
        Distillable Entanglement.

        Parameters
        ----------
        simulation : function
            The simulation function to be wrapped.
        kwargs : dict
            A dictionary of keyword arguments for the simulation function. Expected to include 'initial_distances', 
            'repetition_times', and any number of 'bright_state' entries.

        Returns
        -------
        U_D : numpy.ndarray
            The mean utility of the network across all routes, calculated as the mean of the logarithm of the product of 
            route rates and fidelities, representing Distillable Entanglement.
        U_D_std : numpy.ndarray
            The standard deviation of the utility across all routes, indicating the variability in the utility.
        raw : list
            A list containing the mean of the rates and the mean of the fidelities across all simulations."""

        bright_states = []
        for key,value in list(kwargs.items()):
            if 'bright_state' in key:
                bright_states.append(value)
                kwargs.pop(key)

        buffer_size = []
        for key,value in list(kwargs.items()):
            if 'buffer' in key:
                buffer_size.append(value)
                kwargs.pop(key)

        kwargs['bright_state_population'] = [bright_states[0]] + [bright_states[1]] * (self.args.nleaf-1)\
            if len(bright_states)==2 else bright_states
        
        kwargs['buffer_size'] = buffer_size if len(buffer_size)>1 else buffer_size[0]
        
        # run simulation
        rates, fidelities = simulation(**kwargs)

        # Distillable Entanglement 
        Ds = fidelities.applymap(self.D_H)
        U_Ds = pd.DataFrame(rates.values*Ds.values, columns=rates.columns,
                            index=rates.index).applymap(np.log) # Utility as defined in Vardoyan et al.,2023

        # Set NaN for nodes that are not involved
        bool_involved = np.array([any(U_Ds.columns.str.contains(str(node))) for node in range(self.args.nleaf)])
        for node_not_involved in np.array(range(self.args.nleaf))[~bool_involved]:
            U_Ds[f'leaf_node_{node_not_involved}'] = np.nan


        U_Ds = U_Ds.drop(kwargs['server_node_name'], axis=1)
        U_D = U_Ds.mean(axis=0).values
        U_D_std = U_Ds.std(axis=0).values

        raw = [rates.mean(axis=0).values, rates.std(axis=0).values, fidelities.mean(axis=0).values, fidelities.std(axis=0).values]
        return U_D, U_D_std, raw