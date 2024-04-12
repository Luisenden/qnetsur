"""
Central script containing objects for surrogate optimization and helper function to retrieve format of parameters for Ax-platform.
"""

import numpy as np
import time
import pandas as pd
from scipy.stats import truncnorm
from functools import partial

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method
try:    
    set_start_method('spawn')
except RuntimeError:
     pass

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, make_scorer

import os
from importlib import import_module

USE_CASE = os.environ.get('USE_CASE') or 'usecase_cd'  # default to usecase_cd if not set
try:
    config = import_module(f'{USE_CASE}.config')
except ImportError:
    raise ImportError(f"Cannot import config.py for '{USE_CASE}'")

class Simulation:
    """
    Object to run quantum network simulations. 
    It allows for the specification of both fixed and variable simulation parameters, 
    supports the generation of random parameter sets, and facilitates the running of simulations 
    through a user-defined simulation function.

    Attributes
    ----------
    sim_wrapper : function
        The wrapper function that preprocesses simulation
        parameters before execution and adapts outputs according to needs of use case.
    sim : function
        The actual simulation function to be executed.
    vals : dict
        Fixed parameters for the simulation.
    vars : dict, optional
        Variable parameters for the simulation, allowing for dynamic adjustments.

    Methods
    -------
    run_sim(x, vals=None)
        Executes a single simulation run with the given parameters.
    run_exhaustive(x, N=10)
        Executes `N` simulations in parallel.
    get_random_x(n)
        Generates `n` sets of random parameters based on the
        specified variable parameters.
    """

    def __init__(self, sim_wrapper, sim, vals=None, vars=None):
        # specify fixed parameters
        self.vals = vals
        # specify variable parameters
        self.vars = vars

        # simulation function handler
        self.sim_wrapper = sim_wrapper
        # simulation function
        self.sim = sim

    def run_sim(self, x :dict, vals :dict = None) -> list:
        """
        Runs the quantum network simulation with the provided parameters.

        Args:
            x (dict): Parameters for the simulation.

        Returns:
            list: Results of the simulation.
        """
        xrun = {**self.vals, **x} if vals == None else {**vals, **x}
        res = self.sim_wrapper(self.sim, xrun)
        return res

    def run_exhaustive(self, x :dict, vals :dict = None, N=10, seed=42) -> list:
        """
        Runs the quantum network simulation with the provided parameters N times in parallel.
        """
        xrun = {**x, **vals}
        task = partial(self.sim_wrapper, self.sim)
        with Pool(processes=N) as pool:
            res = pool.map(task, [{**xrun, **{'seed': (i+1)*seed}} for i in range(N)])
            pool.close()
            pool.join()
        return res

    def get_random_x(self, n :int) -> dict:
        """
        Generates random parameters for the simulation.
        """
        assert all(isinstance(val, tuple) for val in self.vars['range'].values()) and n > 0,\
            f"Dimension types must be a tuple (sample-list, dataype) and n must be greater zero."

        x = {}
        for dim, par in self.vars['range'].items():
                vals = par[0]
                if par[1] == 'int':
                    x[dim] = config.rng_sur.integers(vals[0], vals[1], n) if n > 1\
                        else config.rng_sur.integers(vals[0], vals[1])
                elif par[1] == 'float':
                    x[dim] = config.rng_sur.uniform(vals[0], vals[1], n) if n > 1\
                        else config.rng_sur.uniform(vals[0], vals[1])
                else:
                    raise Exception('Datatype must be "int" or "float".')

        for dim, vals in self.vars['ordinal'].items():
                x[dim] = config.rng_sur.choice(vals, size=n) if n > 1\
                    else config.rng_sur.choice(vals)
                    
        for dim, vals in self.vars['choice'].items():
                x[dim] = config.rng_sur.choice(vals, n) if n > 1\
                    else config.rng_sur.choice(vals)       

        return x
    
class Surrogate(Simulation):
    """
    Initializes the Surrogate model with a simulation wrapper,
    the actual simulation function, 
    and a set of fixed and variable parameters. It sets up the 
    environment for running surrogate model-based optimizations, 
    including the multiprocessing setup for parallel simulations.

    Parameters
    ----------
    sim_wrapper : function
        A function that wraps around the actual simulation to 
        perform pre-processing of simulation parameters.
    sim : function
        The actual simulation function to be optimized.
    vals : dict
        Fixed parameters for the simulation, passed to every simulation run.
    sample_size : int
        The number of samples to use for the initial surrogate model training.
    vars : dict, optional
        Variable parameters for the simulation that can be optimized.
    k : int, optional
        Exponent in acquisition transition function.

    Attributes
    ----------
    X : dict
        Training set parameters.
    X_df : pandas.DataFrame
        DataFrame version of `X` for easier manipulation and model training.
    y : list
        Simulation results corresponding to each set of parameters in `X`.
    y_std : list
        Standard deviation of simulation results,
        providing insight into result variability.
    model : class
        Regression model used for surrogate modeling. 
        Default is SVR (Support Vector Regression) from scikit-learn.
    mmodel : class
        Multi-output wrapper for the regression model,
        allowing it to output multiple predictions.
    mmodel_std : class
        Separate multi-output model for predicting the
        standard deviation of simulation results.
    build_time : list
        Time taken to build or update the surrogate model.
    sim_time : list
        Time taken for simulation runs.
    optimize_time : list
        Time taken for optimization processes.
    procs : int
        Number of processes to use for parallel simulations,
        based on CPU count.

    Methods
    -------
    get_neighbour(max_time, current_time, x)
        Generates a neighbouring set of parameters based on current optimization state.
    acquisition(max_time, current_time)
        Selects new parameters for evaluation by balancing exploration and exploitation.
    update()
        Updates the surrogate model with new data points collected from simulation runs.
    optimize(max_time, verbose=False)
        Conducts the optimization process to find optimal simulation parameters.
    """
    def __init__(self, sim_wrapper, sim, vals, vars, sample_size, k=4):
        super().__init__(sim_wrapper, sim, vals, vars)

        # storage for time profiling
        self.sim_time = []
        self.build_time = []
        self.acquisition_time = []
        self.optimize_time = []

        # set multiprocessing
        self.procs = mp.cpu_count()
        self.sample_size = sample_size
        if self.procs > sample_size:
            self.procs = sample_size

        # storage target value
        self.y = []
        self.y_std = []
    
        # storage raw values
        self.y_raw = []

        # storage model declaration
        self.model = SVR()
        self.model_scores = {'SVR': [], 'DecisionTree': []}
        self.k = k  # coefficient in neighbour selection

    def get_neighbour(self, max_time, current_time, x :dict) -> dict:
        """
        Generates most promising parameters in limited neighbouring region 
        according to current knowledge of surrogate model and depending on the time left.
        """
        x_n = {}
        f = (1-np.log(1+current_time/max_time)**2)**self.k

        size = int(current_time/max_time*10000 + 10)
        for dim, par in self.vars['range'].items():
                vals = par[0]
                if par[1] == 'int':
                    std = f * (vals[1] - vals[0])/2
                    x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std,
                                             loc=x[dim], scale=std, size=size, random_state=config.rng_sur).astype(int)
                elif par[1] == 'float':
                    std = f * (vals[1] - vals[0])/2
                    x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std,
                                             loc=x[dim], scale=std, size=size, random_state=config.rng_sur) 
                else:
                    raise Exception('Datatype must be "int" or "float".')

        for dim, vals in self.vars['ordinal'].items():
                pos = x[dim]  # current position
                loc = vals.index(pos)/len(vals)  # corresponding location between 0 and 1
                pval = np.linspace(0,1,len(vals))
                std = f/2
                probs = truncnorm.pdf(pval,
                                      (0-loc)/std, (1-loc)/std, scale=std,
                                      loc=loc, random_state=config.rng_sur)/len(x)  # corresponding weights
                probs /= probs.sum()  # normalize probabilities
                x_n[dim] = config.rng_sur.random.choice(vals, size=size , p=probs)

        for dim, vals in self.vars['choice'].items():
                x_n[dim] = config.rng_sur.random.choice(vals, size)

        # select best prediction as next neighbour
        samples_x = pd.DataFrame(x_n).astype(object)
        samples_y = self.mmodel.predict(samples_x.values)
        fittest_neighbour_index = np.argsort(np.array(samples_y).sum(axis=1))[-1]
        x_fittest = samples_x.iloc[fittest_neighbour_index].to_dict()
        return x_fittest

    def acquisition(self, max_time, current_time, n=10) -> pd.DataFrame:
        """
        Computes n new data points according to estimated improvement 
        and degree of exploration and adds the data to the training sample.
        """
        start = time.time()
        y_obj_vec = np.sum(self.y, axis=1)
        newx = []
        top_selection = self.X_df.iloc[np.argsort(y_obj_vec)[-n:]]  # select top n candidates
        for x in top_selection.iloc:  # get most promising neighbour according to surrogate
            neighbour = self.get_neighbour(max_time=max_time,
                                           current_time=current_time,
                                           x=x.to_dict())
            newx.append(neighbour)
        self.X_df_add = pd.DataFrame.from_records(newx).astype(object)
        self.acquisition_time.append(time.time()-start)       


    def run_multiple_and_add_target_values(self, X, n=10) -> None:
        start = time.time()
        with Pool(processes=n) as pool:
            y_temp = pool.map(self.run_sim, X)
            pool.close()
            pool.join()
        self.sim_time.append(time.time() - start)  # measure simulation time

        # add new data
        for y_i in y_temp:
            yi, yi_std, *yi_raw = y_i
            self.y.append(yi)
            self.y_std.append(yi_std)
            self.y_raw += yi_raw
    

    def train_models(self) -> None:
        """
        Trains the machine lerning models on current data set
        and chooses the one with smaller error to use for prediction.
        """
        # train/update surrogate models
        start = time.time()
        current_min = np.nanmin(self.y)  # nan value handling
        self.y = np.nan_to_num(self.y, copy=True, nan=current_min,
                               posinf=current_min, neginf=current_min).tolist()
        self.y_std = np.nan_to_num(self.y_std, copy=True, nan=0, posinf=0, neginf=0).tolist()
        score_svr = cross_val_score(MultiOutputRegressor(SVR()),  # get current error of models
                                    self.X_df.drop('Iteration', axis=1).values,
                                    self.y, scoring=make_scorer(mean_absolute_error)).mean()
        score_tree = cross_val_score(MultiOutputRegressor(DecisionTreeRegressor(random_state=42)),
                                    self.X_df.drop('Iteration', axis=1).values,
                                    self.y, scoring=make_scorer(mean_absolute_error)).mean()

        if score_svr < score_tree:  # set model that currently performs best (smaller error)
             self.model = SVR()
        else:
             self.model = DecisionTreeRegressor(random_state=42)
        print(f'scores: svr = {score_svr} and tree={score_tree}')
        self.model_scores['SVR'].append(score_svr)
        self.model_scores['DecisionTree'].append(score_tree)

        self.mmodel = MultiOutputRegressor(self.model)
        self.mmodel_std = MultiOutputRegressor(self.model)

        self.mmodel.fit(self.X_df.drop('Iteration', axis=1).values, self.y)
        self.mmodel_std.fit(self.X_df.drop('Iteration', axis=1).values, self.y_std)
        self.build_time.append(time.time()-start)

    def update(self, counter) -> None:
        """
        Updates the model with new data. Executes simulation on most promising points found and adds 
        simulation results to training dataset.
        """

        # execute simulation and add target values
        self.run_multiple_and_add_target_values(X=self.X_df_add.iloc)

        # add parameter set
        self.X_df_add['Iteration'] = counter
        self.X_df = pd.concat([self.X_df, self.X_df_add], axis=0, ignore_index=True)

        # train models
        self.train_models()


    def gen_initial_set(self, max_time, verbose):
        """
        Generates the initial training set.
        """
        if verbose: print("Start optimization ...")
        optimize_start = time.time()
        
        # generate initial training set X
        start = time.time()
        X = self.get_random_x(self.sample_size)
        self.X_df = pd.DataFrame(X).astype(object)
        self.build_time.append(time.time() - start)

        self.run_multiple_and_add_target_values(X=self.X_df.iloc)
        self.X_df['Iteration'] = 0

        # train models
        self.train_models()

        initial_optimize_time = time.time()-optimize_start
        self.optimize_time.append(initial_optimize_time)

        self.max_optimize_time = max_time - initial_optimize_time

    def optimize_with_timer(self, verbose):
        """
        Optimization with a set maximum number of seconds.
        """
        current_times = []
        current_time = 0
        delta = 0
        counter = 1
        while current_time+delta < self.max_optimize_time:
            start = time.time()

            self.acquisition(self.max_optimize_time, current_time)
            self.update(counter)

            self.optimize_time.append(time.time()-start)
            current_times.append(time.time()-start)
            current_time = np.sum(current_times)
            delta = np.mean(current_times)
            counter +=1
            if verbose:
                print(f'Time left for optimization: {self.max_optimize_time-current_time:.2f}s')

    def optimize_with_iteration(self, max_iteration, verbose):
        """
        Optimization with a set maximum number of iterations.
        """
        counter = 1
        while counter < max_iteration:
            start = time.time()

            self.acquisition(max_iteration, counter)
            self.update(counter)

            self.optimize_time.append(time.time()-start)
            counter +=1
            if verbose:
                print(f'Iteration {counter}/{max_iteration}')


    def optimize(self, max_time, verbose=False) -> None:
        """
        Conducts the optimization process to find optimal simulation parameters.
        """
        if isinstance(max_time, float):
            self.gen_initial_set(max_time, verbose=verbose)
            self.optimize_with_timer(verbose=verbose)
        
        else:
            self.gen_initial_set(max_time[0], verbose=verbose)
            if max_time[1]:
                print('Optimize with timer.')
                self.optimize_with_timer(verbose=verbose)
            else:
                if verbose: print('Optimize with iterator.')
                self.optimize_with_iteration(max_time[0], verbose=verbose)
        
        if verbose: print('Optimization finished.')

def get_parameters(variables):
    """
    Extracts and formats parameters from a dictionary for use in the Ax-platform optimization tool.

    Parameters
    ----------
    vars : dict
        A dictionary where keys correspond to parameter types (e.g., 'range', 'ordinal', 'choice')
        and values provide the definitions of these parameters.

    Returns
    -------
    list
        A list of parameter definitions formatted for use in optimization routines,
        with each parameter represented as a dictionary detailing its name,
        type, and constraints or choices.
    """
    parameters = []
    for k in variables:
        for key,value in variables[k].items():
            typ = 'choice' if k == 'ordinal' else k
            if typ != 'choice':
                parameters.append(
                    {
                    "name": str(key),
                    "type": typ,
                    "bounds": value[0],
                    })
            else:
                parameters.append(
                    {
                    "name": str(key),
                    "type": typ,
                    "values": value,
                    })
    return parameters