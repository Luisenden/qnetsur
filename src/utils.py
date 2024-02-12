import numpy as np
import time
import pandas as pd
from scipy.stats import truncnorm

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
    Provides a high-level interface for running quantum network simulations. 
    It allows for the specification of both fixed and variable simulation parameters, 
    supports the generation of random parameter sets for exploratory simulations, and 
    facilitates the running of simulations through a user-defined simulation function.

    Attributes
    ----------
    sim_wrapper : function
        The wrapper function that preprocesses simulation
        parameters before execution.
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
    get_random_x(n)
        Generates `n` sets of random parameters based on the
        specified variable parameters.
    """

    def __init__(self, sim_wrapper, sim, vals, vars):
        # specify fixed parameters
        self.vals = vals
        # specify variable parameters
        self.vars = vars
        # simulation function handler
        self.sim_wrapper = sim_wrapper
        self.sim = sim

    def run_sim(self,x :dict, vals :dict = None) -> list:
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

    def get_random_x(self,n) -> dict:
        """
        Generates random parameters for the simulation.

        Args:
            n (int): Number of random parameter sets to generate.

        Returns:
            dict: Randomly generated parameters.
        """
        assert all(isinstance(val, tuple) for val in self.vars['range'].values()) and n > 0,\
            f"Dimension types must be a tuple (sample-list, dataype) and n must be greater zero."

        x = {}
        for dim, par in self.vars['range'].items():
                vals = par[0]
                if par[1] == 'int':
                    x[dim] = np.random.randint(vals[0], vals[1], n) if n > 1\
                        else np.random.randint(vals[0], vals[1])
                elif par[1] == 'float':
                    x[dim] = np.random.uniform(vals[0], vals[1], n) if n > 1\
                        else np.random.uniform(vals[0], vals[1])
                else:
                    raise Exception('Datatype must be "int" or "float".')

        for dim, vals in self.vars['ordinal'].items():
                x[dim] = np.random.choice(vals, size=n) if n > 1\
                    else np.random.choice(vals)
                    
        for dim, vals in self.vars['choice'].items():
                x[dim] = np.random.choice(vals, n) if n > 1\
                    else np.random.choice(vals)       

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
    improvement : list
        Record of improvements in objective function
        value after each optimization step.

    Methods
    -------
    get_neighbour(max_time, current_time, x)
        Generates a neighboring set of parameters based on current optimization state.
    acquisition(max_time, current_time)
        Selects new parameters for evaluation by balancing exploration and exploitation.
    update()
        Updates the surrogate model with new data points collected from simulation runs.
    optimize(max_time, verbose=False)
        Conducts the optimization process to find optimal simulation parameters.
    """
    def __init__(self, sim_wrapper, sim, vals, vars, sample_size, k=4):
        super().__init__(sim_wrapper, sim, vals, vars)

        # profiling storage
        self.sim_time = []
        self.build_time = []
        self.optimize_time = []

        # multiprocessing
        self.procs = mp.cpu_count()
        print(f'{self.procs} THREADS AVAILABLE.')
        self.sample_size = sample_size
        if self.procs > sample_size:
            self.procs = sample_size

        # storage target value
        self.y = []
        self.y_std = []
        self.y_raw = []

        # model declaration
        self.model = SVR
        self.model_scores = {'SVR': [], 'DecisionTree': []}
        self.k = k  # coefficient in neighbor selection

        # improvement storage
        self.improvement = []

    def get_neighbour(self, max_time, current_time, x :dict) -> dict:
        """
        Generates most promising parameters in limited neighboring region for the simulation
        according to current knowledge of surrogate model.
        """
        x_n = {}
        f = (1-np.log(1+current_time/max_time)**2)**self.k

        size = int(current_time/max_time*10000 + 10)
        for dim, par in self.vars['range'].items():
                vals = par[0]
                if par[1] == 'int':
                    std = f * (vals[1] - vals[0])/2
                    x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std,
                                             loc=x[dim], scale=std, size=size).astype(int)
                elif par[1] == 'float':
                    std = f * (vals[1] - vals[0])/2
                    x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std,
                                             loc=x[dim], scale=std, size=size) 
                else:
                    raise Exception('Datatype must be "int" or "float".')

        for dim, vals in self.vars['ordinal'].items():
                pos = x[dim] # current position
                loc = vals.index(pos)/len(vals)  # corresponding location between 0 and 1
                pval = np.linspace(0,1,len(vals))
                std = f/2
                probs = truncnorm.pdf(pval,
                                      (0-loc)/std, (1-loc)/std, scale=std,
                                      loc=loc)/len(x)  # corresponding weights
                probs /= probs.sum()  # normalize probabilities
                x_n[dim] = np.random.choice(vals, size=size , p=probs)

        for dim, vals in self.vars['choice'].items():
                x_n[dim] = np.random.choice(vals, size)

        # select best prediction as neighbor
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
        y_obj_vec = np.sum(self.y, axis=1)
        newx = []
        top_selection = self.X_df.iloc[np.argsort(y_obj_vec)[-n:]]  # select top n candidates
        for x in top_selection.iloc:  # get most promising neighbor according to surrogate
            neighbour = self.get_neighbour(max_time=max_time,
                                           current_time=current_time,
                                           x=x.to_dict())
            newx.append(neighbour)
        self.X_df_add = pd.DataFrame.from_records(newx).astype(object)

    def update(self, counter, n=10) -> None:
        """
        Updates the model with new data.
        """
        start = time.time()
        with Pool(processes=n) as pool:
            y_temp = pool.map(self.run_sim, self.X_df_add.iloc)
            pool.close()
            pool.join()
        self.sim_time.append(time.time() - start)

        # calculate improvement of new data point to previous best observed point
        impr = np.max([np.mean(y_i[0]) for y_i in y_temp])-np.max([np.mean(y_i) for y_i in self.y])
        self.improvement.append(impr)

        # build surrogate model
        start = time.time()
        self.X_df_add['Iteration'] = counter
        self.X_df = pd.concat([self.X_df, self.X_df_add], axis=0, ignore_index=True)

        # add new data
        for y_i in y_temp:
            yi, yi_std, *yi_raw = y_i
            self.y.append(yi)
            self.y_std.append(yi_std)
            self.y_raw += yi_raw

        # train/update surrogate model
        score_svr = cross_val_score(MultiOutputRegressor(SVR()),
                                    self.X_df.drop('Iteration', axis=1).values,
                                    self.y, scoring=make_scorer(mean_absolute_error)).mean()
        score_tree = cross_val_score(MultiOutputRegressor(DecisionTreeRegressor()),
                                    self.X_df.drop('Iteration', axis=1).values,
                                    self.y, scoring=make_scorer(mean_absolute_error)).mean()

        if score_svr < score_tree:
             self.model = SVR
        else:
             self.model = DecisionTreeRegressor
        print(f'scores: svr = {score_svr} and tree={score_tree}')
        self.model_scores['SVR'].append(score_svr)
        self.model_scores['DecisionTree'].append(score_tree)

        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel_std = MultiOutputRegressor(self.model())

        self.y = np.nan_to_num(self.y, copy=True, nan=np.min(self.y), posinf=np.min(self.y), neginf=np.min(self.y)).tolist()
        self.y_std = np.nan_to_num(self.y_std, copy=True, nan=0, posinf=0, neginf=0).tolist()
        self.mmodel.fit(self.X_df.drop('Iteration', axis=1).values, self.y)
        self.mmodel_std.fit(self.X_df.drop('Iteration', axis=1).values, self.y_std)
        self.build_time.append(time.time()-start)

    def optimize(self, max_time, verbose=False) -> None:
        """
        Conducts the optimization process to find optimal simulation parameters.
        """
        if verbose: print("Start optimization ...")
        optimize_start = time.time()
        
        # generate initial training set X
        start = time.time()
        X = self.get_random_x(self.sample_size)
        self.X_df = pd.DataFrame(X).astype(object)
        self.build_time.append(time.time() - start)

        start = time.time() 
        with Pool(processes=self.procs) as pool:
            y_temp = pool.map(self.run_sim, self.X_df.iloc)
            pool.close()
            pool.join()
        
        self.X_df['Iteration'] = 0
        self.sim_time.append(time.time() - start)

        # train model
        start = time.time()

        # add new training values
        for y_i in y_temp:
            yi,yi_std,*yi_raw = y_i
            self.y.append(yi)
            self.y_std.append(yi_std)
            self.y_raw += yi_raw

        score_svr = cross_val_score(MultiOutputRegressor(SVR()),
                                    self.X_df.drop('Iteration', axis=1).values,
                                    self.y, scoring=make_scorer(mean_absolute_error)).mean()
        score_tree = cross_val_score(MultiOutputRegressor(DecisionTreeRegressor()),
                                     self.X_df.drop('Iteration', axis=1).values,
                                     self.y, scoring=make_scorer(mean_absolute_error)).mean()
        
        if verbose: print(f'scores: svr = {score_svr} and tree={score_tree}')
        self.model_scores['SVR'].append(score_svr)
        self.model_scores['DecisionTree'].append(score_tree)

        if score_svr < score_tree:
             self.model = SVR
        else:
             self.model = DecisionTreeRegressor

        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel_std = MultiOutputRegressor(self.model())

        self.y = np.nan_to_num(self.y, copy=True, nan=np.min(self.y), posinf=np.min(self.y), neginf=np.min(self.y)).tolist()
        self.y_std = np.nan_to_num(self.y_std, copy=True, nan=0, posinf=0, neginf=0).tolist()
        self.mmodel.fit(self.X_df.drop('Iteration', axis=1).values, self.y)
        self.mmodel_std.fit(self.X_df.drop('Iteration', axis=1).values, self.y_std)
        
        self.build_time.append(time.time()-start)
        initial_optimize_time = time.time()-optimize_start
        self.optimize_time.append(initial_optimize_time)

        # optimization
        max_optimize_time = max_time - initial_optimize_time
        if verbose and max_optimize_time < 0:
            print("Initial model generated, but no time left for optimization after initial build.")
        if verbose:
            print(f'After initial build, time left for optimization: {max_optimize_time:.2f}s')
        current_times = []
        current_time = 0
        delta = 0
        counter = 1
        while current_time+delta < max_optimize_time:
            start = time.time()
            self.acquisition(max_optimize_time,current_time)
            self.optimize_time.append(time.time()-start)

            self.update(counter)
            current_times.append(time.time()-start)
            current_time = np.sum(current_times)
            delta = np.mean(current_times)
            counter +=1
            if verbose:
                print(f'Time left for optimization: {max_optimize_time-current_time:.2f}s')

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