import numpy as np
import time
import pandas as pd
from scipy.stats import truncnorm

import multiprocessing as mp
from multiprocessing import get_context
# from usecase_rb.config import * #!!!! change !!!!

from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

import os
from importlib import import_module

USE_CASE = os.environ.get('USE_CASE') or 'usecase_cd'  # Default to usecase_cd if not set

try:
    config = import_module(f'{USE_CASE}.config')

except ImportError:
    raise ImportError(f"Cannot import config.py for '{USE_CASE}'")


class Simulation:
    """
    Class for running quantum network simulations.
s
    Args:
        func (function): The simulation function to be used.

    Attributes:
        vals (dict): A dictionary of fixed parameters.
        vars (dict): A dictionary of variable parameters.
        func (function): The simulation function.

    Methods:
        run_sim(x):
            Runs the simulation with the provided parameters where x is a dict.

        get_random_x(n):
            Generates random parameters for n simulation runs.

        diff(x, sample):
            Computes the gower difference between a point x and all other observed points in sample.
    """

    def __init__(self, func, vals, vars):
        # specify fixed parameters
        self.vals = vals 
        
        # specify variable parameters
        self.vars = vars

        # simulation function handler
        self.func = func

    @config.simwrap()
    def run_sim(self,x :dict) -> list:
        """
        Runs the quantum network simulation with the provided parameters.

        Args:
            x (dict): Parameters for the simulation.

        Returns:
            list: Results of the simulation.
        """
        
        xrun = {**x, **self.vals}
        res = self.func(**xrun)
        return res
    
    def get_random_x(self,n, use_list=False) -> dict:
        """
        Generates random parameters for the simulation.

        Args:
            n (int): Number of random parameter sets to generate.

        Returns:
            dict: Randomly generated parameters.
        """
        
        assert all(isinstance(val, tuple) for val in self.vars['range'].values()) and n > 0, f"Dimension types must be a tuple (sample-list, dataype) and n must be greater zero."

        x = {}
        for dim, par in self.vars['range'].items():
                vals = par[0]
                if par[1] == 'int':
                    x[dim] = np.random.randint(vals[0], vals[1], n) if n > 1 or use_list else np.random.randint(vals[0], vals[1])
                elif par[1] == 'float':
                    x[dim] = np.random.uniform(vals[0], vals[1], n) if n > 1 or use_list else np.random.uniform(vals[0], vals[1])
                else:
                    raise Exception('Datatype must be "int" or "float".')
                    
        for dim, vals in self.vars['choice'].items():
                x[dim] = np.random.choice(vals, n) if n > 1 else np.random.choice(vals)       

        return x
    
class Surrogate(Simulation):
    """
    Class for creating and optimizing a surrogate model.

    Args:
        func (function): The simulation function to be used.
        n (int): Number of initial training set samples.

    Attributes:
        X (dict): Training set parameters.
        X_df (pd.DataFrame): DataFrame of training set parameters (used for training and prediction).
        y (list): Results of the simulation for the training set.
        model (class): The regression model to be used.
        mmodel (class): Multi-output regression model.
        build_time (float): Time taken to build the model.
        improvement (list): List of mean improvements.

    Methods:
        suggest_new_x(temp=10, beta_schedule=5, MAXITER=1e3, seed=42):
            Suggests a new set of parameters using machine learning model optimization (either with random force or simulated annealing).

        acquisition():
            Acquisition Function, computes expected improvement over random parameters.

        update(x):
            Updates the model with new data.
    """
    def __init__(self, func,  vals, vars, initial_model_size):
        super().__init__(func, vals, vars)
        np.random.seed(42)
    
        # profiling storage
        self.sim_time = []
        self.build_time = []
        self.optimize_time = []
    
        # multiprocessing
        self.procs = mp.cpu_count()
        if self.procs > initial_model_size:
            self.procs = initial_model_size

        # generate initial training set 
        ## X
        X = self.get_random_x(initial_model_size)
        self.X_df = pd.DataFrame(X).astype(object)
        
        ## y, run quantum network simulation in parallel
        self.y = []
        self.y_std = []

        start = time.time() 
        with get_context("spawn").Pool(processes=self.procs) as pool:
            y_temp = pool.map(self.run_sim, self.X_df.iloc)

        for y_i in y_temp:
            self.y.append(y_i[0])
            self.y_std.append(y_i[1])

        self.sim_time.append(time.time() - start)

        # build model
        start = time.time()
        self.model = SVR
        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel_std = MultiOutputRegressor(self.model())
        
        self.mmodel.fit(self.X_df.values, self.y)
        self.mmodel_std.fit(self.X_df.values, self.y_std)
        self.build_time.append(time.time()-start)

        # storage
        self.improvement = []

    def get_neighbour(self, MAXITER, count, x :dict) -> dict:
        """
        Generates random parameters for the simulation.

        Args:
            n (int): Number of random parameter sets to generate.

        Returns:
            dict: Randomly generated parameters.
        """
        
        x_n = {}
        f = (1-np.log(1+count/MAXITER)**2)**4

        size = 1000*(count+1)//MAXITER
        for dim, par in self.vars['range'].items():
                vals = par[0]
                if par[1] == 'int':
                    std = f * (vals[1] - vals[0])/2
                    x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=size).astype(int)
                elif par[1] == 'float':
                    std = f * (vals[1] - vals[0])/2
                    x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std, loc=x[dim], scale=std, size=size) 
                else:
                    raise Exception('Datatype must be "int" or "float".')
                    
        for dim, vals in self.vars['choice'].items():
                x_n[dim] = np.random.choice(vals, size)       

        samples_x = pd.DataFrame(x_n).astype(object)
        samples_y = self.mmodel.predict(samples_x.values)
        fittest_neighbour_index = np.argsort(np.array(samples_y).mean(axis=1))[-1]
        
        x_fittest = samples_x.iloc[fittest_neighbour_index].to_dict()
        return x_fittest


    def acquisition(self,MAXITER,count) -> pd.DataFrame:
        """
        Computes new data points according to estimated improvement and degree of exploration and adds the data to the training sample.

        """
        y_obj_vec = np.array(self.y).mean(axis=1)


        newx = []
        for x in self.X_df.iloc[np.argsort(y_obj_vec)[-10:]].iloc:
            newx.append(self.get_neighbour(MAXITER=MAXITER, count=count, x=x.to_dict()))
        self.X_df_add = pd.DataFrame.from_records(newx).astype(object)

        self.X_df = pd.concat([self.X_df, self.X_df_add], axis=0, ignore_index=True)


    
    def update(self) -> None:
        """
        Updates the model with new data.

        Returns:
            None
        """

        start = time.time() 
        with get_context("spawn").Pool(processes=10) as pool:
            y_temp = pool.map(self.run_sim, self.X_df_add.iloc)
        self.sim_time.append(time.time() - start)

        # calculate improvement of new data point to previous best observed point
        impr = np.max([np.mean(y_i[0]) for y_i in y_temp])-np.max([np.mean(y_i) for y_i in self.y])
        self.improvement.append(impr)

        # add new data
        for y_i in y_temp:
            self.y.append(y_i[0])
            self.y_std.append(y_i[1])
        
        # update surrogate model
        start = time.time()
        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel_std = MultiOutputRegressor(self.model())
        
        self.mmodel.fit(self.X_df.values, self.y)
        self.mmodel_std.fit(self.X_df.values, self.y_std)
        self.build_time.append(time.time()-start)

    def optimize(self, MAXITER, verbose=False) -> None:

        for iter in range(MAXITER):
            start = time.time()
            self.acquisition(MAXITER,iter)
            self.optimize_time.append(time.time()-start)
            self.update()
            if verbose: print('{} % done'.format((iter+1)/MAXITER*100))

        if verbose: print('Optimization finished.')