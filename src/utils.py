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

USE_CASE = os.environ.get('USE_CASE') or 'usecase_cd'  # Default to usecase_cd if not set

try:
    config = import_module(f'{USE_CASE}.config_vardoyan')

except ImportError:
    raise ImportError(f"Cannot import config.py for '{USE_CASE}'")


class Simulation:
    """
    Class for running quantum network simulations.

    Attributes:
        vals (dict): A dictionary of fixed parameters.
        vars (dict): A dictionary of variable parameters.
        func (function): The simulation function.

    Methods:
        run_sim(x): Runs the simulation with the provided parameters.
            Args:
                x (dict): Parameters for the simulation.
            Returns:
                list: Results of the simulation.

        get_random_x(n): Generates random parameters for the simulation.
            Args:
                n (int): Number of random parameter sets to generate.
            Returns:
                dict: Randomly generated parameters.
    """

    def __init__(self, sim_wrapper, sim):
        # specify fixed parameters
        self.vals = config.vals
        
        # specify variable parameters
        self.vars = config.vars

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
        
        assert all(isinstance(val, tuple) for val in self.vars['range'].values()) and n > 0, f"Dimension types must be a tuple (sample-list, dataype) and n must be greater zero."

        x = {}
        for dim, par in self.vars['range'].items():
                vals = par[0]
                if par[1] == 'int':
                    x[dim] = np.random.randint(vals[0], vals[1], n) if n > 1 else np.random.randint(vals[0], vals[1])
                elif par[1] == 'float':
                    x[dim] = np.random.uniform(vals[0], vals[1], n) if n > 1 else np.random.uniform(vals[0], vals[1])
                else:
                    raise Exception('Datatype must be "int" or "float".')

        for dim, vals in self.vars['ordinal'].items():
                x[dim] = np.random.choice(vals, size=n) if n > 1 else np.random.choice(vals)
                    
        for dim, vals in self.vars['choice'].items():
                x[dim] = np.random.choice(vals, n) if n > 1 else np.random.choice(vals)       

        return x
    
class Surrogate(Simulation):
    """
    Class for creating and optimizing a surrogate model.

    Args:
        func (function): The simulation function to be used.
        sample_size (int): Number of initial training set samples.

    Attributes:
        X (dict): Training set of parameters.
        X_df (pd.DataFrame): DataFrame of training set parameters (used for training and prediction).
        y (list): Results of the simulation for the training set.
        y_std (list): Standard deviation of results for the training set.
        model (class): The regression model to be used; Currently Support Vector Regression (sklearn.svm.SVR)
        mmodel (class): Multi-output regression model.
        mmodel_std (class): Multi-output regression model for standard deviation.
        build_time (list): Time taken to build the model.
        sim_time (list): Time taken for simulations.
        optimize_time (list): Time taken for optimization steps.
        procs (int): Number of CPU cores available.
        improvement (list): List of mean improvements.

    Methods:
        get_neighbour(max_time, current_time, x): Generates a neighboring parameter set.
            Args:
                max_time (float): Maximum time allowed for optimization.
                current_time (float): Current optimization time.
                x (dict): Current parameter set.
            Returns:
                dict: Neighboring parameter set.

        acquisition(max_time, current_time): Computes new data points according to expected improvement and exploration degree and adds it to the the training set.
            Args:
                max_time (float): Maximum time allowed for optimization.
                current_time (float): Current optimization time.

        update(): Updates the model with new data.

        optimize(max_time, verbose=False): Initiates the optimization process.
            Args:
                max_time (float): Maximum time allowed for optimization.
                verbose (bool, optional): Specifies whether to display optimization information.
    """
    def __init__(self, sim_wrapper, sim, sample_size:int):
        super().__init__(sim_wrapper, sim)
    
        # profiling storage
        self.sim_time = []
        self.build_time = []
        self.optimize_time = []
    
        # multiprocessing
        self.procs = mp.cpu_count()
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

        # improvement storage
        self.improvement = []

    def get_neighbour(self, max_time, current_time, x :dict) -> dict:
        """
        Generates parameters in limited neighboring region for the simulation.

        Args:
            n (int): Number of random parameter sets to generate.

        Returns:
            dict: Randomly generated parameters.
        """
        
        x_n = {}
        f = (1-np.log(1+current_time/max_time)**2)**6

        size = int(current_time/max_time * 10000+10)
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

        for dim, vals in self.vars['ordinal'].items():
                pos = x[dim] # current position
                
                loc = vals.index(pos)/len(vals) # corresponding location between 0 and 1
                pval = np.linspace(0,1,len(vals))

                std = f/2
                probs = truncnorm.pdf(pval, (0-loc)/std, (1-loc)/std, scale=std, loc=loc)/len(x) # corresponding weights
                probs /= probs.sum() # normalize probabilities

                x_n[dim] = np.random.choice(vals, size=size , p=probs)
                    
        for dim, vals in self.vars['choice'].items():
                x_n[dim] = np.random.choice(vals, size)       

        samples_x = pd.DataFrame(x_n).astype(object)
        samples_y = self.mmodel.predict(samples_x.values)
        fittest_neighbour_index = np.argsort(np.array(samples_y).sum(axis=1))[-1] ## !!
        
        x_fittest = samples_x.iloc[fittest_neighbour_index].to_dict()
        return x_fittest


    def acquisition(self,max_time,current_time) -> pd.DataFrame:
        """
        Computes new data points according to estimated improvement and degree of exploration and adds the data to the training sample.

        """

        y_obj_vec = np.array(self.y).sum(axis=1)


        newx = []
        top_selection = self.X_df.iloc[np.argsort(y_obj_vec)[-3:]]


        for x in top_selection.iloc:  ## !!
            neighbour = self.get_neighbour(max_time=max_time, current_time=current_time, x=x.to_dict())
            newx.append(neighbour)
        
        self.X_df_add = pd.DataFrame.from_records(newx).astype(object)


    
    def update(self, counter) -> None:
        """
        Updates the model with new data.

        Returns:
            None
        """

        start = time.time() 
        with Pool(processes=3) as pool:
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
            yi,yi_std,*yi_raw = y_i
            self.y.append(yi)
            self.y_std.append(yi_std)
            self.y_raw += yi_raw
        
        # train/update surrogate model
        score_svr = cross_val_score(MultiOutputRegressor(SVR()), self.X_df.drop('Iteration', axis=1).values, self.y, scoring=make_scorer(mean_absolute_error)).mean()
        score_tree = cross_val_score(MultiOutputRegressor(DecisionTreeRegressor()), self.X_df.drop('Iteration', axis=1).values, self.y, scoring=make_scorer(mean_absolute_error)).mean()
        
        if score_svr < score_tree:
             self.model = SVR
        else:
             self.model = DecisionTreeRegressor
        print(f'scores: svr = {score_svr} and tree={score_tree}')
        self.model_scores['SVR'].append(score_svr)
        self.model_scores['DecisionTree'].append(score_tree)
        
        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel_std = MultiOutputRegressor(self.model())

        self.mmodel.fit(self.X_df.drop('Iteration', axis=1).values, self.y)
        self.mmodel_std.fit(self.X_df.drop('Iteration', axis=1).values, self.y_std)
        self.build_time.append(time.time()-start)

    def optimize(self, max_time, verbose=False) -> None:
    
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

        score_svr = cross_val_score(MultiOutputRegressor(SVR()), self.X_df.drop('Iteration', axis=1).values, self.y, scoring=make_scorer(mean_absolute_error)).mean()
        score_tree = cross_val_score(MultiOutputRegressor(DecisionTreeRegressor()), self.X_df.drop('Iteration', axis=1).values, self.y, scoring=make_scorer(mean_absolute_error)).mean()
        
        print(f'scores: svr = {score_svr} and tree={score_tree}')
        self.model_scores['SVR'].append(score_svr)
        self.model_scores['DecisionTree'].append(score_tree)

        if score_svr < score_tree:
             self.model = SVR
        else:
             self.model = DecisionTreeRegressor

        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel_std = MultiOutputRegressor(self.model())

        self.mmodel.fit(self.X_df.drop('Iteration', axis=1).values, self.y)
        self.mmodel_std.fit(self.X_df.drop('Iteration', axis=1).values, self.y_std)
        
        self.build_time.append(time.time()-start)

        initial_optimize_time = time.time()-optimize_start
        self.optimize_time.append(initial_optimize_time)

        # optimization
        max_optimize_time = max_time - initial_optimize_time

        if verbose and max_optimize_time < 0: print("Initial model generated, but no time left for optimization after initial build.")

        if verbose: print(f'After initial build, time left for optimization: {max_optimize_time:.2f}s')
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
            if verbose: print(f'Time left for optimization: {max_optimize_time-current_time:.2f}s')

        if verbose: print('Optimization finished.')

def get_parameters(vars:dict):
    parameters = []
    for k in vars:
        for key,value in vars[k].items():
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