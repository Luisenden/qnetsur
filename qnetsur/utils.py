"""
Central script containing objects for surrogate optimization and helper functions.
"""

import numpy as np
import time
import pandas as pd
from scipy.stats import truncnorm
from functools import partial
import multiprocessing as mp

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, make_scorer

class Simulation:
    """
    Object to run quantum network simulations. 
    It allows for the specification of both fixed and variable simulation parameters, 
    supports the generation of random parameter sets, and facilitates the running of simulations 
    through a user-defined simulation function.

    Parameters
    ----------
    sim_wrapper : function
        The wrapper function that preprocesses simulation
        parameters before execution and adapts outputs according to needs of use case.
    sim : function
        The actual simulation function to be executed.
    rng : class
        Random number generator object to be used.
    values : dict
        Fixed parameters for the simulation.
    variables : dict, optional
        Variable parameters for the simulation, allowing for dynamic adjustments.

    Methods
    -------
    run_sim(x, vals=None)
        Executes a single simulation run with the given parameters.
    run_exhaustive(x, N=10, seed=42)
        Executes `N` simulations in parallel.
    get_random_x(n)
        Generates `n` sets of random parameters based on the
        specified variable parameters.
    """

    def __init__(self, sim_wrapper, sim, rng, values, variables):
        
        self.rng = rng
        # specify fixed parameters
        self.vals = values
        # specify variable parameters
        self.vars = variables

        # simulation function handler
        self.sim_wrapper = sim_wrapper
        # simulation function
        self.sim = sim

    def run_sim(self, x :dict) -> list:
        """
        Runs the quantum network simulation with the provided parameters.

        Args:
            x (dict): Parameters for the simulation.

        Returns:
            list: Results of the simulation.
        """
        xrun = {**self.vals, **x} 
        res = self.sim_wrapper(self.sim, xrun)
        return res

    def run_exhaustive(self, x :dict, N=10, seed=42) -> list:
        """
        Runs the quantum network simulation with the provided parameters N times in parallel.
        """
        xrun = {**x, **self.vals}
        task = partial(self.sim_wrapper, self.sim)
        with mp.Pool(processes=N) as pool:
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
                    x[dim] = self.rng.integers(vals[0], vals[1], n) if n > 1\
                        else self.rng.integers(vals[0], vals[1])
                elif par[1] == 'float':
                    x[dim] = self.rng.uniform(vals[0], vals[1], n) if n > 1\
                        else self.rng.uniform(vals[0], vals[1])
                else:
                    raise Exception('Datatype must be "int" or "float".')

        for dim, vals in self.vars['ordinal'].items():
                x[dim] = self.rng.choice(vals, size=n) if n > 1\
                    else self.rng.choice(vals)
                    
        for dim, vals in self.vars['choice'].items():
                x[dim] = self.rng.choice(vals, n) if n > 1\
                    else self.rng.choice(vals)       

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
    initial_training_size : int
        The number of configurations to use for the initial surrogate model training.
    ntop : int
        The number of configurations to use in acquisition process
    degree : int, optional
        Exponent in acquisition transition function (exploitation degree).

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
    acquisition(max_T, current_t)
        Selects new parameters for evaluation by balancing exploration and exploitation.
    update()
        Updates the surrogate model with new data points collected from simulation runs.
    optimize(max_T)
        Conducts the optimization process to find optimal simulation parameters.
    """
    def __init__(self, sim_wrapper, sim, rng, values, variables, initial_training_size, ntop, degree=4):
        super().__init__(sim_wrapper, sim, rng, values, variables)

        assert initial_training_size>=5, f"Sample size must be at least 5 (requirement for 5-fold cross validation)."

        # storage for time profiling
        self.sim_time = []
        self.build_time = []
        self.acquisition_time = []
        self.optimize_time = []

        # set multiprocessing
        self.procs = min(mp.cpu_count(),ntop)
        print('Available processors for parallel execution: ', self.procs)
        self.init_size = initial_training_size
        self.ntop = ntop

        # storage target value
        self.y = []
        self.y_std = []
    
        # storage raw values
        self.y_raw = []

        # storage model declaration
        self.model = SVR()
        self.model_scores = {'SVR': [], 'DecisionTree': []}
        self.degree = degree  # coefficient in neighbour selection

    def get_neighbour(self, x :dict) -> dict:
        """
        Generates most promising parameters in limited neighbouring region 
        according to current knowledge of surrogate model and depending on the time left.
        """
        x_n = {}
        if self.isscore:
             f_svr = (1-np.log(1+(self.model_scores['SVR'][0]-self.model_scores['SVR'][-1])/self.model_scores['SVR'][0])**2)**self.degree
             f_tree  = (1-np.log(1+(self.model_scores['DecisionTree'][0]-self.model_scores['DecisionTree'][-1])/self.model_scores['DecisionTree'][0])**2)**self.degree
             f = f_svr if f_svr < f_tree else f_tree
             size = int((1-f)*10000 + 10)
        else:
            f = (1-np.log(1+self.current_time_counter/self.limit)**2)**self.degree
            size = int(self.current_time_counter/self.limit*10000 + 10)

        for dim, par in self.vars['range'].items():
                vals = par[0]
                if par[1] == 'int':
                    std = f * (vals[1] - vals[0])/2
                    x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std,
                                             loc=x[dim], scale=std, size=size, random_state=self.rng).astype(int)
                elif par[1] == 'float':
                    std = f * (vals[1] - vals[0])/2
                    x_n[dim] = truncnorm.rvs((vals[0] - x[dim]) / std, (vals[1] - x[dim]) / std,
                                             loc=x[dim], scale=std, size=size, random_state=self.rng) 
                else:
                    raise Exception('Datatype must be "int" or "float".')

        for dim, vals in self.vars['ordinal'].items():
                pos = x[dim]  # current position
                loc = vals.index(pos)/len(vals)  # corresponding location between 0 and 1
                pval = np.linspace(0,1,len(vals))
                std = f/2
                probs = truncnorm.pdf(pval,
                                      (0-loc)/std, (1-loc)/std, scale=std,
                                      loc=loc, random_state=self.rng)/len(x)  # corresponding weights
                probs /= probs.sum()  # normalize probabilities
                x_n[dim] = self.rng.random.choice(vals, size=size , p=probs)

        for dim, vals in self.vars['choice'].items():
                x_n[dim] = self.rng.random.choice(vals, size)

        # select best prediction as next neighbour
        samples_x = pd.DataFrame(x_n).astype(object)
        samples_y = self.mmodel.predict(samples_x.values)
        fittest_neighbour_index = np.argsort(np.array(samples_y).sum(axis=1))[-1]
        x_fittest = samples_x.iloc[fittest_neighbour_index].to_dict()
        return x_fittest

    def acquisition(self) -> None:
        """
        Computes n new data points according to estimated improvement 
        and degree of exploration and adds the data to the training sample.
        """
        start = time.time()
        y_obj_vec = np.sum(self.y, axis=1)
        newx = []
        top_selection = self.X_df.iloc[np.argsort(y_obj_vec)[-self.ntop:]]  # select top n candidates
        for x in top_selection.iloc:  # get most promising neighbour according to surrogate
            neighbour = self.get_neighbour(x=x.to_dict())
            newx.append(neighbour)
        self.X_df_add = pd.DataFrame.from_records(newx).astype(object)
        self.acquisition_time.append(time.time()-start)      


    def run_multiple_and_add_target_values(self, X) -> None:
        start = time.time()
        if self.issequential:
            y_temp = []
            for x in X:
                y_temp.append(self.run_sim(x))
        else:
            with mp.Pool(processes=self.procs) as pool:
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
        print(f'MAE: svr = {score_svr} and tree={score_tree}')
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


    def gen_initial_set(self) -> None:
        """
        Generates the initial training set.
        """
        if self.verbose: print("Start optimization ...")
        optimize_start = time.time()
        
        # generate initial training set X
        start = time.time()
        X = self.get_random_x(self.init_size)
        self.X_df = pd.DataFrame(X).astype(object)
        self.build_time.append(time.time() - start)

        self.run_multiple_and_add_target_values(X=self.X_df.iloc)
        self.X_df['Iteration'] = 0

        # train models
        self.train_models()

        self.initial_optimize_time = time.time()-optimize_start
        self.optimize_time.append(self.initial_optimize_time)
        if self.verbose and isinstance(self.limit, int): print(f'Iteration 0/{self.limit}')


    def optimize_with_timer(self) -> None:
        """
        Optimization until a set maximum number of seconds.
        """
        current_times = []
        self.current_time_counter = 0
        self.max_optimize_time = self.limit - self.initial_optimize_time
        delta = 0
        counter = 1
        while (self.current_time_counter + delta) < self.max_optimize_time:
            start = time.time()

            self.acquisition()
            self.update(counter)

            self.optimize_time.append(time.time()-start)
            current_times.append(time.time()-start)
            self.current_time_counter = np.sum(current_times)
            delta = np.mean(current_times)
            counter +=1
            if self.verbose:
                print(f'Time left for optimization: {self.max_optimize_time-self.current_time_counter:.2f}s')

    def optimize_with_iteration(self) -> None:
        """
        Optimization until a set maximum number of iterations.
        """
        self.current_time_counter = 1
        while self.current_time_counter <= self.limit:
            start = time.time()

            self.acquisition()
            self.update(self.current_time_counter)

            self.optimize_time.append(time.time()-start)
            if self.verbose: print(f'Iteration {self.current_time_counter}/{self.limit}')
            self.current_time_counter +=1


    def optimize(self, limit, isscore=False, issequential=False, verbose=False) -> None:
        """
        Conducts the optimization process to find optimal simulation parameters.
        """
        self.limit = limit
        self.verbose = verbose
        self.isscore = isscore
        self.issequential = issequential
        if isinstance(limit, float):
            self.limit *= 3600 # to seconds
            if self.verbose: print(f'Optimize with timer. LIMIT: {self.limit} seconds.')
            self.gen_initial_set()
            self.optimize_with_timer()
        else:
            if self.verbose: print(f'Optimize with iterator. LIMIT: {self.limit} cycles.')
            self.gen_initial_set()
            self.optimize_with_iteration()
        if self.verbose: print('Optimization finished.')