import numpy as np
import time
import pandas as pd
from optimizingcd import main_cd as simulation

import multiprocessing as mp
from multiprocessing import Pool
from specifications import simwrap

from sklearn import preprocessing as pp
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

from gower import gower_matrix


from SA_optimize import simulated_annealing
from random_optimize import random_optimize


class Simulation:
    """
    Class for running quantum network simulations.

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

    def __init__(self, func, topology, vals, vars):
        # specify fixed parameters
        self.vals = vals 
        self.size = topology.size
        self.vals['A'] = simulation.adjacency_squared(self.size[0]) if topology.name == 'square' else simulation.adjacency_tree(self.size[0], self.size[1])
        
        # specify variable parameters
        self.vars = vars
        self.dtypes = pd.DataFrame(self.vars).dtypes.to_dict()


        # simulation function handler
        self.func = func

        # multiprocessing
        self.procs = mp.cpu_count()
        if self.procs > 30:
            self.procs = 30

    @simwrap
    def run_sim(self,x :dict) -> list:
        """
        Runs the simulation with the provided parameters.

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
        
        assert all(isinstance(val, list) for val in self.vars.values()) and n > 0, f"Dimension types must be list and n > 0!"

        x = {}
        for dim, vals in self.vars.items():
            if len(vals) > 2:
                x[dim] = np.random.choice(vals, n) if n > 1 else np.random.choice(vals)
            elif all(isinstance(x, int) for x in vals):
                x[dim] = np.random.randint(vals[0], vals[1], n) if n > 1 or use_list else np.random.randint(vals[0], vals[1])
            elif all(isinstance(x, float) for x in vals):
                x[dim] = np.random.uniform(vals[0], vals[1], n) if n > 1 or use_list else np.random.uniform(vals[0], vals[1])         

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
    def __init__(self, func, topology, vals, vars, initial_model_size):
        super().__init__(func, topology, vals, vars)
        np.random.seed(42)
    
        # profiling storage
        self.sim_time = []
        self.build_time = []
        self.predict_time = []
        self.findmax_time = []

        # generate initial training set 
        ## X
        X = self.get_random_x(initial_model_size)
        self.X_df = pd.DataFrame(X).astype(object)
        
        ## y, run simulation in parallel
        self.y = []
        self.y_std = []

        start = time.time() 
        with Pool(processes=self.procs) as pool:
            y_temp = pool.map(self.run_sim, self.X_df.iloc)

        for y_i in y_temp:
            self.y.append(y_i[0])
            self.y_std.append(y_i[1])

        self.sim_time.append(time.time() - start)

        # build model
        start = time.time()
        self.model = GradientBoostingRegressor
        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel_std = MultiOutputRegressor(self.model())

        self.scaler = pp.MinMaxScaler((0,1))
        data = self.scaler.fit_transform(self.X_df)
        
        self.mmodel.fit(data, self.y)
        self.mmodel_std.fit(data, self.y_std)
        self.build_time.append(time.time()-start)

        # storage
        self.improvement = []
        self.flag_vec = np.zeros(initial_model_size)

    # def suggest_new_x(self) -> dict:
    #     """
    #     Suggests a new set of parameters using acquisition optimization.

    #     Args:
    #         temp (int, optional): Initial temperature. Defaults to 10.
    #         beta_schedule (int, optional): Beta schedule parameter. Defaults to 5.
    #         MAXITER (float, optional): Maximum number of iterations. Defaults to 1e3.
    #         seed (int, optional): Random seed. Defaults to 42.

    #     Returns:
    #         tuple: Suggested parameters and corresponding value.
    #     """
    #     start = time.time()
    #     x_opt =  random_optimize(self) 
    #     self.optimize_time += time.time()-start
    #     return x_opt
        
    def acquisition(self) -> pd.DataFrame:
        """
        Computes new data points according to estimated improvement and degree of exploration and adds the data to training sample.

        """
        limit = 0.97

        x_rand_df = pd.DataFrame(self.get_random_x(self.procs)).astype(object) # newly sampled points
        distances = np.array([np.mean(dists) for dists in gower_matrix(x_rand_df, self.X_df)]) # calculates gower distance between newly sampled points and previously observed points

        # optimize surrogate model and suggest new x for sample points that were close enough
        suggested_x = pd.DataFrame(columns=self.X_df.columns)
        nsuggested = sum(distances<limit)

        flag = np.zeros(len(distances)-nsuggested+1) # keep track of suggested point
        if nsuggested > 0:
            start = time.time()
            suggested_x = random_optimize(self)
            flag[-1] = 1 # add flags
            print('time optimization: ', time.time()-start)

        self.flag_vec = np.append(self.flag_vec, flag)

        self.current_points = pd.concat([x_rand_df[distances>=limit], suggested_x], ignore_index=True ,axis=0)

        # add points
        self.X_df = pd.concat([self.X_df, self.current_points], ignore_index=True ,axis=0)

    
    def update(self) -> None:
        """
        Updates the model with new data.

        Returns:
            None
        """

        start = time.time() 
        with Pool(processes=self.procs) as pool:
            y_temp = pool.map(self.run_sim, self.current_points.iloc)
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

        data = self.scaler.fit_transform(self.X_df)
        
        self.mmodel.fit(data, self.y)
        self.mmodel_std.fit(data, self.y_std)
        self.build_time.append(time.time()-start)

    def optimize(self, solver='random', MAXITER=1000, epsilon=1e-2, verbose=False) -> None:

        # acquisition optimizer (random or simulated annealing)
        self.solver = solver

        for iter in range(MAXITER):
            self.acquisition()
            self.update()

            if verbose: print('{} % done'.format((iter+1)/MAXITER*100))

            if iter > 10 and np.mean(list(map(abs,self.improvement[-10:]))) < epsilon:
                break

        if verbose: print('Optimization finished.')
