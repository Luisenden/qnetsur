import numpy as np
import time
import pandas as pd
from multiprocessing import Pool


from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import ExtraTreeRegressor

from gower import gower_matrix


from main_cd import main_cd as main
from SA_optimize import simulated_annealing
from brute_optimize import brute_optimize


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

    def __init__(self, func, vals, vars):
        # specify fixed parameters
        self.vals = vals 
        
        # specify variable parameters
        self.vars = vars

        # simulation function handler
        self.func = func
    
    def run_sim(self,x :dict) -> list:
        """
        Runs the simulation with the provided parameters.

        Args:
            x (dict): Parameters for the simulation.

        Returns:
            list: Results of the simulation.
        """
        result = self.func(**x, **self.vals)
        return [node[-1] for node in result[1]]
    
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
        mean_improvement (list): List of mean improvements.

    Methods:
        suggest_new_x(temp=10, beta_schedule=5, MAXITER=1e3, seed=42):
            Suggests a new set of parameters using machine learning model optimization (either with brute force or simulated annealing).

        improvement():
            Acquisition Function, computes expected improvement over random parameters.

        update(x):
            Updates the model with new data.
    """
    def __init__(self, func, vals, vars, n, brute = 1):
        super().__init__(func, vals, vars)
        np.random.seed(42)
    
        # timing storage
        self.optimize_time = 0
        self.sim_time = 0
        self.build_time = 0

        # solver
        self.brute = brute

        # generate initial training set 
        ## X
        self.X = self.get_random_x(n)
        self.X_df = pd.DataFrame(self.X).astype(object)
        
        ## y, run simulation in parallel
        inputs = [input.to_dict() for input in self.X_df.iloc]

        start = time.time() 
        with Pool() as pool:
            self.y = pool.map(self.run_sim, inputs)
        self.sim_time += time.time() - start

        # build model
        start = time.time()
        self.model = ExtraTreeRegressor
        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel.fit(self.X_df.values, self.y)
        self.build_time += time.time()-start

        # storage
        self.mean_improvement = []
        self.opt_vec = []

    def suggest_new_x(self, temp :int = 10, beta_schedule :int = 5, MAXITER = 1e3, seed=42) -> tuple:
        """
        Suggests a new set of parameters using simulated annealing.

        Args:
            temp (int, optional): Initial temperature. Defaults to 10.
            beta_schedule (int, optional): Beta schedule parameter. Defaults to 5.
            MAXITER (float, optional): Maximum number of iterations. Defaults to 1e3.
            seed (int, optional): Random seed. Defaults to 42.

        Returns:
            tuple: Suggested parameters and corresponding value.
        """
        start = time.time()
        x_opt, val_opt =  brute_optimize(self) if self.brute else simulated_annealing(self, temp, beta_schedule, MAXITER, seed)
        self.optimize_time += time.time()-start
        return x_opt, val_opt
        
    def improvement(self) -> tuple:
        """
        Computes new parameter set according to estimated improvement and degree of exploration.

        Returns:
            tuple: Suggested parameters and corresponding value.
        """
        x_rand = self.get_random_x(1)
        diff = np.mean(gower_matrix(pd.DataFrame(x_rand, index=[0]), self.X_df)[0])

        y = None
        if diff < 0.5:
            opt = 1
            x, y = self.suggest_new_x()
            #print('x_new: The model is %.2f. away from reality.' % (y+np.mean( self.run_sim(x_dict) )))
        else:
            opt = 0
            x = x_rand
        return x, y, opt
    
    def update(self,x :dict, opt :bool):
        """
        Updates the model with new data.

        Args:
            x (dict): New parameter set.
            opt (bool): 0 indicates random sample and 1 indicates the point was suggested by optimizer.

        Returns:
            None
        """
        for key in self.X:
            self.X[key] = np.append(self.X[key],x[key])
        
        self.opt_vec.append(opt)

        start = time.time()
        eval_sim = self.run_sim(x)
        self.sim_time = time.time() - start

        self.y.append(eval_sim)

        # calculate mean improvement to previous observed points
        mean_impr = np.mean(eval_sim)-np.mean([np.mean(y_i) for y_i in self.y])
        self.mean_improvement.append(mean_impr)

        self.X_df = pd.DataFrame(self.X)
        
        self.mmodel = MultiOutputRegressor(self.model())
        self.mmodel.fit(self.X_df.values, self.y)

def surrogate_optimize(s :Surrogate, MAXITER=1000, epsilon=1e-2, verbose=False):

    for iter in range(MAXITER):
        x_new, _, opt = s.improvement()
        s.update(x_new, opt)

        if verbose: print('{} % done'.format((iter+1)/MAXITER*100))

        if iter > 10 and np.mean(list(map(abs,s.mean_improvement[-10:]))) < epsilon:
            break

    if verbose: print('Optimization finished.')
