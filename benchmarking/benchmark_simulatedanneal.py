import time
import numpy as np
import argparse
from cocoex import Suite, Observer
import pandas as pd

import sys
# Include parent directory to use qnetsur package
sys.path.append('..')
from qnetsur.utils import Simulation
from qnetsur.simulatedannealing import simulated_annealing 

def wrapper(cost_function, kwargs: dict):
    x = np.array(list(kwargs.values()))
    return [-cost_function(x)], [0] # one-objective deterministic function (no std deviation)

def simanneal_optimize(objective, lower, upper):
    n = len(lower)
    configurables = { # the optimizer supports all kinds of datatypes
            'range': {},
            'choice':{},
            'ordinal':{}
        }
    for i in range(n):
        configurables['range'][f'x{i}'] = ([lower[i], upper[i]], 'float')

    # run simulated annealing
    sim = Simulation(wrapper, objective, values={}, variables=configurables, rng=np.random.default_rng(7))
    result = simulated_annealing(sim, limit=0.1)

    result = pd.DataFrame.from_records(result)
    print(result)
    return -max(result['objective'])

if __name__ == '__main__':

    # parse global params
    parser = argparse.ArgumentParser(description="Import parameter settings")
    parser.add_argument("--D", type=str, default="2", help="Number of dimensions, choose between 2,3,5,10,20,40. Type: str")
    parser.add_argument("--noisy", action='store_true', help="If argument is used, noisy test functions are benchmarked. Type:bool", default=False)
    args, _ = parser.parse_known_args()
    noisy = "-noisy" if args.noisy else "" # add "-noisy" to suite name if noisy functions are used

    solver = simanneal_optimize  
    suite = Suite("bbob"+noisy, "year: 2009", "dimensions: " + args.D) # https://numbbo.github.io/coco-doc/C/#suite-parameters
    observer = Observer("bbob", "result_folder: %s_on_%s" % (solver.__name__, "bbob2009"))
    for fun in suite:

        fun.observe_with(observer)
        start = time.time()
        solver(fun, fun.lower_bounds, fun.upper_bounds)
        print(time.time() - start)

