import time
import numpy as np
import argparse
from cocoex import Suite, Observer

import sys
# Include parent directory to use qnetsur package
sys.path.append('..')
from qnetsur.utils import Surrogate
from qnetsur.datacollector import SurrogateCollector

def wrapper(cost_function, kwargs: dict):
    x = np.array(list(kwargs.values()))
    return [-cost_function(x)], [0] # one-objective deterministic function (no std deviation)

def sur_optimize(objective, lower, upper, D):
    n = len(lower)
    configurables = { # the optimizer supports all kinds of datatypes
            'range': {},
            'choice':{},
            'ordinal':{}
        }
    for i in range(n):
        configurables['range'][f'x{i}'] = ([lower[i], upper[i]], 'float')

    sur = Surrogate(wrapper, objective, values={}, variables=configurables, initial_training_size=int(1000*np.log10(D)), ntop=int(500*np.log10(D)), rng=np.random.default_rng(7), max_data_size=5000)
    sur.optimize(limit=20, verbose=True, issequential=True)

    # collect and show results
    coll = SurrogateCollector(sur)
    return -max(coll.get_total().objective)

if __name__ == '__main__':

    # parse global params
    parser = argparse.ArgumentParser(description="Import parameter settings")
    parser.add_argument("--D", type=str, default="2", help="Number of dimensions, choose between 2,3,5,10,20,40. Type: str")
    args, _ = parser.parse_known_args()


    solver = sur_optimize
    count = 0   
    suite = Suite("bbob", "year: 2009", "dimensions: " + args.D) # https://numbbo.github.io/coco-doc/C/#suite-parameters
    observer = Observer("bbob", "result_folder: %s_on_%s" % (solver.__name__, "bbob2009"))
    for fun in suite:

        fun.observe_with(observer)
        start = time.time()
        solver(fun, fun.lower_bounds, fun.upper_bounds, int(args.D))
        print(time.time() - start)
        
        count += 1
        print(count)
        if count > 2:
            break


