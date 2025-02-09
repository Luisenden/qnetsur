import time
from cocoex import Suite, Observer
import numpy as np
import argparse

# Ax packaged is used for Bayesian Optimization
from ax.service.ax_client import AxClient, ObjectiveProperties

def meta_optimize(objective, lower, upper):
    # Workflow according to recommended Service API >> https://ax.dev/docs/tutorials/gpei_hartmann_service/
    ax_client = AxClient(verbose_logging=False, random_seed=7)
    ax_client.create_experiment(
        name="benchmarking_experiment",
        parameters=[
            {
                "name": f"x{i}",
                "type": "range",
                "bounds": [loweri, upperi],
                "value_type": "float",  
            } for (i, (loweri, upperi)) in enumerate(zip(lower, upper))
        ],
        objectives={"cost": ObjectiveProperties(minimize=True)}
    )

    n = len(lower)
    for i in range(1000):
        parameterization, trial_index = ax_client.get_next_trial()
        x = np.array([parameterization.get(f"x{i}") for i in range(n)])
        ax_client.complete_trial(trial_index=trial_index, raw_data={"cost": (objective(x), 0.0)})
        print(f"loop {i}")
    
    return ax_client.get_best_trial()[2][0]['cost']


if __name__ == '__main__':
    # parse global params
    parser = argparse.ArgumentParser(description="Import parameter settings")
    parser.add_argument("--D", type=str, default="2", help="Number of dimensions, choose between 2,3,5,10,20,40. Type: str")
    parser.add_argument("--noisy", action='store_true', help="If argument is used, noisy test functions are benchmarked. Type:bool", default=False)
    args, _ = parser.parse_known_args()
    noisy = "-noisy" if args.noisy else "" # add "-noisy" to suite name if noisy functions are used

    suite = Suite("bbob"+noisy, "year: 2009", "dimensions: " + args.D) # https://numbbo.github.io/coco-doc/C/#suite-parameters
    solver = meta_optimize
    observer = Observer("bbob", "result_folder: %s_on_%s" % (solver.__name__, "bbob2009"))

    count = 0   
    for fun in suite:

        fun.observe_with(observer)
        start = time.time()
        solver(fun, fun.lower_bounds, fun.upper_bounds)
        print(time.time() - start)
        
        count += 1
        print(count)