import time
from cocoex import Suite, Observer
import numpy as np

# Ax packaged is used for Bayesian Optimization
from ax.service.ax_client import AxClient, ObjectiveProperties

def ax_optimize(objective, lower, upper):
    
    ax_client = AxClient(verbose_logging=False, random_seed=7)
    ax_client.create_experiment(
        name="benchmarking_experiment",
        parameters=[
            {
                "name": f"x{i}",
                "type": "range",
                "bounds": [loweri, upperi],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            } for (i, (loweri, upperi)) in enumerate(zip(lower, upper))
        ],
        objectives={"cost": ObjectiveProperties(minimize=True)},
    )

    n = len(lower)
    for _ in range(int(1e3)):
        parameterization, trial_index = ax_client.get_next_trial()
        x = np.array([parameterization.get(f"x{i}") for i in range(n)])
        ax_client.complete_trial(trial_index=trial_index, raw_data={"cost": (objective(x), 0.0)})
    
    return ax_client.get_best_trial()[2][0]['cost']


if __name__ == '__main__':
    suite = Suite("bbob", "year: 2009", "dimensions: 1 instance_indices: 1-3") # https://numbbo.github.io/coco-doc/C/#suite-parameters
    solver = ax_optimize
    observer = Observer("bbob", "result_folder: %s_on_%s" % (solver.__name__, "bbob2009"))

    count = 0   
    for fun in suite:

        fun.observe_with(observer)
        start = time.time()
        solver(fun, fun.lower_bounds, fun.upper_bounds)
        print(time.time() - start)
        
        count += 1
        if count>2:
            break
        print(count)