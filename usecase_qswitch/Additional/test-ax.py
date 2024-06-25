from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import branin

ax_client = AxClient()
ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [-5.0, 10.0],
            "value_type": "float",
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
    ],
    objective_name="branin",
    minimize=True,
)

for _ in range(15):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=(branin(parameters["x1"], parameters["x2"]), 0.0))

best_parameters_1, metrics = ax_client.get_best_parameters()


from ax.service.managed_loop import optimize
best_parameters_2, values, experiment, model = optimize(
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [-5.0, 10.0],
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
    ],
    evaluation_function=lambda p: (branin(p["x1"], p["x2"]), 0.0),
    minimize=True,
    total_trials=15
)

print("Optimizing with client:", best_parameters_1, metrics)
print("vs optimizing with managed loop:", best_parameters_2, values)