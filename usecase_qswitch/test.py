# from ax import optimize
# best_parameters, best_values, experiment, model = optimize(
#         parameters=[
#           {
#             "name": "x1",
#             "type": "range",
#             "bounds": [-10.0, 10.0],
#           },
#         ],
#         # Booth function
#         evaluation_function=lambda p: (p["x1"] - 7)**2 + (2*p["x1"] - 5)**2,
#         minimize=True,
#     )
# print(best_parameters)

from ax import optimize
from ax.utils.measurement.synthetic_functions import branin

best_parameters, values, experiment, model = optimize(
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
)