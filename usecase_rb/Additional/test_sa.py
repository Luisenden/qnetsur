

from test_config import *
from src.simulatedannealing import * 
import matplotlib.pyplot as plt

booth_function = lambda x,y: (x + 2*y - 7)**2 + (2*x +  - 5)**2 # booth benchmark function (f20 in https://arxiv.org/pdf/1308.4008.pdf)

si = Simulation(test_simwrapper, booth_function, vals=vals, vars=vars)
result = simulated_annealing(si, MAX_TIME=2, beta_schedule=10, seed=12)

result = pd.DataFrame.from_records(result)

plt.plot(result['objective'])
plt.hlines(0, xmin=0, xmax=len(result), linestyles='--', colors='tab:red')
plt.xscale('log')
plt.yscale('symlog')
plt.ylabel('Objective')
plt.xlabel('Iteration')
plt.show()