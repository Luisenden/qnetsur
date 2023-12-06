import numpy as np
from deap.benchmarks import schwefel

def simulation(x,y):
    return schwefel([x, y])[0] + np.random.rand()*schwefel([x, y])[0]

print(simulation(1,2))