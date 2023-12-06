import numpy as np
from deap.benchmarks import schwefel

def simulation(**kwargs):
    val = schwefel(kwargs.values())[0] + np.random.random_sample*schwefel(kwargs.values())[0]
