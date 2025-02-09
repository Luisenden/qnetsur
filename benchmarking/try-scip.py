from pyscipopt import Model
import numpy as np
from cocoex import Suite, Observer


suite = Suite("bbob", "year: 2009", "dimensions: 10")

xtrial = np.random.rand(10)
fun = suite.get_problem("bbob_f001_i01_d10")

def problem_wrapper_for_scip(x):
    x.current
    return x @ x

scip = Model()

n = 10
xvars = np.zeros(n, dtype=object) # dtype object allows arbitrary storage
for i in range(n):
        xvars[i] = scip.addVar(vtype='C', name=f"x_{i}")

obj = scip.addVar(vtype='C', name='z') # This will be our replacement objective variable
cons = scip.addCons(obj >= fun(xvars), name="cons")
scip.setObjective(obj, sense='minimize')
scip.optimize()

print(scip.getVal(obj))