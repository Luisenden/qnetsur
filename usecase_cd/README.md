We simulated continuous protocols using [OptimizingCD](https://github.com/AlvaroGI/optimizing-cd-protocols) within a Python 3.9 environment. Installation via `git+https://github.com/AlvaroGI/optimizing-cd-protocols.git@package`.

**cd_1-10h**
The simulation experiments are conducted in the same manner as above, using the associated scripts in the git directory `usecase_cd`.
```python
python surrogate.py --topo randtree-100 --seed 1 --time i #where i=1,5 and 10 (hours) for seeds 1-10 and all four reference methods
...
python run_sim_large.py --method Surrogate #and accordingly for the three reference methods
...
```
to simulate the overall best found solution using n=1000 runs.

**notebook_cd_n3** and **notebook_cd_n20**
To generate the presented outcomes, we optimize the swap parameters per node using continuous protocols with `python surrogate.py --topo tree-2-1 --seed 42 --iterator 100` and `python surrogate.py --topo randtree-10 --seed 42 --iterator 100` using the parameter settings listed in SIMULATION_INPUT_VALUES.csv.

