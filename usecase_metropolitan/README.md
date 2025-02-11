In order to run the network simulation, the python module [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe), where we used a Python 3.9 environment.
Installation e.g., with `pip install git+https://github.com/sequence-toolbox/SeQUeNCe.git`. This time we executed what follows from `usecase_metropolitan`.

**rb_25h**
Similar to above, we use the parameter settings listed in SIMULATION_INPUT_VALUES.csv and executed for seeds 1-10 for all methods.
```python
python surrogate.py --time 25 --seed 1 #generating the "SU_*.csv" file 
...
```

After retrieving the best found solutions (Best_found_solutions.csv), we again simulated for n=1000 runs
`python run_sim_large.py --method Surrogate`, `python run_sim_large.py --method Meta`, `python run_sim_large.py --method 'Random Search`, `python run_sim_large.py --method 'Simulated Annealing'`.
We further run the simulation with the policies used in the associated [sequence paper](https://iopscience.iop.org/article/10.1088/2058-9565/ac22f6/pdf)
`python run_sim_large.py --method Even`, `python run_sim_large.py --method 'Wu et. al, 2021'`.
