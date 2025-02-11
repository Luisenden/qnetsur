In order to run the qswitch simulation in [NetSquid](https://netsquid.org) `netsquid==0.10.3` and the [qswitch snippet](https://github.com/Luisenden/netsquid-qswitch), `version 1.1.1` are required, using Python 3.7. Clone this repository and navigate to `usecase_qswitch` and execute what follows from this directory.

**notebook_qswitch_vardoyan**
Execute `python surrogate_vardoyan_netsquid_comparison.py --serverdist <distance> --iterator 100` (~3-10h runtime per distance) for distances [1, 12, 23, 34, 45, 56, 67, 78, 89, 100].
**qswitch_30min**
Using the parameter settings listed in SIMULATION_INPUT_VALUES.csv we executed for seeds 1-10 (30min runtime in total)

```python
python surrogate.py --nleaf 6 --time 0.5 --seed 1 #generating all "SU_*.csv" files
python vs_meta.py --nleaf 6 --time 0.5 --seed 1 #generating all "AX_*.csv" files
python vs_simulatedannealing.py --nleaf 6 --time 0.5 --seed 1 #generating all "SA_*.csv" files
python vs_randomsearch.py --nleaf 6 --time 0.5 --seed 1 #(generating all "RS_*.csv" files.
``` 

Then, after successful execution the best found solution of the above is retrieved and the simulation is run with the latter solution using
`python run_sim_large.py --method Surrogate`, `python run_sim_large.py --method Meta`, `python run_sim_large.py --method 'Random Search`, `python run_sim_large.py --method 'Simulated Annealing'`