import sys
sys.path.append('../')
sys.path.append('../src')
from src.utils import *
from config import *
from simulation import *
from plotting_tools import *

import warnings
warnings.filterwarnings("ignore")

parser.add_argument("--method", type=str, default='Surrogate', 
                    help="Choose one of the following methods: 'Surrogate', 'Meta', 'Simulated Annealing', 'Random Search', 'Even', 'Wu et. al, 2021'")
args, _ = parser.parse_known_args()
METHOD = args.method


if __name__ == '__main__':

    folder = '../../surdata/rb_budget'

    _, xs, vals = get_best_paremters(folder)
    
    vals['N'] = 1
    nprocs = mp.cpu_count()
    x = xs[METHOD]

    # x_df = pd.DataFrame.from_records(xs).T
    # print(x_df.T.to_latex())

    dfs = []
    seed_count = 1
    while True:
        sim = Simulation(simwrapper, simulation_rb)
        start = time.time()
        res = sim.run_exhaustive(x=x, vals=vals, N=nprocs, seed=seed_count)
        print(time.time()-start)

        df = to_dataframe(res)
        df['Method'] = METHOD
        dfs.append(df)

        seed_count += 1
        if len(dfs)*nprocs >= 10:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)

    result_folder = f'../../surdata/rb_budget/Results_starlight_compare{METHOD}.csv'
    df_exhaustive.to_csv(result_folder) 