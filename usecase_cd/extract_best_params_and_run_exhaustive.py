import sys
sys.path.append('../')
sys.path.append('../src')
from src.utils import *
from config import *
from optimizingcd import main_cd as simulation
from plotting_tools import *

import warnings
warnings.filterwarnings("ignore")

parser.add_argument("--method", type=str, default='Surrogate', 
                    help="Choose one of the following methods: 'Surrogate', 'Meta', 'Simulated Annealing', 'Random Search'")
args, _ = parser.parse_known_args()
METHOD = args.method


if __name__ == '__main__':

    folder = '../../surdata/cd'

    _, xs, vals = get_policies(folder)
    
    vals['N_samples'] = 1
    users = vals['user'][0]
    vals.pop('user')

    nprocs = mp.cpu_count()
    x = xs[METHOD]

    dfs = []
    seed_count = 1
    while True:
        sim = Simulation(simwrapper, simulation.simulation_cd)
        start = time.time()
        res = sim.run_exhaustive(x=x, vals=vals, N=nprocs, seed=seed_count)
        print(time.time()-start)

        df = to_dataframe(res, users=users)
        df['Method'] = METHOD
        dfs.append(df)

        seed_count += 1
        if len(dfs)*nprocs >= 1000:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    result_folder = f'../../surdata/cd/Results_cd_compare{METHOD}.csv'
    df_exhaustive.to_csv(result_folder) 