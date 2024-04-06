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
                    help="Choose one of the following methods: 'Surrogate', 'Meta', 'Simulated Annealing', 'Random Gridsearch', 'Even', 'Budget 450'")
args, _ = parser.parse_known_args()
METHOD = args.method

def get_best_x(df):
    return df.iloc[df['Utility'].idxmax()][df.columns.str.contains('mem_size|Method')]


if __name__ == '__main__':

    folder = '../../surdata/rb'
    result_folder = '../../surdata/rb/Results_starlight_compare.csv'

    df_sur, vals = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_gridsearch(folder)

    xs = dict()
    for df in [df_sur, df_meta, df_sa, df_gs]:
        xmethod = get_best_x(df)  
        xs[xmethod['Method']] = xmethod.drop('Method')
    

    # even distribution
    even = dict()
    for i in range(9):
        even[f'mem_size_node_{i}'] = 50
    xs['even'] = even
    
    # weighted distribution according to Wu X. et al., 2021
    xs['Budget 450'] = {'mem_size_node_0': 25, 'mem_size_node_1': 91, 'mem_size_node_2': 67,
               'mem_size_node_3': 24, 'mem_size_node_4': 67, 'mem_size_node_5': 24, 
               'mem_size_node_6': 103, 'mem_size_node_7': 25, 'mem_size_node_8':24}
    
    vals['N'] = 1
    nprocs = mp.cpu_count()
    x = xs[METHOD]

    print(x)
    print(vals)

    # dfs = []
    # seed_count = 1
    # while True:
    #     sim = Simulation(simwrapper, simulation_rb)
    #     res = sim.run_exhaustive(x=x, vals=vals, N=nprocs, seed=seed_count)

    #     df = to_dataframe(res)
    #     df['Method'] = METHOD
    #     dfs.append(df)

    #     seed_count += 1
    #     if len(dfs)*nprocs >= 1000:
    #         break
    
    # df_exhaustive = pd.concat(dfs, axis=0)
    # df_exhaustive.to_csv(result_folder) 