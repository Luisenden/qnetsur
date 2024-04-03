import sys
sys.path.append('../')
sys.path.append('../src')
from src.utils import *
from config import *
from simulation import *
from plotting_tools import *

import warnings
warnings.filterwarnings("ignore")

def get_best_x(df):
    return df.iloc[df['Utility'].idxmax()][df.columns.str.contains('mem_size')]


if __name__ == '__main__':

    folder = '../../surdata/rb'
    result_folder = '../../surdata/rb/Results_starlight_compare.csv'

    df_sur, vals = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_gridsearch(folder)

    xs = []
    for df in [df_sur, df_meta, df_sa, df_gs]:
        x = get_best_x(df)  
        xs.append(x)  
        print(x)

    # even distribution
    even = dict()
    for i in range(9):
        even[f'mem_size_node_{i}'] = 50
    xs.append(even)
    # weighted distribution according to Wu X. et al., 2021
    xs.append({'mem_size_node_0': 25, 'mem_size_node_1': 91, 'mem_size_node_2': 67,
               'mem_size_node_3': 24, 'mem_size_node_4': 67, 'mem_size_node_5': 24, 
               'mem_size_node_6': 103, 'mem_size_node_7': 25, 'mem_size_node_8':24})

    vals['N'] = 1
    nprocs = mp.cpu_count()

    dfs = []
    seed_count = 1
    while True:
        for x, method in zip(xs[-2:], ['Surrogate', 'Meta',
                                'Simulated Annealing', 'Random Gridsearch', 'Even', 'Budget 450'][-2:]):
            sim = Simulation(simwrapper, simulation_rb)
            res = sim.run_exhaustive(x=x, vals=vals, N=nprocs, seed=seed_count)

            df = to_dataframe(res)
            print(df)
            df['Method'] = method
            dfs.append(df)
        seed_count += 1
        if len(dfs)*nprocs >= 10:
            break
    
    df_exhaustive = pd.concat(dfs, axis=0)
    df_exhaustive.to_csv(result_folder) 