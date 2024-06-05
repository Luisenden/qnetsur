import glob
import argparse
import pandas as pd

def get_policies(folder):
    dfs = []
    for i,name in enumerate(glob.glob(f'{folder}/RS_*.pkl')): 
        with open(name,'rb') as file: dfs.append(pd.read_pickle(file))
        dfs[i]['Trial'] = i
    df = pd.concat(dfs, axis=0)
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="set method")
    parser.add_argument("--method", type=str, default='Surrogate', 
                        help="Choose one of the following methods: 'Surrogate', 'Meta', 'Simulated Annealing', 'Random Search'")
    args, _ = parser.parse_known_args()

    folder = '../../surdata/cd/'

    res = get_policies(folder)
    print(res)
    
    # vals['N_samples'] = 1
    # users = vals['user'][0]
    # vals.pop('user')

    # nprocs = mp.cpu_count()
    # x = xs[args.method]

    # dfs = []
    # seed_count = 1
    # while True:
    #     sim = s
    #     start = time.time()
    #     res = sim.run_exhaustive(x=x, N=nprocs, seed=seed_count)
    #     print(time.time()-start)

    #     df = to_dataframe(res, users=users)
    #     df['Method'] = args.method
    #     dfs.append(df)

    #     seed_count += 1
    #     if len(dfs)*nprocs >= 10:
    #         break
    
    # df_exhaustive = pd.concat(dfs, axis=0)
    # result_folder = folder+f'Results_cd_compare{args.method}.csv'
    # df_exhaustive.to_csv(result_folder) 