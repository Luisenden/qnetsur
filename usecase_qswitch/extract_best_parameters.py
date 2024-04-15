from plotting_tools import *

if __name__ == '__main__':

    folder = '../../surdata/qswitch'
    result_folder = '../../surdata/qswitch/Results_qswitch_5users_T30min.csv'

    df_sur, vals = read_pkl_surrogate(folder)
    df_meta = read_pkl_meta(folder)
    df_sa = read_pkl_sa(folder)
    df_gs = read_pkl_gridsearch(folder)

    xs = []
    for df in [df_sur, df_meta, df_sa, df_gs]:
        x = get_best_x(df)  
        xs.append(x)  

    vals['N'] = 1
    nprocs = mp.cpu_count()

    dfs = []
    seed_count = 1
    while True:
        for x,method in zip(xs, ['Surrogate', 'Meta',
                                'Simulated Annealing', 'Random Gridsearch']):
            sim = Simulation(simwrapper, simulation_qswitch)
            res = sim.run_exhaustive(x=x, vals=vals, N=nprocs, seed=seed_count)
            df = to_dataframe(res)
            df['Method'] = method
            dfs.append(df)
        seed_count += 1
        if len(dfs)*nprocs > 4000:
            break