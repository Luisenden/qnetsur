from config import *

from src.utils import *

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    q0 = []
    q1 = []
    objs = []

    # for _ in range(100):
    #     simul = Simulation(sim_wrapper=simwrapper, sim=simulation.simulation_cd, vals=vals, vars=vars)
    #     x = simul.get_random_x(1)

    #     x = {**x, **vals}
    #     q0.append(x['q_swap_level0'])
    #     q1.append(x['q_swap_level1'])

    #     objs.append(np.array(simul.run_sim(x)[0]))
    
    # df = pd.DataFrame.from_records(objs).explode(0)
    # df['q_swap0'] = q0
    # df['q_swap1'] = q1

    # df_plot = df.melt(id_vars=['q_swap0', 'q_swap1'], var_name='Node', value_name='# VN')
    # sns.lineplot(data=df_plot, x='q_swap0', y='# VN', hue='Node', marker='^')
    # sns.lineplot(data=df_plot, x='q_swap1', y='# VN', hue='Node', marker='o')
    # plt.ylabel('# virtual neighbours')
    # plt.ylim([0,4])
    # plt.xlabel(r'$q_{swap}$')
    # plt.show()

    objs = []
    xs = []
    q_swap = np.linspace(0,1,20)
    grid = np.array(np.meshgrid(q_swap, q_swap)).T.reshape(-1,2)
    for point in grid:
        simul = Simulation(sim_wrapper=simwrapper, sim=simulation.simulation_cd, vals=vals, vars=vars)
        x = {'q_swap_level0': point[0], 'q_swap_level1': point[1]}
        xs.append(x)
        x = {**x, **vals}

        obj, _, _ = np.array(simul.run_sim(x))
        objs.append(obj)

    res = pd.DataFrame(objs).join(pd.DataFrame(xs)).round(2)
    res['# VN'] = res[[0,1,2]].sum(axis=1)
    res = res.drop([0,1,2], axis=1)
    
    df_plot = res.pivot(index='q_swap_level0', columns='q_swap_level1', values='# VN')
    df_plot.to_pickle('21tree-heatmap.pkl')
    sns.heatmap(df_plot)
    plt.show()



