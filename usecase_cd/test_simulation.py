from config import *

from src.utils import *

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=3)

if __name__ == '__main__':

    simul = Simulation(simulation.simulation_cd, vals, vars)
    x = simul.get_random_x(1)

    x = {**x, **vals}
    mean = simul.run_sim(x)[0]
    df = pd.DataFrame(mean).T
    sns.lineplot(data=df)
    plt.ylabel('# virtual neighbours')
    plt.xlabel('time [ms]')
    plt.show()




