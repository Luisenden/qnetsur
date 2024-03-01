from config import *
from src.utils import *
from load_params import *
from simulation import *

if __name__ == '__main__':

        folder = 'rb_N10_24h'
        params = load_max_params(folder)[:1]
        sim = Simulation(simwrapper, simulation_rb)
        for i in range(len(params))[:1]:
                res = sim.run_exhaustive(vals=vals, x=params.loc[params.index[i]], N=10)
                raw = [sum(result[0] + params.loc[params.index[i]]/m_max) for result in res]
                mean = np.mean(raw)
                std = np.std(raw)

        params['raw'] = [raw]
        params['mean'] = mean
        params['std'] = std
        print(params)
        # with open(f'../../surdata/rb/SU_starlight_{MAX_TIME:.1f}h_objective-meanopt_SEED{SEED}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
        #         pickle.dump(sur, file)