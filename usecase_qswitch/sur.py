
from config import *
from src.utils import *

from simulation import simulation_qswitch


if __name__ == '__main__':
        
        # user input:
        max_time= MAX_TIME * 3600 # in sec

        # instatiante surrogate model and run optimization
        sim = Surrogate(simwrapper, simulation_qswitch, sample_size=initial_model_size)
        sim.optimize(max_time=max_time, verbose=True)

        print(np.mean(sim.y,axis=1))
        print(sim.y_raw)

        with open(f'../../surdata/qswitch/Sur_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-servernode_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(sim, file)
