from config import *
from src.utils import *

from optimizingcd import main_cd as simulation

from src.simulatedannealing import simulated_annealing 


if __name__ == '__main__':

        # user input: network topology type
        vv = sys.argv[1]

        # user input: number of maximum iterations optimization
        MAXITER = int(sys.argv[2]) 

        # user input: number of trials
        ntrials = int(sys.argv[3]) 


        v = vv.split(',') 
        assert(len(v) in [1,2]), 'Argument must be given for network topology: e.g. "11" yields 11x11 square lattice, while e.g. "2,3" yields 2,3-tree network.'
        topo = NetworkTopology((int(v[0]), ), 'square') if len(v)==1 else NetworkTopology((int(v[0]), int(v[1])), 'tree')
        size = topo.size
        vals['A'] = simulation.adjacency_squared(size[0]) if topo.name == 'square' else simulation.adjacency_tree(size[0], size[1])

        vars = { # define variables and bounds for given simulation function
            'range': {
                'M': ([1, 10],'int')
                },
            'choice':{}
        } 
        for i in range(np.shape(vals['A'])[0]):
            vars['range'][f'q_swap{i}'] = ([0., 1.], 'float')


        # baseline simulated annealing
        start = time.time()
        si = Simulation(simulation.simulation_cd, vals, vars)
        simaneal = partial(simulated_annealing, MAXITER=MAXITER)
        with Pool(processes=ntrials) as pool:
            result = pool.map(simaneal, [si]*ntrials) 
        
        total_timeSA = time.time()-start
        result = pd.DataFrame.from_records(result)

        with open('../../surdata/SA_ND_'+topo.name+vv.replace(',','')+'_iter-'+str(MAXITER)+'_objective-meanopt'+datetime.now().strftime("%m-%d-%Y_%H:%M")+'.pkl', 'wb') as file:
                pickle.dump([result,[total_timeSA]], file)
