
from config_vardoyan import *
from src.utils import *


if __name__ == '__main__':
        

        # user input:
        max_time= MAX_TIME * 3600 # in sec

        result = {'server_distance':[], 'Utility': [], 'Utility_std':[], 'Rate':[], 'Fidelity':[]}
        best_params = []
        for i, server_distance in enumerate(np.linspace(50,1,10)):
                # instatiante surrogate model and run optimization
                try:
                        sim = Surrogate(simwrapper, simulation_qswitch, sample_size=initial_model_size)
                        sim.vals['initial_distances'] = [server_distance,2]
                        sim.vals['total_runtime_in_seconds'] = 30 # simulation time [s]
                        sim.optimize(max_time=max_time, verbose=True)

                        obj_sums = np.sum(sim.y, axis=1)
                        best = np.max(obj_sums)
                        best_index = np.argmax(obj_sums)
                        best_std = np.sum(sim.y_std, axis=1)[best_index]
                        best_e2e_rate = sim.y_raw[best_index][0]
                        best_e2e_fidel = sim.y_raw[best_index][1]

                        result['Utility'].append(best)
                        result['Utility_std'].append(best_std)
                        result['Rate'].append(best_e2e_rate)
                        result['Fidelity'].append(best_e2e_fidel)
                        result['server_distance'].append(server_distance)

                        best_params.append(sim.X_df.iloc[best_index])

                        print('server distance = ', server_distance)
                except:
                        print(f'An exception occurred at server distance {server_distance}')
        
        df_params = pd.DataFrame.from_records(best_params)
        df_result = pd.DataFrame.from_records(result)
        
        df = df_params.join(df_result, how='left')
        print(df)

        with open(f'../../surdata/qswitch/Sur_qswitch_nleafnodes{NLEAF_NODES}_{MAX_TIME:.2f}h_objective-servernode_SEED{SEED_OPT}_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
                pickle.dump(df, file)
