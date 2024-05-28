import pandas as pd
import pickle
import glob
import numpy as np

import pickle
import sys
sys.path.append('../')

class ResultCollector:
    def __init__(self, folder, file_prefix):
        self.prefix = file_prefix
        self.folder = folder
        self.dfs = []
        files = glob.glob(folder + file_prefix + '*.pkl')
        for i, file in enumerate(files):
            with open(file, 'rb') as f:
                self.data = pickle.load(f)
                data_df = self.get_total()
                data_df['Trial'] = i
                self.dfs.append(data_df)

    def get_total(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_final(self, name=None):
        results = pd.concat(self.dfs, axis=0).reset_index(drop=True)
        results['Method'] = {'SA_':'Simulated Annealing', 'SU_':'Surrogate', 'AX_':'Meta', 'RS_':'Random Search'}[self.prefix]
        if name is not None:
            results.to_csv(self.folder + name + '.csv')
        return results

class SurrogateCollector(ResultCollector):
    def __init__(self, folder):
        super().__init__(folder, 'SU_')

    def get_data_df(self):
        self.y = pd.DataFrame.from_records(self.data.y)
        self.objective = self.y.sum(axis=1).rename('objective')
        self.y_raw = pd.DataFrame.from_records(self.data.y_raw).add_suffix('_raw')
        self.y_std = pd.DataFrame.from_records(self.data.y_std).add_suffix('_std')
        self.data_df = pd.concat([self.data.X_df, self.objective, self.y, self.y_std, self.y_raw], axis=1)
        return self.data_df

    def get_timing(self):
        timing = {'Simulation':self.data.sim_time, 'Build':self.data.build_time, 'Acquisition': [0]+self.data.acquisition_time, 'Total':self.data.optimize_time}
        self.timing = pd.DataFrame.from_dict(timing, orient='index').T.add_suffix(' [s]')
        return self.timing
    
    def get_total(self):
        self.get_data_df()
        self.get_timing()
        self.total = self.data_df.merge(self.timing, left_on='Iteration', right_index=True)
        return self.total

class AxCollector(ResultCollector):
    def __init__(self, folder):
        super().__init__(folder, 'AX_')

    def get_total(self):
        return self.data[0]
    
class SaCollector(ResultCollector):
    def __init__(self, folder):
        super().__init__(folder, 'SA_')

    def get_total(self):
        print(self.data)
        return self.data[0]


class RsCollector(ResultCollector):
    def __init__(self, folder):
        super().__init__(folder, 'RS_')

    def get_total(self):
        self.data = self.data[0]
        self.data['objective'] = self.data['objective'].apply(lambda x: sum(x))
        self.data['std'] = self.data['std'].apply(lambda x: np.sqrt(sum(x**2)))
        return self.data





