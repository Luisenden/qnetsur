import pandas as pd
import pickle
import glob
import numpy as np
import sys

class SurrogateCollector:
    """Collector designed for surrogate-based optimization results."""
    def __init__(self, sim):
        self.model = sim
        
    def get_model_df(self):
        self.y = pd.DataFrame.from_records(self.model.y)
        self.objective = self.y.sum(axis=1).rename('objective')
        self.y_raw = pd.DataFrame.from_records(self.model.y_raw).add_suffix('_raw')
        self.y_std = pd.DataFrame.from_records(self.model.y_std).add_suffix('_std')
        self.model_df = pd.concat([self.model.X_df, self.objective, self.y, self.y_std, self.y_raw], axis=1)
        return self.model_df

    def get_timing(self):
        timing = {'Simulation':self.model.sim_time, 'Build':self.model.build_time, 'Acquisition': [0]+self.model.acquisition_time, 'Total':self.model.optimize_time}
        self.timing = pd.DataFrame.from_dict(timing, orient='index').T.add_suffix(' [s]')
        return self.timing
    
    def get_machine_learning_scores(self):
        self.ml_model_scores = pd.DataFrame.from_dict(self.model.model_scores, orient='index').T
        return self.ml_model_scores
    
    def get_total(self):
        self.get_model_df()
        self.get_timing()
        self.get_machine_learning_scores()
        self.total = self.model_df.merge(self.timing, left_on='Iteration', right_index=True)
        self.total = self.total.merge(self.ml_model_scores, left_on='Iteration', right_index=True)
        return self.total
    

# class OtherCollector():
#     def __init__(self, result):
#         self.result = result

# class AxCollector(OtherCollector):
#     """Collector for optimization results from Ax platform."""
#     def __init__(self, folder):
#         super().__init__(folder, 'AX_')

#     def get_total(self):
#         return self.model[0]
    
# class SaCollector(ResultCollector):
#     """Collector for optimization results from Simulated Annealing."""
#     def __init__(self, folder):
#         super().__init__(folder, 'SA_')

#     def get_total(self):
#         print(self.model)
#         return self.model[0]


# class RsCollector(ResultCollector):
#     """Collector for optimization results from Random Search."""
#     def __init__(self, folder):
#         super().__init__(folder, 'RS_')

#     def get_total(self):
#         self.model = self.model[0]
#         self.model['objective'] = self.model['objective'].apply(lambda x: sum(x))
#         self.model['std'] = self.model['std'].apply(lambda x: np.sqrt(sum(x**2)))
#         return self.model