from config import *
from src.utils import *

import glob

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
plt.style.use('seaborn')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, make_scorer



if __name__ == '__main__':
        
        # user input:
        max_time= 0 # in sec

 
        # sim_ref = Surrogate(simulation_qswitch, vals=vals, vars=vars, sample_size=100)
        # sim_ref.optimize(max_time=max_time, verbose=True)

        # with open(f'ref_'+datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'.pkl', 'wb') as file:
        #         pickle.dump(sim_ref, file)


        for name in glob.glob('ref_12-20-2023_19:16:42.pkl'):
            with open(name,'rb') as file: ref = pickle.load(file)

        ref.X_df = ref.X_df.drop(['Iteration'], axis=1)


        print(cross_val_score(MultiOutputRegressor(SVR()), ref.X_df, ref.y, scoring=make_scorer(mean_absolute_error)).mean())
        print(cross_val_score(MultiOutputRegressor(DecisionTreeRegressor()), ref.X_df, ref.y, scoring=make_scorer(mean_absolute_error)).mean())

        X_train, X_test, y_train, y_test = train_test_split(ref.X_df, ref.y, test_size=0.1, random_state=2)
        sur = MultiOutputRegressor(SVR()).fit(X_train, y_train)

        print(X_train)
        y_predicted = sur.predict(X_test)
        X_test[['states_predicted', 'fidelity_predicted']] = y_predicted
        X_test[['states_simulation', 'fidelity_simulation']] = y_test
        print(X_test)


        df = pd.melt(X_test, id_vars = 'num_positions', value_vars=['states_predicted', 'states_simulation'], value_name='states', var_name='Comparison')
        g = sns.lineplot(data = df, x='num_positions', y='states', hue='Comparison', style='Comparison', marker='^')
        plt.show()
