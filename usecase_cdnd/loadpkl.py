import pickle
import sys
import numpy as np
sys.path.append('../')
sys.path.append('../src')
import src

with open('../../surdata/Sur_ND_tree_180h_objective-meanopt_SEED42_12-06-2023_22:59:37.pkl', 'rb') as file: 
    sur = pickle.load(file)

print(sur.X_df)
print(np.mean(sur.y, axis=1))