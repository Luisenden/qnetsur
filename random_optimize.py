import numpy as np
import pandas as pd
from specifications import objective
import time


def get_candidates(s,n) -> dict: # get points to evaluate current surrogate model in its (sub)space
    x = {}
    X = s.X_df.copy()
    X = X.astype(s.dtypes)
    cols = X.select_dtypes(int)
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.randint(col.min(), col.max(),n)
    cols = X.select_dtypes(float)
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.uniform(col.min(), col.max(),n)
    cols = X.select_dtypes(exclude=(int,float))
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.choice(col,n)
  
    return x 

def random_optimize(s):
    X_rand = get_candidates(s,1000000) # used 1M sample points for S settings (2,3)tree and 3lattice
    X_rand_df = pd.DataFrame(X_rand).astype(object)
    data_scaled = s.scaler.fit_transform(X_rand_df)
    x = objective(s,data_scaled)
    x_rescaled = s.scaler.inverse_transform([x])
    newx = pd.DataFrame(columns=s.X_df.columns, index=[0], data=np.array(x_rescaled)).astype(s.dtypes).astype(object)
    return newx
