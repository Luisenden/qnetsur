import numpy as np
import pandas as pd
from multiprocessing import Pool

def objective(s, X) -> tuple:
    y = s.mmodel.predict(X.values)
    y_mean = np.array([np.mean(y_i) for y_i in y])
    y_max = y_mean.max()
    index = np.where(y_mean == y_max)
    imax = index[0][0]
    return X.iloc[imax], y_max


def get_candidates(s,n) -> dict:
    x = {}
    cols = s.X_df.select_dtypes(int)
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.randint(col.min(), col.max(),n)
    cols = s.X_df.select_dtypes(float)
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.uniform(col.min(), col.max(),n)
    cols = s.X_df.select_dtypes(exclude=(int,float))
    for key in cols.keys():
        col = cols[key]
        x[key] = np.random.choice(col,n)
  
    return x 

def brute_optimize(s):
    X_rand = get_candidates(s,1000000) # used 1M sample points for S settings (2,3)tree and 3lattice
    X_rand_df = pd.DataFrame(X_rand).astype(object)
    x,y = objective(s,X_rand_df)
    return x,y
