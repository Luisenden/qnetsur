import numpy as np
import pandas as pd

def objective(s, X) -> tuple:
    y = s.mmodel.predict(X.values)
    y_mean = np.array(list(map(np.mean, y)))
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
    X_rand = get_candidates(s,1000000)
    X_rand_df = pd.DataFrame(X_rand).astype(object)
    x,y = objective(s,X_rand_df)
    return x,y
