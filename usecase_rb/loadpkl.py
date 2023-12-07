import pickle
import glob
import sys
import numpy as np
import pandas as pd
sys.path.append('../')
sys.path.append('../src')
sys.path.append('../usecase_cd')
sys.path.append('../usecase_rb')
import src
import config

surs = []
for name in glob.glob('../../surdata/Sur_starlight_3h_*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))


df = pd.DataFrame([np.sum(pd.DataFrame(sur.y).apply(lambda col: col+pd.DataFrame(sur.X_df).T.sum()/(9*110), axis=0), axis=1) for sur in surs]).T
print(df)