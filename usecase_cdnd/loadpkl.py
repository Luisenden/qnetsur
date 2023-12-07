import pickle
import glob
import sys
import numpy as np
sys.path.append('../')
sys.path.append('../src')
sys.path.append('../usecase_cd')
sys.path.append('../usecase_rb')
import src
import config

surs = []
for name in glob.glob('../../surdata/Sur_ND_tree_0.1h_objective-meanopt_SEED42_12-07-2023_17:18:24.pkl'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

print(np.mean(surs[0].y,axis=1)-np.mean(surs[0].y_raw,axis=1))