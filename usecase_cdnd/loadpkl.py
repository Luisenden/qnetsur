import pickle
import glob
import sys
import numpy as np
sys.path.append('../')
sys.path.append('../src')
import src
import config

axs = []
for name in glob.glob('../../surdata/CD/23tree/Ax_ND_tree_0.2h_objective-meanopt_*'):
    with open(name,'rb') as file: axs.append(pickle.load(file))

surs = []
for name in glob.glob('../../surdata/CD/23tree/Sur_ND_tree_0.2h_objective-meanopt_*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

sas = []
for name in glob.glob('../../surdata/CD/23tree/SA_ND_tree_0.2h_objective-meanopt_*'):
    with open(name,'rb') as file: sas.append(pickle.load(file))

print([ax[0].get_trials_data_frame() for ax in axs])
print(surs)
print(sas)
#print(np.mean(surs[0].y,axis=1)-np.mean(surs[0].y_raw,axis=1))