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
for name in glob.glob('../../surdata/Sur_starlight_3h_*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))

print(surs)