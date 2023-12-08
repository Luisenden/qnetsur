import pickle
import glob
import sys
import numpy as np
import pandas as pd
sys.path.append('../')
sys.path.append('../src')
sys.path.append('../usecase_rb')

import src
from config import *

surs = []
for name in glob.glob('../../surdata/RB/*'):
    with open(name,'rb') as file: surs.append(pickle.load(file))


print(surs)
print(len(surs))