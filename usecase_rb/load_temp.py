import pickle
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
import src
import config

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
plt.style.use("seaborn-v0_8-paper")
# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.size': 40,  
# })
font = 14
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': font,
    'axes.labelsize': font,  
    'xtick.labelsize': font,  
    'ytick.labelsize': font, 
    'legend.fontsize': font,
    'legend.title_fontsize': font
})

import warnings
warnings.filterwarnings("ignore")


folder = 'rb'

sas = []
for name in glob.glob(f'../../surdata/{folder}/SA_*.pkl'):
    with open(name,'rb') as file: sas.append(pickle.load(file))

print(sas[2]['time'].sum()/60/60)