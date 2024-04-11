import pickle
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
import src
import config

from plotting_tools import *

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
plt.style.use("seaborn-v0_8-paper")
font = 14
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'arial',
    'font.size': font,
    'axes.labelsize': font,  
    'xtick.labelsize': font,  
    'ytick.labelsize': font, 
    'legend.fontsize': font,
    'legend.title_fontsize': font,
    'axes.titlesize': font
})

import warnings
warnings.filterwarnings("ignore")


folder = 'rb'

#plot results of optimization (Utility)
# plot_optimization_results(folder)

# plot from exhaustive run
method_names = ['Surrogate', 'Meta', 'Simulated Annealing', 'Random Gridsearch', 'Budget 450', 'even']
dfs = [None]*6
for name in glob.glob(f'../../surdata/{folder}/Results_*.csv'):
    df = pd.read_csv(name)
    method = df.Method[0]
    index = method_names.index(method)
    dfs[index] = df

df = pd.concat(dfs, axis=0)
df['Method'] = df['Method'].apply(lambda x: 'Even' if x == 'even' else x)
df['Method'] = df['Method'].apply(lambda x: 'Wu et. al, 2021' if x == 'Budget 450' else x)
df = df.drop('Unnamed: 0', axis=1)
plot_from_exhaustive(df)

