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
    'text.usetex': True,
    'font.family': 'serif',
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


folder = '../../surdata/rb'

# plot results of optimization (Utility)
# plot_optimization_results(folder)

# plot from exhaustive run
df = pd.read_csv(f'{folder}/Results_starlight.csv').drop('Unnamed: 0', axis=1)
df_compare = pd.read_csv(f'{folder}/Results_starlight_compare.csv').drop('Unnamed: 0', axis=1)
df_compare['Method'] = df_compare['Method'].apply(lambda x: 'Even' if x == 'Random Gridsearch' else x)
print(df_compare.Method.unique)
print(df_compare)

df_plot = pd.concat([df, df_compare], axis=0)
print(df_plot)
plot_from_exhaustive(df_plot)

