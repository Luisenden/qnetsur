import pickle
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
import src
import config

from extract_best_params_and_run_exhaustive import *

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
    'legend.title_fontsize': font
})

import warnings
warnings.filterwarnings("ignore")


folder = '../../surdata/rb'


target_columns = ['Trial', 'Utility', 'Method']

df = pd.concat([read_pkl_sa(folder)[target_columns], read_pkl_surrogate(folder)[0][target_columns],
                read_pkl_gridsearch(folder)[target_columns], read_pkl_meta(folder)[target_columns]], axis=0, ignore_index=True)

# grouped = df.groupby(['Method', 'Trial']).max()
# sns.pointplot(data=grouped, x='Method', y='Utility', errorbar='se', linestyles='None', hue='Method')
# plt.xlabel('')
# plt.title('Maximum Utility of N=10 Trials (per Method)')
# plt.xticks(rotation=45)
# plt.grid()
# plt.tight_layout()
# plt.show()

df_sur = read_pkl_surrogate(folder)[0]
x_best = df_sur[df_sur.Trial==df_sur.iloc[df_sur['Utility'].idxmax()].Trial].reset_index()

param_cols = df_sur.columns[df_sur.columns.str.contains('mem_size')]
sns.lineplot(x_best[param_cols])
plt.show()



