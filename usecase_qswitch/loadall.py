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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from cycler import cycler

plt.style.use("seaborn-paper")
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
    'axes.titlesize': font+2,
})
custom_palette = "Pastel1"

folders = ['qswitch_3nodes_0.5h', 'qswitch_3nodes_buffers_3h']#,  'qswitch_5nodes_6h', 'qswitch_5nodes_buffers_6h', 'qswitch_10nodes_12h', 'qswitch_10nodes_buffers_12h']
methods = ['Surrogate Optimizer', 'Ax Platform', 'Simulated Annealing', 'Grid Search']

def load_from_pkl(folder, return_surs=False):
    axs = []
    for name in glob.glob(f'../../surdata/{folder}/AX_*.pkl'):
        with open(name,'rb') as file: axs.append(pickle.load(file))

    surs = []
    for name in glob.glob(f'../../surdata/{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))

    sas = []
    for name in glob.glob(f'../../surdata/{folder}/SA_*.pkl'):
        with open(name,'rb') as file: sas.append(pickle.load(file))

    gss = []
    for name in glob.glob(f'../../surdata/{folder}/GS_*.pkl'):
        with open(name,'rb') as file: gss.append(pickle.load(file))

    if return_surs: 
        raw = []
        for sur in surs:
            rate, fidels = sur.y_raw[np.argmax(np.sum(sur.y, axis=1))]
            fidels['rate'] = rate
            raw.append(fidels)
        raws = pd.concat(raw, axis=1).T
        raws['folder'] = folder
        return raws

    data = pd.DataFrame()
    data[methods[3]] = [gs[0].objective.apply(lambda x: np.sum(x)).max() for gs in gss]
    data[methods[2]] = [np.max(sa.objective) for sa in sas]
    data[methods[1]] = [np.max(ax[0]['evaluate']) for ax in axs]
    data[methods[0]] = [np.sum(sur.y, axis=1).max() for sur in surs]
    data['folder'] = folder

    return data


# raws = pd.concat([load_from_pkl(folder,return_surs=True) for folder in folders], axis=0)
# g = sns.catplot(kind='box', data=raws.drop('rate', axis=1), row='folder')
# g.set_xlabels('')
# g.set_xticklabels('')
# g.set_titles('')
# g.tight_layout(pad=1, rect=[0, 0, 1, 1])
# plt.show()

# raws = pd.concat([load_from_pkl(folder,return_surs=True) for folder in folders], axis=0)
# g = sns.catplot(kind='box', data=raws[['rate', 'folder']], x='folder', y='rate')
# g.set_xlabels('')
# g.set_xticklabels('')
# g.set_titles('')
# g.tight_layout(pad=1, rect=[0, 0, 1, 1])
# g.set_xticklabels(['\# users = 3', '\# users = 3\n + buffer', '\# users = 5', '\# users = 5\n + buffer', '\# users = 10', '\# users = 10\n + buffer'], rotation=45)
# plt.tight_layout(pad=1, rect=[0, 0, 1, 1])
# plt.ylabel('Rate [pairs/second]')
# plt.show()


# plot over all maximum outcomes
dfs = pd.concat([load_from_pkl(folder) for folder in folders], axis=0)
df_plot = dfs.melt(id_vars='folder', value_name='Utility', var_name='Method')

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_plot, x='folder', y='Utility', hue='Method', ax=ax, palette=custom_palette)
sns.stripplot(x='folder', y='Utility', hue='Method', data=df_plot, jitter=True, ax=ax, dodge=True, palette='dark:#404040', alpha=0.6, legend=False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_xticklabels(['\# users = 3', '\# users = 3\n + buffer', '\# users = 5', '\# users = 5\n + buffer', '\# users = 10', '\# users = 10\n + buffer'], rotation=45)
plt.grid()
plt.xlabel('')

zoomed_data = df_plot[df_plot['folder'].isin(['qswitch_3nodes_0.5h', 'qswitch_3nodes_buffers_3h'])]
axins = inset_axes(ax, width="30%", height="40%", loc='upper left', 
                   bbox_to_anchor=(0.05, -0.09, 1, 1), bbox_transform=ax.transAxes)
axins.patch.set_alpha(0.7)
sns.boxplot(data=zoomed_data, x='folder', y='Utility', hue='Method', ax=axins, palette=custom_palette)
sns.stripplot(x='folder', y='Utility', hue='Method', data=zoomed_data, jitter=True, ax=axins, dodge=True, color="#404040", alpha=0.6)
axins.set_xticklabels(['\# users = 3', '\# users = 3\n incl. buffer'])
axins.set_xlabel('')
axins.set_ylabel('')
axins.set_title('')
axins.set_xticklabels('')
axins.legend().remove()  

con_patch1 = ConnectionPatch(xyA=(0, 0.), coordsA=axins.transAxes, xyB=(0.01, 0.1), coordsB=ax.transAxes,
                             arrowstyle="-", linestyle="--", color="black", shrinkB=5, lw=1)
ax.add_artist(con_patch1)
con_patch2 = ConnectionPatch(xyA=(1, 1), coordsA=axins.transAxes, xyB=(0.32, 0.22), coordsB=ax.transAxes,
                             arrowstyle="-", linestyle="--", color="black", shrinkB=5, lw=1)
ax.add_artist(con_patch2)
plt.tight_layout(pad=1, rect=[0, 0, 1, 1])
plt.show()
