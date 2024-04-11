"""
Plotting script for results from `extract_best_params_and_run_exhaustive.py`.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from operator import itemgetter
import glob
import sys
sys.path.append('../')
sys.path.append('../src')
from src.utils import *
from config import *

import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-paper")
font = 18
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

def read_in_surrogate(folder):
    surs = []
    for name in glob.glob(f'{folder}/SU_*.pkl'): 
        with open(name,'rb') as file: surs.append(pickle.load(file))

    vals = surs[0].vals

    Xs = []
    for i, sur in enumerate(surs):
        sur.X_df['Trial'] = i
        Xs.append(sur.X_df)

    Xs = pd.concat(Xs, axis=0, ignore_index=True)
    ys = pd.concat([pd.DataFrame(sur.y, columns = pd.Series(range(sur.vals['nnodes']-1)).astype('str')) 
                    for sur in surs], axis=0, ignore_index=True)
    ys['Utility'] = ys.sum(axis=1)
    ys_std = pd.concat([pd.DataFrame(sur.y_std, columns = pd.Series(range(sur.vals['nnodes']-1)).astype('str')).add_suffix('_std') 
                        for sur in surs], axis=0, ignore_index=True)

    ys_std['Utility Std'] = ys_std.apply(np.square).sum(axis=1).apply(np.sqrt)
    df = Xs.join(ys)
    df = df.join(ys_std)
    return df, vals

def read_in_meta(folder):
    metas = []
    for name in glob.glob(f'{folder}/AX_*.pkl'): 
        with open(name,'rb') as file: metas.append(pickle.load(file))
    dfs = []
    for meta in metas:
        dfs.append(meta[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['evaluate']
    return df

def read_in_gridsearch(folder):
    gss = []
    for name in glob.glob(f'{folder}/GS_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for gs in gss:
        dfs.append(gs[0])
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    return df

def read_in_sa(folder):
    gss = []
    for name in glob.glob(f'{folder}/SA_*.pkl'): 
        with open(name,'rb') as file: gss.append(pickle.load(file))
    dfs = []
    for gs in gss:
        dfs.append(gs)
    df = pd.concat(dfs, axis=0)
    df['Utility'] = df['objective'].apply(np.nansum)
    print(df)
    return df


def to_dataframe(res):
    df = pd.DataFrame.from_records(res)
    df_raw = df[2].transform({i: itemgetter(i) for i in range(4)}).drop([1, 3], axis=1)
    df_raw = df_raw.applymap(lambda x: x[1:]) # ignore entry for server node
    df_raw.columns = ['Rate [Hz]', 'Fidelity']
    
    df = df.drop([1, 2], axis=1)
    df.columns = ['Utility']
    df = df.explode('Utility').reset_index()
    df_raw = df_raw.apply(lambda x: x.explode(ignore_index=True), axis=0)
    df = df.join(df_raw)
    df['Aggregated Utility'] = df.groupby('index')['Utility'].transform(np.sum)
    df['Aggregated Rate [Hz]'] = df.groupby('index')['Rate [Hz]'].transform(np.sum)
    df['Fidelity Mean'] = df.groupby('index')['Fidelity'].transform(np.mean)
    df['Fidelity Std'] = df.groupby('index')['Fidelity'].transform(np.std)
    df['User'] = df.groupby('index').cumcount()
    return df

def get_best_x(df):
    return df.iloc[df['Utility'].idxmax()][df.columns.str.contains('bright_state')]

def plotting(df):
    markers = ['o', '^', 'x', 's']
    linestyles = '-', '--', '-.', ':'
    fig, axs = plt.subplots(3,2, figsize=(10,7))
    sns.pointplot(data= df, x='User', y='Utility', hue='Method', ax=axs.flat[0], errorbar='se', markers=markers, linestyles=linestyles)
    sns.pointplot(data= df, x='Method', y='Aggregated Utility', hue='Method', ax=axs.flat[1], errorbar='se', markers=markers)

    sns.pointplot(data= df, x='User', y='Rate [Hz]', hue='Method', ax=axs.flat[2], errorbar='se', markers=markers, linestyles=linestyles)
    sns.pointplot(data= df, x='Method', y='Aggregated Rate [Hz]', ax=axs.flat[3], hue='Method', errorbar='se', markers=markers)

    sns.pointplot(data=df, x='User', y='Fidelity', hue='Method', ax=axs.flat[4], errorbar='se', markers=markers, linestyles=linestyles)

    for i in range(len(axs.flat)):
        axs.flat[i].grid(alpha=0.3)
        axs.flat[i].legend().remove()
        axs.flat[i].set_xlabel('')        
        if i % 2:
            axs.flat[i].set_xticks([])
            axs.flat[i].set_ylabel('')
    
    axs.flat[0].set_title('Utility per User')
    axs.flat[1].set_title('Aggregated Utility')
    axs.flat[2].set_title('Rate [Hz] per User')
    axs.flat[3].set_title('Aggregated Rate [Hz]')
    axs.flat[4].set_title('Fidelity per User')
    axs.flat[4].set_xlabel('User')
    axs.flat[-1].axis('off')

    handles, labels = axs.flat[1].get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.85, 0.1))
    plt.tight_layout()
    plt.show()

def plot_surrogate_linklevelfidels(folder):
    df, _ = read_in_surrogate(folder)
    
    cols_i =df.columns.str.contains('bright_state')
    cols_names = ['Server']+[f'User {i}' for i in range(4,-1,-1)]
    df[cols_names] = df[df.columns[cols_i]].applymap(lambda x: 1-x)

    df = pd.melt(df, value_vars=cols_names, var_name='Node', value_name='Link-level Fidelity', id_vars=['Iteration', 'Trial'])
    
    fig, ax = plt.subplots(figsize = (10,4))
    sns.lineplot(data=df[df.Trial==0], x='Iteration', y='Link-level Fidelity', hue='Node', style='Node')
    sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))
    plt.title('Link-level Fidelity per Node')
    plt.ylabel('Fidelity')
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # folder = '../../surdata/qswitch'
    # plot_surrogate_linklevelfidels(folder)

    df = pd.read_csv('../../surdata/qswitch/Results_qswitch_5users_T30min.csv')
    plotting(df)

    