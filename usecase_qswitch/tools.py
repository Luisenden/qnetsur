import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
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
    'axes.titlesize': font
})

def plotting(df):
    markers = ['o', '^', 'x', 's']
    linestyles = '-', '--', '-.', ':'
    fig, axs = plt.subplots(3,2)
    sns.pointplot(data= df.drop_duplicates(), x='User', y='Utility', hue='Method', ax=axs.flat[0], errorbar='se', markers=markers, linestyles=linestyles, scale=0.8)
    sns.pointplot(data= df[['Method', 'Aggregated Utility']].drop_duplicates(), x='Method', y='Aggregated Utility', hue='Method', ax=axs.flat[1], errorbar='se', markers=markers, scale=0.8)

    sns.pointplot(data= df.drop_duplicates(), x='User', y='Rate [Hz]', hue='Method', ax=axs.flat[2], errorbar='se', markers=markers, linestyles=linestyles, scale=0.8)
    sns.pointplot(data= df[['Method', 'Aggregated Rate [Hz]']].drop_duplicates(), x='Method', y='Aggregated Rate [Hz]', ax=axs.flat[3], hue='Method', errorbar='se', markers=markers, scale=0.8)

    sns.pointplot(data=df.drop_duplicates(), x='User', y='Fidelity', hue='Method', ax=axs.flat[4], errorbar='se', markers=markers, linestyles=linestyles, scale=0.8)
    sns.pointplot(data=df.drop_duplicates(), x='Method', y='Fidelity', hue='Method', ax=axs.flat[5], errorbar='sd', markers=markers, scale=0.8)

    for i in range(len(axs.flat)):
        axs.flat[i].grid(alpha=0.3)
        axs.flat[i].legend().remove()        
        if i % 2:
            axs.flat[i].set_xticks([])
            axs.flat[i].set_xlabel('')
            axs.flat[i].set_ylabel('')
    
    axs.flat[1].set_title('Aggregated Utility')
    axs.flat[3].set_title('Aggregated Rate [Hz]')
    axs.flat[5].set_title('Mean Fidelity')
    axs.flat[5].set_xlabel('Method')
    axs.flat[1].legend()
    plt.show()

    