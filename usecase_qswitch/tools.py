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
    'legend.title_fontsize': font
})

def plotting(df):
    alpha = 0.3
    size = 2
    fig, axs = plt.subplots(3,2)
    sns.stripplot(data= df.drop('index', axis=1), x='User', y='Utility', hue='Method', ax=axs.flat[0], size=size, dodge=True)
    sns.boxplot(data= df.drop('index', axis=1), x='User', y='Utility', hue='Method', 
                boxprops=dict(alpha=alpha), flierprops=dict(alpha=alpha), capprops=dict(alpha=alpha), medianprops=dict(alpha=alpha), whiskerprops=dict(alpha=alpha), ax=axs.flat[0])
    
    sns.stripplot(data= df[['Method', 'Aggregated Utility']].drop_duplicates(), x='Method', y='Aggregated Utility', ax=axs.flat[1], palette=sns.color_palette(), size=size, dodge=True)
    sns.boxplot(data= df[['Method', 'Aggregated Utility']].drop_duplicates(), x='Method', y='Aggregated Utility', 
            boxprops=dict(alpha=alpha), flierprops=dict(alpha=alpha), capprops=dict(alpha=alpha), medianprops=dict(alpha=alpha), whiskerprops=dict(alpha=alpha), ax=axs.flat[1])
    axs.flat[1].set_xticks([])
    axs.flat[1].set_xlabel('Aggregated Utility')
    axs.flat[1].set_ylabel('')

    sns.stripplot(data= df.drop('index', axis=1), x='User', y='Rate [Hz]', hue='Method', ax=axs.flat[2], size=size, dodge=True)
    sns.boxplot(data= df.drop('index', axis=1), x='User', y='Rate [Hz]', hue='Method', 
                boxprops=dict(alpha=alpha), flierprops=dict(alpha=alpha), capprops=dict(alpha=alpha), medianprops=dict(alpha=alpha), whiskerprops=dict(alpha=alpha), ax=axs.flat[2])
    
    sns.stripplot(data= df[['Method', 'Aggregated Rate [Hz]']].drop_duplicates(), x='Method', y='Aggregated Rate [Hz]', ax=axs.flat[3], palette=sns.color_palette(), size=size, dodge=True)
    sns.boxplot(data= df[['Method', 'Aggregated Rate [Hz]']].drop_duplicates(), x='Method', y='Aggregated Rate [Hz]', 
            boxprops=dict(alpha=alpha), flierprops=dict(alpha=alpha), capprops=dict(alpha=alpha), medianprops=dict(alpha=alpha), whiskerprops=dict(alpha=alpha), ax=axs.flat[3])
    axs.flat[3].set_xticks([])
    axs.flat[3].set_xlabel('Aggregated Rate [Hz]')
    axs.flat[3].set_ylabel('')

    sns.scatterplot(data=df.drop_duplicates(), x='User', y='Fidelity', hue='Method', ax=axs.flat[4], style='Method')
    sns.scatterplot(data=df.drop_duplicates(), x='Method', y='Fidelity Mean', style='Method', hue='Method', palette=sns.color_palette(), ax=axs.flat[5])
    axs.flat[5].errorbar(x=df['Method'].drop_duplicates(), y=df['Fidelity Mean'].drop_duplicates(), yerr=df['Fidelity Std'].drop_duplicates(), ecolor=sns.color_palette()[:3], fmt='_')
    axs.flat[5].set_ylabel('')
    axs.flat[5].set_xlabel('Mean Fidelity')
    axs.flat[5].set_xticklabels(['', '', ''])

    for ax in axs.flat:
        ax.grid(alpha=alpha)
        ax.legend().remove()
    axs.flat[-1].legend()
    plt.show()

def plotting2(df):
    alpha = 0.3
    size = 2
    fig, axs = plt.subplots(3,2)
    sns.pointplot(data= df.drop('index', axis=1), x='User', y='Utility', hue='Method', ax=axs.flat[0], size=size, dodge=True)

    sns.pointplot(data= df[['Method', 'Aggregated Utility']].drop_duplicates(), x='Method', y='Aggregated Utility', ax=axs.flat[1], palette=sns.color_palette(), size=size, dodge=True)
    axs.flat[1].set_xticks([])
    axs.flat[1].set_xlabel('Aggregated Utility')
    axs.flat[1].set_ylabel('')

    sns.pointplot(data= df.drop('index', axis=1), x='User', y='Rate [Hz]', hue='Method', ax=axs.flat[2], size=size, dodge=True)
    
    sns.pointplot(data= df[['Method', 'Aggregated Rate [Hz]']].drop_duplicates(), x='Method', y='Aggregated Rate [Hz]', ax=axs.flat[3], palette=sns.color_palette(), size=size, dodge=True)

    axs.flat[3].set_xticks([])
    axs.flat[3].set_xlabel('Aggregated Rate [Hz]')
    axs.flat[3].set_ylabel('')

    sns.scatterplot(data=df.drop_duplicates(), x='User', y='Fidelity', hue='Method', ax=axs.flat[4], style='Method')
    sns.scatterplot(data=df.drop_duplicates(), x='Method', y='Fidelity Mean', style='Method', hue='Method', palette=sns.color_palette(), ax=axs.flat[5])
    axs.flat[5].errorbar(x=df['Method'].drop_duplicates(), y=df['Fidelity Mean'].drop_duplicates(), yerr=df['Fidelity Std'].drop_duplicates(), ecolor=sns.color_palette()[:3], fmt='_')
    axs.flat[5].set_ylabel('')
    axs.flat[5].set_xlabel('Mean Fidelity')
    axs.flat[5].set_xticklabels(['', '', ''])

    for ax in axs.flat:
        ax.grid(alpha=alpha)
        ax.legend().remove()
    axs.flat[-1].legend()
    plt.show()

    