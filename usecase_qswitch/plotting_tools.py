"""
Plotting script for results from `extract_best_params_and_run_exhaustive.py`.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

import warnings
warnings.filterwarnings("ignore")

def plotting(df):
    markers = ['o', '^', 'x', 's']
    linestyles = '-', '--', '-.', ':'
    fig, axs = plt.subplots(3,2, figsize=(10,7))
    sns.pointplot(data= df, x='User', y='Utility', hue='Method', ax=axs.flat[0], errorbar='se', markers=markers, linestyles=linestyles)
    sns.pointplot(data= df, x='Method', y='Aggregated Utility', hue='Method', ax=axs.flat[1], errorbar='se', markers=markers)

    sns.pointplot(data= df, x='User', y='Rate [Hz]', hue='Method', ax=axs.flat[2], errorbar='se', markers=markers, linestyles=linestyles)
    sns.pointplot(data= df, x='Method', y='Aggregated Rate [Hz]', ax=axs.flat[3], hue='Method', errorbar='se', markers=markers)

    sns.pointplot(data=df, x='User', y='Fidelity', hue='Method', ax=axs.flat[4], errorbar='se', markers=markers, linestyles=linestyles)
    sns.pointplot(data=df, x='Method', y='Fidelity', hue='Method', ax=axs.flat[5], errorbar='sd', markers=markers)

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
    axs.flat[5].set_title('Average Fidelity')
    axs.flat[5].set_xlabel('Method')
    axs.flat[1].legend()
    plt.tight_layout()
    plt.show()

    
    
if __name__ == '__main__':    
    
    df_plot = pd.read_csv('../../../surdata/qswitch/Results_qswitch_5users_T30min.csv')
    plotting(df_plot)

    