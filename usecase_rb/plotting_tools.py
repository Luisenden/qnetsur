"""
Plotting script for results from `extract_best_params_and_run_exhaustive.py`.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


def plotting(df):
    markers = ['o', '^', 'x', 's']
    # linestyles = '-', '--', '-.', ':', '-:', '.'
    fig, axs = plt.subplots(1,2, figsize=(10,7))
    sns.pointplot(data= df, x='User', y='Number of Completed Requests', hue='Method', ax=axs[0], errorbar='se', markers=markers, linestyles=['']*9, legend=False)
    sns.pointplot(data= df, x='Method', y='Aggregated Completed Requests', hue='Method', ax=axs[1], errorbar='se', markers=markers, legend=True, linestyles=['']*4)

    # handles, labels = axs[1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.5, 0.1))
    plt.xticks(['']*9)
    plt.tight_layout()
    plt.show()

    
    
if __name__ == '__main__':    
    
    df_plot = pd.read_csv('../../surdata/rb/Results_starlight.csv').drop('Unnamed: 0', axis=1)
    print(df_plot)

    df_plot = df_plot.melt(id_vars=['Method', 'Aggregated Completed Requests'], var_name='User', value_name='Number of Completed Requests')

    df_plot['User'] = df_plot['User'].apply(lambda x: str.replace(x, 'Node', ''))
    plotting(df_plot)

    