import sys
sys.path.append('../')
sys.path.append('../usecase_rb')
sys.path.append('../usecase_cd')
import src
import simulation
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
plt.style.use('seaborn-v0_8')

import pickle
import glob
import sys

def reduce_to_means_per_iteration(sur_df, group_size):
    sur_df['Iteration'] = [i for i in range(group_size) for _ in range(len(sur_df)//group_size)]
    return pd.DataFrame(sur_df.groupby('Iteration').mean().to_numpy()) 

def get_comparison_dataframe(raw_data:list):
    
    # merge and plot overall mean and standard deviation over different trials 

    sur_raw, ax_raw, sa_raw = raw_data # assign raw data 


    dfs = []

    if sur_raw != None:
        sur_results = pd.DataFrame([np.mean(sur_raw[0][i].y, axis=1)[10:] for i in range(ntrials)]).T # get results of trials # take all y values except initial training sample
        sur_results = reduce_to_means_per_iteration(sur_results,int(niter)) # take mean over 10 samples (=evaluations done in parallel per iteration)
        sur_df = pd.melt(sur_results, var_name='Trial', value_name='Surrogate', ignore_index=False).reset_index(names='Iteration') # convert to one column
        dfs.append(sur_df)

    if ax_raw != None:
        ax_results = pd.DataFrame([np.array(ax_raw[0][i].get_trials_data_frame()['mean']) for i in range(ntrials)]).T # get results of trials
        ax_df = pd.melt(ax_results, var_name='Trial', value_name='Ax', ignore_index=False).reset_index(names='Iteration') # convert to one column
        ax_df = ax_df.drop(['Iteration','Trial'], axis=1)
        dfs.append(ax_df)

    if sa_raw != None:
        sa_results = pd.concat([pd.DataFrame().from_records(sa_raw[0].iloc[i]).objective[1:] for i in range(ntrials)], axis=1) # get results of trials ####### for some reason 21 iterations?? #######
        sa_df = pd.melt(sa_results, var_name='Trial', value_name='SA', ignore_index=False).reset_index(names='Iteration') # convert to one column
        sa_df = sa_df.drop(['Iteration','Trial'], axis=1)
        dfs.append(sa_df)

    df_plot = pd.concat(dfs, axis=1).melt(id_vars=['Iteration', 'Trial'], var_name='Method', value_name='Number of virtual neighbours') # concatenate all three methods' results

    return df_plot

def plot_overall(df, store=False):

    dspace = len(sur_loaded_data[0][0].vars['range'])+len(sur_loaded_data[0][0].vars['choice']) # extract number of parameters

    df['Iteration'] = df['Iteration'].astype(int)

    g = sns.lineplot(data = df, x='Iteration', y='Number of virtual neighbours', hue='Method') # plot the Number of Neighbours for all methods
    plt.title(f'Optimization Quantum Network ({topo}) over {dspace} parameters')
    plt.gcf().set_size_inches(15,7)
    g.grid(which='major', color='w', linewidth=1.0)
    g.grid(which='minor', color='w', linewidth=0.5)
    g.set_xticks(range(0,int(niter),2))
    if store: plt.savefig(f'../../surdata/Figures/compare_overall_{topo}_iter-{niter}.pdf')
    plt.show()

def plot_trial(df, store=False):

    dspace = len(sur_loaded_data[0][0].vars['range'])+len(sur_loaded_data[0][0].vars['choice']) # extract number of parameters

    df['Trial'] = df['Trial'].astype(int)
    g = sns.FacetGrid(df, col='Trial', hue='Method', col_wrap=5)
    g.map(sns.scatterplot, 'Iteration', 'Number of virtual neighbours')
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Optimization Quantum Network ({topo}) over {dspace} parameters')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    
    if store: plt.savefig(f'../../surdata/Figures/compare_trials_{topo}_iter-{niter}.pdf')
   #plt.show()

def plot_avg_time(raw_data:list, store=False):

    sur_loaded_data = raw_data[0]

    rdata = [raw for raw in raw_data if raw != None] 
    
    sur_sim_mean_time_per_trial = [np.mean(sur_loaded_data[0][i].sim_time) for i in range(ntrials)]
    sur_sim_total_time_per_trial = [np.sum(sur_loaded_data[0][i].sim_time) for i in range(ntrials)]
    sur_sim_mean_time = np.mean(sur_sim_mean_time_per_trial)
    sur_sim_std_time = np.std(sur_sim_mean_time_per_trial)
    sur_sim_total_time = np.mean(sur_sim_total_time_per_trial)
    sur_sim_std_total_time = np.std(sur_sim_total_time_per_trial)


    x = ['Simulation', 'Simulation total', 'Surrogate', 'Ax', 'SA']
    if raw_data[2] == None: x.pop()
    t = [sur_sim_mean_time, sur_sim_total_time]+[np.mean(data[1]) for data in rdata]
    barlist = plt.bar(x=x, height=t)
    plt.errorbar(x, t, yerr=[sur_sim_std_time, sur_sim_std_total_time]+[np.std(data[1]) for data in rdata], fmt='none', color="grey")
    barlist[0].set_color('grey')
    barlist[1].set_color('grey')
    plt.ylabel('Execution time [s]')
    plt.yscale('symlog')

    if store: plt.savefig(f'../../surdata/Figures/compare_times_{topo}_iter-{niter}_symlog.pdf')
    plt.show()


if __name__ == '__main__':

    topo = sys.argv[1]
    niter = sys.argv[2]
    ntrials = int(sys.argv[3])

    path = f'../../surdata/*_{topo}_iter-{niter}*.pkl'
    files = [file for file in glob.glob(path)]
    assert(len(files)<=3), 'The received pattern was ambigious - there are more than three files.'

    sur_loaded_data, ax_loaded_data, sa_loaded_data = [None]*3
    for filename in files:
        if 'Sur' in filename:
            with open(filename, 'rb') as file: 
                sur_loaded_data = pickle.load(file)
        elif 'Ax' in filename:
            with open(filename, 'rb') as file: 
                ax_loaded_data = pickle.load(file)
        elif 'SA' in filename:
            with open(filename, 'rb') as file: 
                sa_loaded_data = pickle.load(file)
        else:
            print(files)
            raise Exception('Naming of files wrong, could not find Sur, Ax or Sa specifying the data.')

    raw_data_list = [sur_loaded_data, ax_loaded_data, sa_loaded_data]

    df_plot = get_comparison_dataframe(raw_data_list)
    
    plot_trial(df=df_plot, store=True)
    plt.show()

    plot_overall(df=df_plot, store=True)

    plot_avg_time(raw_data=raw_data_list, store=True)