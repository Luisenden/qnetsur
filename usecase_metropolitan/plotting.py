"""
Plotting tools to reproduce figures and tables."
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import argparse

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

def plot_from_exhaustive(folder)->None:
    """
    Reads the best-found solutions and merges with results from CSV files 
    (output data of `surrogate.py`, `vs_meta.py`, etc.)
    within the given folder, processes the data, and creates a point plot to compare
    the aggregated completed requests across different methods.

    Parameters:
    folder (str): The folder containing the CSV files with the results.

    Returns:
    None
    """
    x_df= pd.read_csv(folder+'Best_found_solutions.csv',index_col=0)

    method_names = ['Surrogate', 'Meta', 'Simulated Annealing',\
                    'Random Search', 'Even', 'Wu et. al, 2021']
    dfs = [None]*6
    for name in glob.glob(f'{folder}/Results_*.csv'):
        df = pd.read_csv(name, index_col=0).reset_index()
        method = df.Method[0]
        index = method_names.index(method) # Find the index of the method in the predefined list
        dfs[index] = df

    df = pd.concat(dfs, axis=0)
    df = df.melt(id_vars=['Method', 'Aggregated Completed Requests'],\
                var_name='User', value_name='Number of Completed Requests')
    df['User'] = df['User'].apply(lambda x: str.replace(x, 'Node', '')) # Replace 'Node' with an empty string in the User column
    df = df.merge(x_df, left_on='Method', right_index=True, how='left')
    markers = ['o', '^', 'v', 's', 'd', 'P']
    fig, axs = plt.subplots(1,1, figsize=(5,3))
    sns.pointplot(data= df, x='Total Number of Allocated Memories', y='Aggregated Completed Requests',\
                  hue='Method', ax=axs, errorbar='se', markers=markers, legend=True, linestyles=['']*6, native_scale=True)
    axs.grid()
    plt.ylabel('Total Number of Completed Requests')
    plt.tight_layout()
    plt.show()

def get_performance_distribution_per_method(folder)->pd.DataFrame:
    """
    Computes the performance distribution statistics per method.

    This function reads performance data from CSV files for different methods,
    calculates the maximum utility per trial, and returns the minimum, maximum,
    mean, standard deviation, and relative standard deviation of these utilities
    found per trial for each method.

    Parameters:
    folder (str): The folder containing the CSV files with performance data.

    Returns:
    pd.DataFrame: A dataframe with the performance distribution statistics per method.
    """
    filename = '_rb_starlight_budget_25h.csv'
    df_sur = pd.read_csv(folder+'SU'+filename)
    df_meta = pd.read_csv(folder+'AX'+filename)
    df_sa = pd.read_csv(folder+'SA'+filename)
    df_rs = pd.read_csv(folder+'RS'+filename)
    
    df_sur['Utility'] = df_sur['objective']
    df_rs['Utility'] = df_rs['objective']

    columns = ['Trial', 'Method', 'Utility']
    df = pd.concat([df_sur[columns], df_meta[columns], df_sa[columns], df_rs[columns]])
    max_per_trial = df.groupby(['Method', 'Trial'])['Utility'].max()
    mean_std = max_per_trial.groupby(level='Method').agg(['min', 'max', 'mean', 'std'])
    mean_std['rel_std'] = mean_std['std']/mean_std['mean']
    return mean_std

def get_policies(folder)->tuple:
    """
    Retrieves the best policy configurations and their corresponding information.

    This function reads policy data from CSV files for different methods, identifies the
    best policy based on the objective value, and returns two dataframes: one with the
    best policies and another with detailed information about these policies.

    Parameters:
    folder (str): The folder containing the CSV files with policy data.

    Returns:
    tuple: A tuple containing two dataframes:
        - policies (pd.DataFrame): The best policies for each method.
        - infos (pd.DataFrame): Information about the best policies.
    """
    policies = []
    infos = []
    mapping = {'Surrogate':'SU', 'Meta':'AX', 'Simulated Annealing':'SA','Random Search':'RS'}
    for key, value in mapping.items():
        dfs = []
        for i,name in enumerate(glob.glob(folder + f'/{value}_*.csv')): 
            with open(name,'rb') as file: dfs.append(pd.read_csv(file, index_col=0))
            dfs[i]['Trial'] = i # Add trial number to the dataframe
        df = pd.concat(dfs, axis=0).reset_index()
        df['Method'] = key # Add method name to the dataframe
        df['Aggregated Memories'] = df[df.columns[df.columns.str.contains('mem_size')]].sum(axis=1)
        best_index = df['objective'].idxmax() # Find the index of the best policy
        infos.append(df.iloc[best_index]) # Append best policy info
        policies.append(df.iloc[best_index][df.columns[df.columns.str.contains('mem_size|Method')]])
    
    infos = pd.DataFrame.from_records(infos)
    policies = pd.DataFrame.from_records(policies)
    return policies, infos

def get_surrogate_timeprofiling(file)->tuple:
    """
    Analyzes the time profiling data for surrogate optimization.

    This function reads time profiling data from a CSV file, calculates the average time
    spent in each phase relative to the total time, and returns the raw times, the relative
    times, and the average number of phases per trial.

    Parameters:
    file (str): The CSV file containing the measured times.

    Returns:
    tuple: A tuple containing three elements:
        - times (pd.DataFrame): The raw time profiling data.
        - relative (pd.Series): The relative times for each phase.
        - mean_phase_count (float): The average number of phases per trial.
    """
    times = pd.read_csv(file)
    times = times[times.columns[times.columns.str.contains(r'\[s\]|Trial')]]
    times = times.drop_duplicates(ignore_index=True)
    relative = times.drop('Trial', axis=1).agg('mean')/times.drop('Trial', axis=1).agg('mean')['Total [s]']
    return times, relative, np.mean(np.mean(times.groupby('Trial').count()))

def print_policies(file)->None:
    """
    Reads and prints policy configurations from a CSV file (e.g., generated by `get_policies`).

    This function reads policy data from a CSV file, processes it to remove irrelevant
    columns and transpose the dataframe, and then prints the resulting dataframe.

    Parameters:
    file (str): The CSV file containing the policy data.

    Returns:
    None
    """
    df = pd.read_csv(file, index_col=0).T
    df = df.drop('Total Number of Allocated Memories').reset_index()
    df = df.drop('index', axis=1).T
    df.columns = ["NU", "StarLight", "UChicago_PME", "UChicago_HC", "Fermilab_1", \
                  "Fermilab_2", "Argonne_1", "Argonne_2", "Argonne_3"]
    df = df.reset_index(names='Method').T
    print(df)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="set directory to output data")
    
    parser.add_argument(
        "--folder",
        type=str,
        help="Set path/to/QNETSUR-DATA. Type: str"
    )

    # Parse 
    args, _ = parser.parse_known_args()
    folder = args.folder
    base_dir = os.path.expanduser("~")

    # best found solutions (Supplementary Notes)
    file_path = os.path.join(base_dir, folder, 'Best_found_solutions.csv')
    best_solutions= pd.read_csv(file_path, index_col=0)
    print('Best Found Solutions:\n', best_solutions)
    
    # exhaustive run results (main text)
    plot_from_exhaustive(folder)

    # performance distribution (Supplementary Notes)
    print('\nDistribution of objective values:')
    distr = get_performance_distribution_per_method(folder)
    print('\n', distr)

    # time profiling (Supplementary Notes)
    file_path_timing = os.path.join(folder, 'SU_rb_starlight_budget_25h.csv')
    times, relative, cycles = get_surrogate_timeprofiling(file_path_timing)
    print('\nOverall:\n', times)
    print('Relative:\n', relative)
    print('Mean number of cycles:', cycles)
    print('\n')

    # overview found policies (Supplementary Notes)
    file_path_pol = os.path.join(folder, 'Best_found_solutions.csv')
    print('Overview policies:')
    print_policies(file_path_pol)
