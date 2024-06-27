import pandas as pd

class SurrogateCollector:
    """
    Collector designed for gathering and organizing data from surrogate-based optimization results.
    
    This class provides methods to retrieve various sets of data including model outputs,
    timing metrics, and machine learning performance scores, all structured into pandas DataFrames.

    Attributes:
        sim (Simulation): An instance of the Simulation class that provides methods 
                          and properties necessary for the simulation environment.
    """

    def __init__(self, sim):
        self.model = sim
        
    def get_model_df(self):
        """
        Constructs a DataFrame containing all relevant outputs from the simulation model.

        The DataFrame includes objective values, raw outputs, standardized outputs,
        and the inputs to the model, providing a comprehensive view of the model's performance.

        Returns:
            pd.DataFrame: A DataFrame combining input data with simulation outputs including
                          objective values, raw and standardized outputs.
        """
        self.y = pd.DataFrame.from_records(self.model.y)
        self.objective = self.y.sum(axis=1).rename('objective')
        self.y_raw = pd.DataFrame.from_records(self.model.y_raw).add_suffix('_raw')
        self.y_std = pd.DataFrame.from_records(self.model.y_std).add_suffix('_std')
        self.model_df = pd.concat([self.model.X_df, self.objective, self.y, self.y_std, self.y_raw], axis=1)
        return self.model_df

    def get_timing(self):
        """
        Retrieves and formats the timing information for the simulation.

        The method extracts timing data for the simulation setup, model building,
        and acquisition phases, and formats it into a readable DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing detailed timing information for different
                          phases of the simulation, suffixed with ' [s]' to denote seconds.
        """
        timing = {
            'Simulation': self.model.sim_time,
            'Build': self.model.build_time,
            'Acquisition': [0] + self.model.acquisition_time,
            'Total': self.model.optimize_time
        }
        self.timing = pd.DataFrame.from_dict(timing, orient='index').T.add_suffix(' [s]')
        return self.timing
    
    def get_machine_learning_scores(self):
        """
        Retrieves the performance scores of the machine learning models used in the simulation.

        The scores are extracted from the simulation model and formatted into a DataFrame
        to provide a clear view of each model's performance.

        Returns:
            pd.DataFrame: A DataFrame containing performance scores for each machine learning model.
        """
        self.ml_model_scores = pd.DataFrame.from_dict(self.model.model_scores, orient='index').T
        return self.ml_model_scores
    
    def get_total(self):
        """
        Compiles all collected data into a single comprehensive DataFrame.

        This method aggregates model outputs, timing data, and machine learning scores into one DataFrame,
        facilitating an integrated view of all results and metrics from the simulation.

        Returns:
            pd.DataFrame: A merged DataFrame containing all collected data, allowing for easy analysis and visualization.
        """
        self.get_model_df()
        self.get_timing()
        self.get_machine_learning_scores()
        self.total = self.model_df.merge(self.timing, left_on='Iteration', right_index=True)
        self.total = self.total.merge(self.ml_model_scores, left_on='Iteration', right_index=True)
        return self.total


def get_parameters(variables):
    """
    Extracts and formats parameters from a dictionary for use in the Ax-platform optimization tool.

    Parameters
    ----------
    variables : dict
        A dictionary where keys correspond to parameter types (e.g., 'range', 'ordinal', 'choice')
        and values provide the definitions of these parameters.

    Returns
    -------
    list
        A list of parameter definitions formatted for use in optimization routines,
        with each parameter represented as a dictionary detailing its name,
        type, and constraints or choices.
    """
    parameters = []
    for k in variables:
        for key,value in variables[k].items():
            typ = 'choice' if k == 'ordinal' else k
            if typ != 'choice':
                parameters.append(
                    {
                    "name": str(key),
                    "type": typ,
                    "bounds": value[0],
                    })
            else:
                parameters.append(
                    {
                    "name": str(key),
                    "type": typ,
                    "values": value,
                    })
    return parameters