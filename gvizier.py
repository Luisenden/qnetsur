from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz

import numpy as np
from functools import partial

from optimizingcd import main_cd as simulation

vals = { # define fixed parameters for your simulation function
        'A': simulation.adjacency_tree(2,3),
        'protocol':'srs', 
        'p_gen': 0.9, 
        'p_swap':1,  
        'return_data':'avg', 
        'progress_bar': None,
        'total_time': 300,
        'N_samples' : 10,
        }

vars = { # define variables and bounds for your simulation function
        'M': [1, 10],
        'qbits_per_channel': [1,50],
        'cutoff':[1.,10.],
        'q_swap': [0., 1.],
        'p_cons':[0.01, 0.2]
        } 

def evaluate(M,qbits_per_channel,cutoff,q_swap,p_cons) -> float:
  sim = partial(simulation.simulation_cd, **vals)
  result = sim(M=M,qbits_per_channel=qbits_per_channel,cutoff=cutoff,q_swap=q_swap,p_cons=p_cons)
  return np.mean([node[-1] for node in result[1]])

problem = vz.ProblemStatement()
problem.search_space.root.add_int_param('M', 1, 10)
problem.search_space.root.add_int_param('qbits_per_channel', 1, 50)
problem.search_space.root.add_float_param('cutoff', 1.0, 10.0)
problem.search_space.root.add_float_param('q_swap', 0.0, 1.0)
problem.search_space.root.add_float_param('p_cons', 0.01, 0.2)
problem.metric_information.append(
    vz.MetricInformation(
        name='maximize_metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

study_config = vz.StudyConfig.from_problem(problem)
study_config.algorithm = 'GAUSSIAN_PROCESS_BANDIT'

study_client = clients.Study.from_study_config(study_config, owner='LP', study_id='cd0')
print('Local SQL database file located at: ', service.VIZIER_DB_PATH)

for optimal_trial in study_client.optimal_trials():
  optimal_trial = optimal_trial.materialize()
  print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
        optimal_trial.final_measurement)