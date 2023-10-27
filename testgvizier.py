from vizier.service import clients
from vizier.service import pyvizier as vz

# Objective function to maximize.
def evaluate(w: float, x: int, y: float, z: str) -> float:
  return w**2 - y**2 + x * ord(z)

def evaluate(w: float) -> float:
  return w**2

# Algorithm, search space, and metrics.
study_config = vz.StudyConfig(algorithm='GAUSSIAN_PROCESS_BANDIT')
study_config.search_space.root.add_float_param('w', 0.0, 1.0)
# study_config.search_space.root.add_int_param('x', -2, 2)
# study_config.search_space.root.add_discrete_param('y', [0.3, 7.2])
# study_config.search_space.root.add_categorical_param('z', ['a', 'g', 'k'])
study_config.metric_information.append(vz.MetricInformation('metric_name', goal=vz.ObjectiveMetricGoal.MINIMIZE))

# Setup client and begin optimization. Vizier Service will be implicitly created.
study = clients.Study.from_study_config(study_config, owner='my_name', study_id='example')
print(study)
suggestions = study.suggest(count=1)
print(suggestions)
for suggestion in suggestions:
  params = suggestion.parameters
  objective = evaluate(params['w'], params['x'], params['y'], params['z'])
  suggestion.complete(vz.Measurement({'metric_name': objective}))