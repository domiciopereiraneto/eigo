"""Run one EIGO optimization job from algorithms/config/config_eigo.yaml.

Use this entrypoint for quick single-prompt checks before launching larger
scheduled experiments. For batched paper runs, use run_experiments.py together
with experiments_schedule.py.
"""
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from eigo import Eigo
import yaml
config_path = 'algorithms/config/config_eigo.yaml'
with open(config_path, 'r') as file:
    experimental_setup_parameters = yaml.safe_load(file)
eigo_engine = Eigo(experimental_setup_parameters)
print('Starting optimization using method:', experimental_setup_parameters['optimization_method'])
if experimental_setup_parameters['optimization_method'] == 'ga':
    results_folder = eigo_engine.run_ga_optimization()
elif experimental_setup_parameters['optimization_method'] == 'random_sampler':
    results_folder = eigo_engine.run_random_sampler_optimization()
else:
    if experimental_setup_parameters['optimization_method'] == 'gomea':
        results_folder = eigo_engine.run_gomea_optimization()
    else:
        raise ValueError(f"Unknown optimization method: {experimental_setup_parameters['optimization_method']}")
print('Results saved in folder:', results_folder)
