# EIGO engine initialization 

import sys
import os

# Add parent directory to Python path for module imports
# This allows importing modules from the parent directory.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from eigo import Eigo
import yaml

# Use the provided config path or default
config_path = "algorithms/config/config_eigo.yaml"

# Load configuration parameters
# Reads the YAML configuration file and extracts parameters for the optimization process.
with open(config_path, 'r') as file:
    experimental_setup_parameters = yaml.safe_load(file)

eigo_engine = Eigo(experimental_setup_parameters)

#Optimization execution

print("Starting optimization using method:", experimental_setup_parameters["optimization_method"])

if experimental_setup_parameters["optimization_method"] == "cmaes":
    results_folder = eigo_engine.run_cmaes_optimization()
elif experimental_setup_parameters["optimization_method"] == "adam":
    results_folder = eigo_engine.run_adam_optimization()

print("Results saved in folder:", results_folder)