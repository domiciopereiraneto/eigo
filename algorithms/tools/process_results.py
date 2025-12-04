import os
import sys

# Add parent directory to Python path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

import yaml
import argparse
from algorithms.tools.src.summary_table import create_summary_table
from algorithms.tools.src.image_grid import create_image_grid
from algorithms.tools.src.prompt_category_analysis import create_prompt_category_results_comparison
from algorithms.tools.src.distance_table import create_distance_table_and_plots
from algorithms.tools.src.evolution_plots import create_evolution_plots
from algorithms.tools.src.final_prompts import create_best_prompts_table

# Add argument parsing
parser = argparse.ArgumentParser(description='Process results with configuration file')
parser.add_argument('--config', type=str, default="algorithms/tools/config.yml",
                   help='Path to configuration YAML file')
args = parser.parse_args()

# Use the provided config path or default
config_path = args.config

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

source_dirs = config.get('source_dirs', [])
method_names = config.get('method_names', [])
save_folder = config.get('save_folder', 'results_processed')
algo_labels = config.get('algorithm_labels', [])
algo_labels = [tuple(label) for label in algo_labels] 
aesthetic_max = config.get('aesthetic_max', 10.0)
clip_max = config.get('clip_max', 0.5)


if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# GENERATE BEST PROMPTS COMPARISON TABLE
# ===============================
create_best_prompts_table(source_dirs, save_folder, algo_labels)

# GENERATE TABLE WITH AESTHETIC, CLIP AND FITNESS RESULTS AMONG DIFFERENT METHODS
# ===============================
#create_summary_table(source_dirs, save_folder, algo_labels, aesthetic_max, clip_max)

# GENERATE EVOLUTION PLOTS PER WEIGHT COMBINATION
# ===============================
#create_evolution_plots(source_dirs, save_folder, algo_labels, aesthetic_max, clip_max)

# GENERATE IMAGE GRID WITH THE BEST IMAGES FOR EACH METHOD
# ===============================
#create_image_grid(source_dirs, method_names, save_folder)

# GENERATE COMPARISON OF PROMPTS AND CATEGORIES
# ===============================
#create_prompt_category_results_comparison(source_dirs, save_folder)

# GENERATE DISTANCE TABLE + GROUPED PLOTS
# ===============================
#create_distance_table_and_plots(source_dirs, save_folder, algo_labels)
