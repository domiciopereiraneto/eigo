"""
CMA-ES-based optimization of text embeddings for image generation using SDXL.

This script employs the CMA-ES optimizer (including standard CMA-ES, sep-CMA-ES, or VD-CMA variants) to modify text embeddings while maximizing aesthetic and CLIP scores. It supports configuration through a YAML file and provides functionality for prompt sampling, image generation, and evaluation.

Main Features:
- Loads configuration parameters from a YAML file.
- Samples prompts from a dataset and groups them by category.
- Generates images using Stable Diffusion XL with optimized text embeddings.
- Evaluates images using aesthetic and CLIP scores.
- Saves results, including metrics and generated images, to an output folder.
- Provides visualization of score evolution over generations.

Dependencies:
- PyTorch for deep learning operations.
- diffusers for Stable Diffusion pipelines.
- PIL for image processing.
- datasets for loading prompt datasets.
- matplotlib for plotting results.
- pptx for generating PowerPoint presentations.
- cma for CMA-ES optimization.

Usage:
Run the script with a configuration file specifying the parameters:
    python cmaes.py --config path/to/config.yaml
"""

# System imports
import sys
import os
import shutil
import json
import yaml

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path to obtain access to the submodules
sys.path.insert(0, parent_dir)

# External imports - grouped by functionality
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
from pptx import Presentation
from pptx.util import Inches
from datasets import load_dataset
import argparse

from eigo import Eigo

# Argument parsing for configuration file
# Allows specifying a custom configuration file path.
parser = argparse.ArgumentParser(description='Run optimization with configuration file')
parser.add_argument('--config', type=str, default="algorithms/config/config_p2_experiments.yaml",
                   help='Path to configuration YAML file')
args = parser.parse_args()

# Use the provided config path or default
config_path = args.config

# Load configuration parameters
# Reads the YAML configuration file and extracts parameters for the optimization process.
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

SEED = config['seed']
SEED_PATH = config['seed_path']
OUTPUT_FOLDER = config['results_folder']

# Prompt dataset loading and preprocessing
# Groups prompts by category and samples a specified number per category.
prompt_dataset = load_dataset("nateraw/parti-prompts")["train"]

N_PER_CATEGORY = config['prompt_per_categorie']  # Number of prompts to sample per category
SUBSET_SEED = config['prompt_sample_seed']
random.seed(SUBSET_SEED)

# Group prompts by category
category_prompts = defaultdict(list)
for item in prompt_dataset:
    category = item.get("Category", "Uncategorized")
    category_prompts[category].append(item["Prompt"])

# Sample N_PER_CATEGORY prompts from each category and keep track of category
selected_prompts_with_category = []
for category, prompts in category_prompts.items():
    if len(prompts) >= N_PER_CATEGORY:
        sampled = random.sample(prompts, N_PER_CATEGORY)
    else:
        sampled = prompts  # If not enough, take all
    for prompt in sampled:
        selected_prompts_with_category.append((prompt, category))

# Save selected prompts to a file
# Stores the sampled prompts and their categories for reference.
# prompt_list_path = os.path.join(OUTPUT_FOLDER, "selected_prompts.txt")
# with open(prompt_list_path, "w", encoding="utf-8") as f:
#     for prompt, category in selected_prompts_with_category:
#         f.write(f"{category}\t{prompt}\n")
# print(f"Saved selected prompts to {prompt_list_path}")

print(f"Selected {len(selected_prompts_with_category)} prompts from {len(category_prompts)} categories.")

# Seed handling
# Initializes the random seed for reproducibility.
if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        # Read each line, strip newline characters, and convert to integers
        seed_list = [int(line.strip()) for line in file]


def aggregate_results_cmaes():
    """
    Combine results from multiple runs and calculate summary statistics.

    Aggregates fitness, aesthetic, and CLIP scores across different seeds and prompts.
    Saves the aggregated results to an Excel file and generates summary plots.
    """

    def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
        """
        Plot the mean and standard deviation with optional labels and title.

        Args:
            x_axis (iterable): The x-axis values.
            m_vec (iterable): The mean values.
            std_vec (iterable): The standard deviation values.
            description (str): A description for the plot legend.
            title (str): The title of the plot (optional).
            y_label (str): The label for the y-axis (optional).
            x_label (str): The label for the x-axis (optional).
        """
        lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
        upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]

        plt.plot(x_axis, m_vec, '--', label=description + " Avg.")
        plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3, label=description + " Avg. ± SD")
        if title is not None:
            plt.title(title)
        if y_label is not None:
            plt.ylabel(y_label)
        if x_label is not None:
            plt.xlabel(x_label)

    def save_plot_results(results, results_folder):
        """
        Generate and save plots for the evolution of scores and losses over generations.

        Args:
            results (pd.DataFrame): The DataFrame containing the results data.
            results_folder (str): The folder path to save the plots.
        """
        # Plot main fitness evolution
        plt.figure(figsize=(10, 6))  # Increase figure size
        plot_mean_std(results['generation'], results['avg_fitness'], results['std_fitness'], "Fitness")
        plt.plot(results['generation'], results['max_fitness'], 'r-', label="Best Fitness")
        #plot_mean_std(results['generation'], results['avg_fitness_1'], results['std_fitness_1'], "F1 (Aesthetic Score)")
        #plt.plot(results['generation'], results['max_fitness_1'], 'orange', label="Best F1")
        #plot_mean_std(results['generation'], results['avg_fitness_2'], results['std_fitness_2'], "F2 (CLIP Score)")
        #plt.plot(results['generation'], results['max_fitness_2'], 'green', label="Best F2")
        plt.ylim(0, 1.1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
        plt.tight_layout()  # Adjust layout
        plt.savefig(results_folder + "/fitness_evolution.png")

        # Plot aesthetic score evolution
        plt.figure(figsize=(10, 6))  # Increase figure size
        plot_mean_std(results['generation'], results['avg_aesthetic_score'], results['std_aesthetic_score'], "Population")
        plt.plot(results['generation'], results['max_aesthetic_score'], 'r-', label="Best")
        plt.ylim(0, 10)
        plt.xlabel('Generation')
        plt.ylabel('Aesthetic Score')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
        plt.tight_layout()  # Adjust layout
        plt.savefig(results_folder + "/aesthetic_score_evolution.png")

        # Plot clip score evolution
        plt.figure(figsize=(10, 6))  # Increase figure size
        plot_mean_std(results['generation'], results['avg_clip_score'], results['std_clip_score'], "Population")
        plt.plot(results['generation'], results['max_clip_score'], 'r-', label="Best")
        plt.ylim(0, 0.6)
        plt.xlabel('Generation')
        plt.ylabel('CLIP Score')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
        plt.tight_layout()  # Adjust layout
        plt.savefig(results_folder + "/clip_score_evolution.png")

    # Initialize aggregated_data as None
    aggregated_data = None

    # Iterate over all subdirectories
    for folder_name in os.listdir(OUTPUT_FOLDER):
        if folder_name.startswith(f"results_"):
            seed = folder_name.split("_")[-2]  # Extract the seed number
            prompt_number = folder_name.split("_")[-1]  # Extract the prompt number

            file_path = os.path.join(OUTPUT_FOLDER, folder_name, "fitness_results.csv")

            if os.path.exists(file_path):
                # Read the CSV file
                df = pd.read_csv(file_path)

                df = df.drop(columns=["prompt", "category"]) 

                df = df.rename(columns={"avg_fitness": f"avg_fitness_{seed}_{prompt_number}"})
                df = df.rename(columns={"max_fitness": f"max_fitness_{seed}_{prompt_number}"})
                df = df.rename(columns={"std_fitness": f"std_fitness_{seed}_{prompt_number}"})
                #df = df.rename(columns={"avg_fitness_1": f"avg_fitness_1_{seed}_{prompt_number}"})
                #df = df.rename(columns={"max_fitness_1": f"max_fitness_1_{seed}_{prompt_number}"})
                #df = df.rename(columns={"std_fitness_1": f"std_fitness_1_{seed}_{prompt_number}"})
                #df = df.rename(columns={"avg_fitness_2": f"avg_fitness_2_{seed}_{prompt_number}"})
                #df = df.rename(columns={"max_fitness_2": f"max_fitness_2_{seed}_{prompt_number}"})
                #df = df.rename(columns={"std_fitness_2": f"std_fitness_2_{seed}_{prompt_number}"})
                df = df.rename(columns={"avg_aesthetic_score": f"avg_aesthetic_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"max_aesthetic_score": f"max_aesthetic_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"std_aesthetic_score": f"std_aesthetic_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"avg_clip_score": f"avg_clip_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"max_clip_score": f"max_clip_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"std_clip_score": f"std_clip_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"elapsed_time": f"elapsed_time_{seed}_{prompt_number}"})

                if aggregated_data is None:
                    aggregated_data = df

                aggregated_data = pd.merge(aggregated_data, df, on="generation", how="outer")
            else:
                print(f"File not found: {file_path}")

    # Ensure aggregated_data is not None before saving
    if aggregated_data is not None:
        # Save the aggregated data to an Excel file
        output_file = os.path.join(OUTPUT_FOLDER, "aggregated_score_results.xlsx")
        aggregated_data.to_excel(output_file, index=False)
        print(f"Aggregated results saved to {output_file}")
    else:
        print("No data was aggregated. Check the input folders and files.")

    data = pd.read_excel(output_file)

    # Calculate the average fitness across all seeds for each iteration
    data['avg_fitness'] = data.filter(like='avg_fitness_').mean(axis=1)
    # Calculate the standard deviation of fitness across all seeds for each iteration
    data['std_fitness'] = data.filter(like='std_fitness_').std(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['best_avg_fitness'] = data.filter(like='max_fitness_').mean(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['best_std_fitness'] = data.filter(like='max_fitness_').std(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['max_fitness'] = data.filter(like='max_fitness_').max(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['avg_aesthetic_score'] = data.filter(like='avg_aesthetic_score_').mean(axis=1)
    # Calculate the standard deviation of fitness across all seeds for each iteration
    data['std_aesthetic_score'] = data.filter(like='std_aesthetic_score_').std(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['best_avg_aesthetic_score'] = data.filter(like='max_aesthetic_score_').mean(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['best_std_aesthetic_score'] = data.filter(like='max_aesthetic_score_').std(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['max_aesthetic_score'] = data.filter(like='max_aesthetic_score_').max(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['avg_clip_score'] = data.filter(like='avg_clip_score_').mean(axis=1)
    # Calculate the standard deviation of fitness across all seeds for each iteration
    data['std_clip_score'] = data.filter(like='std_clip_score_').std(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['best_avg_clip_score'] = data.filter(like='max_clip_score_').mean(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['best_std_clip_score'] = data.filter(like='max_clip_score_').std(axis=1)
    # Calculate the average fitness across all seeds for each iteration
    data['max_clip_score'] = data.filter(like='max_clip_score_').max(axis=1)

    # Fitness Evolution
    plt.figure(figsize=(10, 6))
    plot_mean_std(data['generation'], data['avg_fitness'], data['std_fitness'], "Population")
    plot_mean_std(data['generation'], data['best_avg_fitness'], data['best_std_fitness'], "Bests")
    plt.plot(data['generation'], data['max_fitness'], 'r-', label="Best")
    plt.ylim(0, 1.1)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid()
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "/fitness_evolution.png")
    plt.close()

    # Aesthetic Score Evolution
    plt.figure(figsize=(10, 6))
    plot_mean_std(data['generation'], data['avg_aesthetic_score'], data['std_aesthetic_score'], "Population")
    plot_mean_std(data['generation'], data['best_avg_aesthetic_score'], data['best_std_aesthetic_score'], "Bests")
    plt.plot(data['generation'], data['max_aesthetic_score'], 'r-', label="Best")
    plt.ylim(0, 10.5)
    plt.xlabel('Generation')
    plt.ylabel('Aesthetic Score')
    plt.grid()
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "/aesthetic_score_evolution.png")
    plt.close()

    # CLIP Score Evolution
    plt.figure(figsize=(10, 6))
    plot_mean_std(data['generation'], data['avg_clip_score'], data['std_clip_score'], "Population")
    plot_mean_std(data['generation'], data['best_avg_clip_score'], data['best_std_clip_score'], "Bests")
    plt.plot(data['generation'], data['max_clip_score'], 'r-', label="Best")
    plt.xlabel('Generation')
    plt.ylabel('CLIP Score')
    plt.ylim(0, 0.6)
    plt.grid()
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "/clip_score_evolution.png")
    plt.close()

    # Initialize PowerPoint presentation
    presentation = Presentation()

    # Collect folders with seed numbers
    folders = []
    for folder_name in os.listdir(OUTPUT_FOLDER):
        if folder_name.startswith("results_"):
            # Extract the seed number from the folder name
            seed_number = int(folder_name.split("_")[-2])  # Convert to integer for sorting
            prompt_number = int(folder_name.split("_")[-1])  # Extract prompt number
            folder_path = os.path.join(OUTPUT_FOLDER, folder_name)
            folders.append((seed_number, prompt_number, folder_path))

    # Sort folders by prompt number in ascending order
    folders.sort(key=lambda x: x[1])

    # Iterate over sorted folders
    for seed_number, prompt_number, folder_path in folders:
        # Paths for required images and CSV file
        it_0_path = os.path.join(folder_path, "it_0.png")
        best_all_path = os.path.join(folder_path, "best_all.png")
        fitness_evolution_path = os.path.join(folder_path, "fitness_evolution.png")
        aesthetic_evolution_path = os.path.join(folder_path, "aesthetic_score_evolution.png")
        clip_evolution_path = os.path.join(folder_path, "clip_score_evolution.png")
        csv_path = os.path.join(folder_path, "fitness_results.csv")

        # Extract scores and prompt from CSV
        fitness_initial = None
        fitness_best = None
        aesthetic_initial = None
        aesthetic_best = None
        clip_initial = None
        clip_best = None
        prompt_text = None
        category = None

        if os.path.exists(csv_path):
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                if rows:
                    # Initial values from the first row
                    first_row = rows[0]
                    fitness_initial = float(first_row['max_fitness'])
                    aesthetic_initial = float(first_row['max_aesthetic_score'])
                    clip_initial = float(first_row['max_clip_score'])
                    prompt_text = first_row['prompt']
                    category = first_row['category']

                    # Find the row with the best (maximum) max_fitness
                    best_row = max(rows, key=lambda r: float(r['max_fitness']))
                    fitness_best = float(best_row['max_fitness'])
                    aesthetic_best = float(best_row['max_aesthetic_score'])
                    clip_best = float(best_row['max_clip_score'])

        # Slide 1: it_0.png and it_1000.png
        if os.path.exists(it_0_path) and os.path.exists(best_all_path):
            slide = presentation.slides.add_slide(presentation.slide_layouts[5])  # Blank slide
            title = slide.shapes.title
            title.text = f"Seed {seed_number}"

            # Add prompt below the title
            if prompt_text:
                left = Inches(0.5)
                top = Inches(1)
                width = Inches(9)
                textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
                textbox.text = f"Prompt: {prompt_text}"

            if category:
                left = Inches(0.5)
                top = Inches(1.5)
                width = Inches(9)
                textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
                textbox.text = f"Category: {category}"

            # Add it_0.png
            slide.shapes.add_picture(it_0_path, Inches(0.5), Inches(2), height=Inches(4))

            # Add legend below it_0.png
            left = Inches(0.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "Initial iteration"
            if fitness_initial is not None:
                text += f"\nInitial Fitness: {fitness_initial:.4f}"
            if aesthetic_initial is not None:
                text += f"\nAesthetic Score: {aesthetic_initial:.4f}"
            if clip_initial is not None:
                text += f"\nCLIP Score: {clip_initial:.4f}"
            textbox.text = text

            # Add it_1000.png
            slide.shapes.add_picture(best_all_path, Inches(5.5), Inches(2), height=Inches(4))

            # Add legend below it_1000.png
            left = Inches(5.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "Best iteration"
            if fitness_best is not None:
                text += f"\nBest fitness: {fitness_best:.4f}"
            if aesthetic_best is not None:
                text += f"\nAesthetic Score: {aesthetic_best:.4f}"
            if clip_best is not None:
                text += f"\nCLIP Score: {clip_best:.4f}"
            textbox.text = text

        # Slide 2: fitness_evolution
        if os.path.exists(fitness_evolution_path):
            slide = presentation.slides.add_slide(presentation.slide_layouts[5])  # Blank slide
            title = slide.shapes.title
            title.text = f"Seed {seed_number}"

            # Add aesthetic_evolution.png
            slide.shapes.add_picture(fitness_evolution_path, Inches(0), Inches(2), height=Inches(4))

            left = Inches(0.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "Fitness evolution"
            textbox.text = text

        # Slide 3: clip_score_evolution.png and aesthetic_score_evolution.png
        if os.path.exists(clip_evolution_path) and os.path.exists(aesthetic_evolution_path):
            slide = presentation.slides.add_slide(presentation.slide_layouts[5])  # Blank slide
            title = slide.shapes.title
            title.text = f"Seed {seed_number}"

            # Add aesthetic_evolution.png
            slide.shapes.add_picture(clip_evolution_path, Inches(0), Inches(2), height=Inches(4))

            left = Inches(0.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "CLIP score evolution"
            textbox.text = text

            # Add loss_evolution.png
            slide.shapes.add_picture(aesthetic_evolution_path, Inches(5), Inches(2), height=Inches(4))

            left = Inches(5.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "Aesthetic score evolution"
            textbox.text = text

    output_filename = os.path.join(OUTPUT_FOLDER, f"summary.pptx")
    # Save the presentation
    presentation.save(output_filename)
    print(f"Presentation saved as {output_filename}")

def aggregate_results_adam():

    # Plot results
    # Generates and saves plots for the evolution of scores and losses over iterations.
    def plot_results(results, results_folder):
        plt.figure(figsize=(10, 6))  # Increase figure size
        plt.plot(results['iteration'], results['aesthetic_score'], label="Aesthetic Score")
        plt.xlabel('Iteration')
        plt.ylabel('Aesthetic Score')
        plt.title('Aesthetic Score Evolution')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
        plt.tight_layout()  # Adjust layout
        plt.savefig(results_folder + "/aesthetic_evolution.png")
        plt.close()

        plt.figure(figsize=(10, 6))  # Increase figure size
        plt.plot(results['iteration'], results['clip_score'], label="CLIP Score")
        plt.xlabel('Iteration')
        plt.ylabel('CLIP Score')
        plt.title('CLIP Score Evolution')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
        plt.tight_layout()  # Adjust layout
        plt.savefig(results_folder + "/clip_evolution.png")
        plt.close()

        # Plot all losses in one plot
        plt.figure(figsize=(10, 6))  # Increase figure size
        plt.plot(results['iteration'], results['combined_loss'], label="Combined Loss")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Evolution')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
        plt.tight_layout()  # Adjust layout
        plt.savefig(results_folder + "/loss_evolution.png")
        plt.close()

    def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
        lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
        upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]

        plt.plot(x_axis, m_vec, '--', label=description + " Avg.")
        plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3, label=description + " Avg. ± SD")
        if title is not None:
            plt.title(title)
        if y_label is not None:
            plt.ylabel(y_label)
        if x_label is not None:
            plt.xlabel(x_label)

    # Initialize aggregated_data as None
    aggregated_data = None

    # Variables to track the maximum final combined score
    max_final_combined_score = float('-inf')
    max_seed = None
    max_prompt_number = None

    # Iterate over all subdirectories
    for folder_name in os.listdir(OUTPUT_FOLDER):
        if folder_name.startswith(f"results_"):
            seed = folder_name.split("_")[-2]  # Extract the seed number
            prompt_number = folder_name.split("_")[-1]  # Extract the prompt number

            file_path = os.path.join(OUTPUT_FOLDER, folder_name, "score_results.csv")

            if os.path.exists(file_path):
                # Read the CSV file
                df = pd.read_csv(file_path)

                df = df.drop(columns=["prompt", "category"]) 

                df = df.rename(columns={"combined_score": f"combined_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"combined_loss": f"combined_loss_{seed}_{prompt_number}"})
                df = df.rename(columns={"aesthetic_score": f"aesthetic_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"clip_score": f"clip_score_{seed}_{prompt_number}"})
                df = df.rename(columns={"elapsed_time": f"elapsed_time_{seed}_{prompt_number}"})

                # Check the final combined score in this file
                max_score = df[f"combined_score_{seed}_{prompt_number}"].max()
                if max_score > max_final_combined_score:
                    max_final_combined_score = max_score
                    max_seed = seed
                    max_prompt_number = prompt_number

                if aggregated_data is None:
                    aggregated_data = df

                else:
                    aggregated_data = pd.merge(aggregated_data, df, on="iteration", how="outer")
            else:
                print(f"File not found: {file_path}")

    # Ensure aggregated_data is not None before saving
    if aggregated_data is not None:
        # Save the aggregated data to an Excel file
        output_file = os.path.join(OUTPUT_FOLDER, "aggregated_score_results.xlsx")
        aggregated_data.to_excel(output_file, index=False)
        print(f"Aggregated results saved to {output_file}")
    else:
        print("No data was aggregated. Check the input folders and files.")

    data = pd.read_excel(output_file)

    data['max_combined_score'] = aggregated_data[f"combined_score_{max_seed}_{max_prompt_number}"]
    data['max_combined_loss'] = aggregated_data[f"combined_loss_{max_seed}_{max_prompt_number}"]
    data['max_aesthetic_score'] = aggregated_data[f"aesthetic_score_{max_seed}_{max_prompt_number}"]
    data['max_clip_score'] = aggregated_data[f"clip_score_{max_seed}_{max_prompt_number}"]

    # Calculate the average metrics across all seeds for each iteration
    data['avg_combined_score'] = data.filter(like='combined_score_').mean(axis=1)
    data['avg_combined_loss'] = data.filter(like='combined_loss_').mean(axis=1)
    data['avg_aesthetic_score'] = data.filter(like='aesthetic_score_').mean(axis=1)
    data['avg_clip_score'] = data.filter(like='clip_score_').mean(axis=1)

    # Calculate the standard deviation for each metric
    data['std_combined_score'] = data.filter(like='combined_score_').std(axis=1)
    data['std_combined_loss'] = data.filter(like='combined_loss_').std(axis=1)
    data['std_aesthetic_score'] = data.filter(like='aesthetic_score_').std(axis=1)
    data['std_clip_score'] = data.filter(like='clip_score_').std(axis=1)

    # Loss Evolution
    plt.figure(figsize=(10, 6))
    plot_mean_std(data['iteration'], data['avg_combined_loss'], data['std_combined_loss'], "Loss")
    plt.plot(data['iteration'], data['max_combined_loss'], '-', label="Best")
    plt.ylim(0, 1.1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "/loss_evolution.png")
    plt.close()

    # Aesthetic Score Evolution
    plt.figure(figsize=(10, 6))
    plot_mean_std(data['iteration'], data['avg_aesthetic_score'], data['std_aesthetic_score'], "")
    plt.plot(data['iteration'], data['max_aesthetic_score'], '-', label="Best")
    plt.ylim(0, 10.5)
    plt.xlabel('Iteration')
    plt.ylabel('Aesthetic Score')
    plt.grid()
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "/aesthetic_score_evolution.png")
    plt.close()

    # CLIP Score Evolution
    plt.figure()
    plot_mean_std(data['iteration'], data['avg_clip_score'], data['std_clip_score'], "Population")
    plt.plot(data['iteration'], data['max_clip_score'], '-', label="Best")
    plt.ylim(0, 0.6)
    plt.xlabel('Iteration')
    plt.ylabel('CLIP Score')
    plt.grid()
    plt.legend()
    plt.savefig(OUTPUT_FOLDER + "/clip_score_evolution.png")
    plt.close()

    # Initialize PowerPoint presentation
    presentation = Presentation()

    # Collect folders with seed numbers
    folders = []
    for folder_name in os.listdir(OUTPUT_FOLDER):
        if folder_name.startswith("results_"):
            # Extract the seed number from the folder name
            seed_number = int(folder_name.split("_")[-2])  # Convert to integer for sorting
            prompt_number = int(folder_name.split("_")[-1])  # Extract prompt number
            folder_path = os.path.join(OUTPUT_FOLDER, folder_name)
            folders.append((seed_number, prompt_number, folder_path))

    # Sort folders by prompt number in ascending order
    folders.sort(key=lambda x: x[1])

    # Iterate over sorted folders
    for seed_number, prompt_number, folder_path in folders:
        # Paths for required images and CSV file
        it_0_path = os.path.join(folder_path, "it_0.png")
        best_all_path = os.path.join(folder_path, "best_all.png")
        loss_evolution_path = os.path.join(folder_path, "loss_evolution.png")
        score_evolution_path = os.path.join(folder_path, "aesthetic_evolution.png")
        clip_evolution_path = os.path.join(folder_path, "clip_evolution.png")
        csv_path = os.path.join(folder_path, "score_results.csv")

        # Extract scores and prompt from CSV
        combined_score_initial = None
        combined_score_best = None
        aesthetic_initial = None
        aesthetic_best = None
        clip_initial = None
        clip_best = None
        prompt_text = None
        category = None

        if os.path.exists(csv_path):
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                if rows:
                    # Initial values from the first row
                    first_row = rows[0]
                    combined_score_initial = float(first_row['combined_score'])
                    aesthetic_initial = float(first_row['aesthetic_score'])
                    clip_initial = float(first_row['clip_score'])
                    prompt_text = first_row['prompt']
                    category = first_row['category']

                    # Find the row with the best (maximum) max_fitness
                    best_row = max(rows, key=lambda r: float(r['combined_score']))
                    combined_score_best = float(best_row['combined_score'])
                    aesthetic_best = float(best_row['aesthetic_score'])
                    clip_best = float(best_row['clip_score'])

        # Slide 1: it_0.png and it_1000.png
        if os.path.exists(it_0_path) and os.path.exists(best_all_path):
            slide = presentation.slides.add_slide(presentation.slide_layouts[5])  # Blank slide
            title = slide.shapes.title
            title.text = f"Seed {seed_number}"

            # Add prompt below the title
            if prompt_text:
                left = Inches(0.5)
                top = Inches(1)
                width = Inches(9)
                textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
                textbox.text = f"Prompt: {prompt_text}"

            if category:
                left = Inches(0.5)
                top = Inches(1.5)
                width = Inches(9)
                textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
                textbox.text = f"Category: {category}"

            # Add it_0.png
            slide.shapes.add_picture(it_0_path, Inches(0.5), Inches(2), height=Inches(4))

            # Add legend below it_0.png
            left = Inches(0.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "Initial iteration"
            if combined_score_initial is not None:
                text += f"\nInitial Combined Score: {combined_score_initial:.4f}"
            if aesthetic_initial is not None:
                text += f"\nAesthetic Score: {aesthetic_initial:.4f}"
            if clip_initial is not None:
                text += f"\nCLIP Score: {clip_initial:.4f}"
            textbox.text = text

            # Add it_1000.png
            slide.shapes.add_picture(best_all_path, Inches(5.5), Inches(2), height=Inches(4))

            # Add legend below it_1000.png
            left = Inches(5.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "Best iteration"
            if combined_score_best is not None:
                text += f"\nBest Combined Score: {combined_score_best:.4f}"
            if aesthetic_best is not None:
                text += f"\nAesthetic Score: {aesthetic_best:.4f}"
            if clip_best is not None:
                text += f"\nCLIP Score: {clip_best:.4f}"
            textbox.text = text

        # Slide 2: loss_evolution.png
        if os.path.exists(loss_evolution_path) and os.path.exists(score_evolution_path):
            slide = presentation.slides.add_slide(presentation.slide_layouts[5])  # Blank slide
            title = slide.shapes.title
            title.text = f"Seed {seed_number}"

            # Add aesthetic_evolution.png
            slide.shapes.add_picture(loss_evolution_path, Inches(0), Inches(2), height=Inches(4))

            left = Inches(0.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "Loss evolution"
            textbox.text = text

        # Slide 3: aesthetic_evolution.png and loss_evolution.png
        if os.path.exists(loss_evolution_path) and os.path.exists(score_evolution_path):
            slide = presentation.slides.add_slide(presentation.slide_layouts[5])  # Blank slide
            title = slide.shapes.title
            title.text = f"Seed {seed_number}"

            # Add aesthetic_evolution.png
            slide.shapes.add_picture(clip_evolution_path, Inches(0), Inches(2), height=Inches(4))

            left = Inches(0.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "CLIP evolution"
            textbox.text = text

            # Add loss_evolution.png
            slide.shapes.add_picture(score_evolution_path, Inches(5), Inches(2), height=Inches(4))

            left = Inches(5.5)
            top = Inches(6.2)
            width = Inches(4)
            textbox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
            text = "Aesthetic evolution"
            textbox.text = text

    output_filename = os.path.join(OUTPUT_FOLDER, f"summary.pptx")
    # Save the presentation
    presentation.save(output_filename)
    print(f"Presentation saved as {output_filename}")

if __name__ == "__main__":
    # Entry point for the script
    # Parses arguments, loads configuration, and starts the optimization process.

    eigo_engine = Eigo(config)

    seed_number = 1
    for seed in seed_list:
        prompt_number = 1
        for prompt, category in selected_prompts_with_category:
            print(f"Running seed {seed}, prompt: {prompt} (Category: {category})")
            if config['optimization_method'] == "cmaes":
                eigo_engine.run_cmaes_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "adam":
                eigo_engine.run_adam_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            else:
                raise ValueError(f"Unknown optimization method: {config['optimization_method']}")
            print(f"Run with seed {seed} and prompt '{prompt}' finished!")
            #aggregate_results()
            prompt_number += 1
        seed_number += 1
