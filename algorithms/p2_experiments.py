"""
Population-based optimization of text embeddings for image generation using SDXL.

This script employs a population-based optimizer (CMA-ES, GA, or Adam) to modify text embeddings while maximizing aesthetic and CLIP scores. It supports configuration through a YAML file and provides functionality for prompt sampling, image generation, and evaluation.

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
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
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

# Seed handling
# Initializes the random seed for reproducibility.
if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        # Read each line, strip newline characters, and convert to integers
        seed_list = [int(line.strip()) for line in file]

# Prompt dataset loading and preprocessing
# Supports either per-category prompt sampling or a single prompt per seed.
prompt_dataset = load_dataset("nateraw/parti-prompts")["train"]

N_PER_CATEGORY = config['prompt_per_categorie']  # Number of prompts to sample per category
SUBSET_SEED = config['prompt_sample_seed']
SINGLE_PROMPT_PER_SEED = config.get("single_prompt_per_seed", False)

# Group prompts by category and also keep a flat prompt/category list.
category_prompts = defaultdict(list)
all_prompts_with_category = []
for item in prompt_dataset:
    category = item.get("Category", "Uncategorized")
    prompt = item["Prompt"]
    category_prompts[category].append(prompt)
    all_prompts_with_category.append((prompt, category))


def build_selected_prompts(seed):
    if SINGLE_PROMPT_PER_SEED:
        rng = random.Random(seed)
        selected_prompt = rng.choice(all_prompts_with_category)
        print(
            f"Selected 1 prompt for seed {seed} from {len(all_prompts_with_category)} total prompts "
            f"across {len(category_prompts)} categories."
        )
        return [selected_prompt]

    rng = random.Random(SUBSET_SEED)
    selected_prompts_with_category = []
    for category, prompts in category_prompts.items():
        if len(prompts) >= N_PER_CATEGORY:
            sampled = rng.sample(prompts, N_PER_CATEGORY)
        else:
            sampled = prompts  # If not enough, take all
        for prompt in sampled:
            selected_prompts_with_category.append((prompt, category))

    print(
        f"Selected {len(selected_prompts_with_category)} prompts from {len(category_prompts)} categories."
    )
    return selected_prompts_with_category



def _first_non_empty(series):
    for value in series.dropna():
        text = str(value).strip()
        if text:
            return text
    return None


def _find_experiment_dirs(output_folder):
    experiment_dirs = set()
    for root, _, files in os.walk(output_folder):
        has_csv = "score_results.csv" in files or "fitness_results.csv" in files
        if not has_csv:
            continue
        prompt_dir = os.path.abspath(root)
        prompt_name = os.path.basename(prompt_dir)
        if not prompt_name.startswith("results_"):
            continue
        experiment_dirs.add(os.path.dirname(prompt_dir))
    return sorted(experiment_dirs)


def _load_result_runs(experiment_dir):
    runs = []
    for entry in sorted(os.listdir(experiment_dir)):
        prompt_dir = os.path.join(experiment_dir, entry)
        if not os.path.isdir(prompt_dir) or not entry.startswith("results_"):
            continue

        score_csv = os.path.join(prompt_dir, "score_results.csv")
        fitness_csv = os.path.join(prompt_dir, "fitness_results.csv")

        if os.path.exists(score_csv):
            csv_path = score_csv
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = df.sort_values("iteration").drop_duplicates(subset=["iteration"], keep="last")
            prompt = _first_non_empty(df["prompt"]) if "prompt" in df.columns else None
            category = _first_non_empty(df["category"]) if "category" in df.columns else None
            runs.append({
                "path": csv_path,
                "prompt_dir": prompt_dir,
                "prompt": prompt or os.path.basename(prompt_dir),
                "category": category or "",
                "x_label": "iteration",
                "x": pd.to_numeric(df["iteration"], errors="coerce").to_numpy(dtype=float),
                "time": pd.to_numeric(df["elapsed_time"], errors="coerce").to_numpy(dtype=float),
                "aesthetic": pd.to_numeric(df["aesthetic_score"], errors="coerce").to_numpy(dtype=float),
                "clip": pd.to_numeric(df["clip_score"], errors="coerce").to_numpy(dtype=float),
                "objective": pd.to_numeric(df["combined_loss"], errors="coerce").to_numpy(dtype=float),
                "objective_name": "loss",
            })

        if os.path.exists(fitness_csv):
            csv_path = fitness_csv
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = df.sort_values("generation").drop_duplicates(subset=["generation"], keep="last")
            prompt = _first_non_empty(df["prompt"]) if "prompt" in df.columns else None
            category = _first_non_empty(df["category"]) if "category" in df.columns else None
            runs.append({
                "path": csv_path,
                "prompt_dir": prompt_dir,
                "prompt": prompt or os.path.basename(prompt_dir),
                "category": category or "",
                "x_label": "generation",
                "x": pd.to_numeric(df["generation"], errors="coerce").to_numpy(dtype=float),
                "time": pd.to_numeric(df["elapsed_time"], errors="coerce").to_numpy(dtype=float),
                "aesthetic": pd.to_numeric(df["max_aesthetic_score"], errors="coerce").to_numpy(dtype=float),
                "clip": pd.to_numeric(df["max_clip_score"], errors="coerce").to_numpy(dtype=float),
                "objective": pd.to_numeric(df["max_fitness"], errors="coerce").to_numpy(dtype=float),
                "objective_name": "fitness",
            })

    cleaned_runs = []
    for run in runs:
        valid = np.isfinite(run["x"]) & np.isfinite(run["time"])
        valid = valid & np.isfinite(run["aesthetic"]) & np.isfinite(run["clip"]) & np.isfinite(run["objective"])
        if valid.sum() == 0:
            continue
        cleaned_runs.append({
            **run,
            "x": run["x"][valid],
            "time": run["time"][valid],
            "aesthetic": run["aesthetic"][valid],
            "clip": run["clip"][valid],
            "objective": run["objective"][valid],
        })
    return cleaned_runs


def _stack_on_axis(runs, key, axis_key, axis_values):
    stacked = []
    for run in runs:
        if axis_key == "x":
            idx = run["x"].astype(int)
            series = pd.Series(run[key], index=idx)
            row = series.reindex(axis_values).to_numpy(dtype=float)
        else:
            row = np.interp(axis_values, run["time"], run[key], left=np.nan, right=np.nan)
            mask = (axis_values >= run["time"].min()) & (axis_values <= run["time"].max())
            row = np.where(mask, row, np.nan)
        stacked.append(row)
    return np.array(stacked, dtype=float)


def _compute_similarity_for_run(prompt_dir, clip_model, clip_preprocess, clip_device):
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    import torch

    base_path = os.path.join(prompt_dir, "it_0.png")
    best_path = os.path.join(prompt_dir, "best_all.png")
    if not (os.path.exists(base_path) and os.path.exists(best_path)):
        return np.nan, np.nan

    with Image.open(base_path).convert("RGB") as img0:
        t0 = clip_preprocess(img0).unsqueeze(0).to(clip_device)
    with Image.open(best_path).convert("RGB") as img1:
        t1 = clip_preprocess(img1).unsqueeze(0).to(clip_device)

    with torch.no_grad():
        v0 = clip_model.encode_image(t0).float().cpu().numpy().flatten()
        v1 = clip_model.encode_image(t1).float().cpu().numpy().flatten()

    den = float(np.linalg.norm(v0) * np.linalg.norm(v1))
    cosine_similarity = float(np.dot(v0, v1) / den) if den > 0 else np.nan

    with Image.open(base_path).convert("L") as img0:
        g0 = np.array(img0, dtype=np.uint8)
    with Image.open(best_path).convert("L") as img1:
        g1 = np.array(img1, dtype=np.uint8)

    h = min(g0.shape[0], g1.shape[0])
    w = min(g0.shape[1], g1.shape[1])
    ssim_value = float(ssim(g0[:h, :w], g1[:h, :w], data_range=255))
    return cosine_similarity, ssim_value


def _compute_similarity_metrics(runs):
    cos_vals = []
    ssim_vals = []
    try:
        import torch
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        for run in runs:
            cos_sim, ssim_value = _compute_similarity_for_run(
                run["prompt_dir"], clip_model, clip_preprocess, device
            )
            cos_vals.append(cos_sim)
            ssim_vals.append(ssim_value)
    except Exception as exc:
        print(f"Warning: similarity metrics unavailable ({exc}). Filling with NaN.")
        cos_vals = [np.nan] * len(runs)
        ssim_vals = [np.nan] * len(runs)
    return cos_vals, ssim_vals


def _plot_similarity_boxplot(values_df, output_path):
    metrics = ["cosine_similarity", "ssim"]
    data = [pd.to_numeric(values_df[m], errors="coerce").dropna().to_numpy(dtype=float) for m in metrics]
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=["Cosine Similarity", "SSIM"], showmeans=True)
    plt.ylabel("Similarity")
    plt.title("Image Similarity (it_0 vs best_all)")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_stats(axis, values, axis_name):
    return pd.DataFrame({
        axis_name: axis,
        "count": np.sum(~np.isnan(values), axis=0),
        "avg": np.nanmean(values, axis=0),
        "std": np.nanstd(values, axis=0),
        "min": np.nanmin(values, axis=0),
        "max": np.nanmax(values, axis=0),
    })


def _plot_evolution(stats_df, x_col, y_name, title, output_path):
    x = stats_df[x_col].to_numpy(dtype=float)
    mean = stats_df["avg"].to_numpy(dtype=float)
    std = stats_df["std"].to_numpy(dtype=float)
    min_v = stats_df["min"].to_numpy(dtype=float)
    max_v = stats_df["max"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, label="avg")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="avg ± std")
    plt.plot(x, min_v, "--", label="min")
    plt.plot(x, max_v, "--", label="max")
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_name.title())
    plt.title(title)
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _summary_rows(df, group_name):
    row = {"prompt": group_name, "n_runs": len(df)}
    for metric in ["aesthetic", "clip", "objective", "cosine_similarity", "ssim"]:
        values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        row[f"{metric}_avg"] = float(np.nanmean(values))
        row[f"{metric}_std"] = float(np.nanstd(values))
        row[f"{metric}_min"] = float(np.nanmin(values))
        row[f"{metric}_max"] = float(np.nanmax(values))
    return row


def _aggregate_one_experiment(experiment_dir):
    runs = _load_result_runs(experiment_dir)
    if not runs:
        print(f"No score files found under experiment folder: {experiment_dir}")
        return

    objective_names = sorted(set(run["objective_name"] for run in runs))
    x_labels = sorted(set(run["x_label"] for run in runs))
    if len(objective_names) != 1 or len(x_labels) != 1:
        raise ValueError(
            "Mixed run types found (Adam and CMA-ES together). "
            "Run aggregation on one method folder at a time."
        )

    axis_name = x_labels[0]
    objective_name = objective_names[0]
    x_axis = np.arange(0, int(max(np.max(run["x"]) for run in runs)) + 1)
    max_time = max(np.max(run["time"]) for run in runs)
    time_axis = np.linspace(0, max_time, 200) if max_time > 0 else np.array([0.0])

    for metric, label in [("aesthetic", "aesthetic"), ("clip", "clip"), ("objective", objective_name)]:
        step_values = _stack_on_axis(runs, metric, "x", x_axis)
        step_stats = _build_stats(x_axis, step_values, axis_name)
        step_stats.to_csv(os.path.join(experiment_dir, f"{label}_evolution_by_{axis_name}.csv"), index=False)
        _plot_evolution(
            step_stats,
            axis_name,
            label,
            f"{label.title()} evolution by {axis_name}",
            os.path.join(experiment_dir, f"{label}_evolution_by_{axis_name}.png"),
        )

        time_values = _stack_on_axis(runs, metric, "time", time_axis)
        time_stats = _build_stats(time_axis, time_values, "elapsed_time_seconds")
        time_stats.to_csv(os.path.join(experiment_dir, f"{label}_evolution_by_time.csv"), index=False)
        _plot_evolution(
            time_stats,
            "elapsed_time_seconds",
            label,
            f"{label.title()} evolution by elapsed time",
            os.path.join(experiment_dir, f"{label}_evolution_by_time.png"),
        )

    cosine_similarity, ssim_vals = _compute_similarity_metrics(runs)
    final_rows = []
    for run, cos_sim, ssim_value in zip(runs, cosine_similarity, ssim_vals):
        final_rows.append({
            "prompt": run["prompt"],
            "category": run["category"],
            "aesthetic": float(run["aesthetic"][-1]),
            "clip": float(run["clip"][-1]),
            "objective": float(run["objective"][-1]),
            "cosine_similarity": cos_sim,
            "ssim": ssim_value,
        })
    finals_df = pd.DataFrame(final_rows)

    summary_rows = []
    for prompt, group in finals_df.groupby("prompt", dropna=False):
        summary_rows.append(_summary_rows(group, str(prompt)))
    summary_rows.append(_summary_rows(finals_df, "TOTAL"))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.insert(1, "objective_name", objective_name)
    summary_df.to_csv(os.path.join(experiment_dir, "aggregate_prompt_summary.csv"), index=False)
    finals_df.to_csv(os.path.join(experiment_dir, "aggregate_prompt_similarity_values.csv"), index=False)
    _plot_similarity_boxplot(finals_df, os.path.join(experiment_dir, "similarity_boxplot.png"))
    print(f"Aggregation completed for {len(runs)} runs in {experiment_dir}")


def aggregate_results(output_folder=None):
    output_folder = output_folder or OUTPUT_FOLDER
    experiment_dirs = _find_experiment_dirs(output_folder)
    if not experiment_dirs:
        print(f"No experiment folders found under: {output_folder}")
        return
    for experiment_dir in experiment_dirs:
        _aggregate_one_experiment(experiment_dir)


if __name__ == "__main__":
    # Entry point for the script
    # Parses arguments, loads configuration, and starts the optimization process.

    eigo_engine = Eigo(config)

    seed_number = 1
    for seed in seed_list:
        selected_prompts_with_category = build_selected_prompts(seed)
        prompt_number = 1
        for prompt, category in selected_prompts_with_category:
            print(f"Running seed {seed}, prompt: {prompt} (Category: {category})")
            if config['optimization_method'] == "cmaes":
                eigo_engine.run_cmaes_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "ga":
                eigo_engine.run_ga_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "adam":
                eigo_engine.run_adam_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            else:
                raise ValueError(f"Unknown optimization method: {config['optimization_method']}")
            print(f"Run with seed {seed} and prompt '{prompt}' finished!")
            prompt_number += 1
        seed_number += 1

    aggregate_results()
