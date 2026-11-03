#!/usr/bin/env python3
"""Run EIGO over a sampled prompt dataset.

The config controls the diffusion backend, scoring objective, optimizer, prompt
dataset, prompt subset, seeds, and output folder. Each selected prompt is passed
to the shared Eigo backend and saved as an independent prompt-level result
directory inside the configured results folder.

Example:
    python algorithms/run_experiments.py --config algorithms/config/config_run_experiments.yaml
"""

import sys
import os
import shutil
import json
import yaml
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root so local project modules can be imported.
sys.path.insert(0, parent_dir)

import random
from collections import defaultdict
from datasets import load_dataset
import argparse

from eigo import Eigo

parser = argparse.ArgumentParser(description='Run optimization with configuration file')
parser.add_argument('--config', type=str, default="algorithms/config/config_run_experiments.yaml",
                   help='Path to configuration YAML file')
parser.add_argument('--resume_experiment', '--resume-experiment',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Extend checkpointed runs to the new configured total')
args = parser.parse_args()

config_path = args.config

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

if args.resume_experiment is not None:
    config['resume_experiment'] = args.resume_experiment
if not isinstance(config.get('resume_experiment', False), bool):
    raise ValueError('resume_experiment must be a boolean.')

SEED = config['seed']
SEED_PATH = config['seed_path']
OUTPUT_FOLDER = config['results_folder']

if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        seed_list = [int(line.strip()) for line in file]

# Prompt dataset presets define the default Hugging Face dataset path, split,
# prompt column aliases, and category column aliases for supported benchmark
# datasets. Config fields can override any of these defaults.
PROMPT_DATASET_PRESETS = {
    "parti": {
        "path": "nateraw/parti-prompts",
        "split": "train",
        "prompt_columns": ("Prompt", "prompt", "Prompts", "prompts"),
        "category_columns": ("Category", "category"),
    },
    "parti_prompts": {
        "path": "nateraw/parti-prompts",
        "split": "train",
        "prompt_columns": ("Prompt", "prompt", "Prompts", "prompts"),
        "category_columns": ("Category", "category"),
    },
    "drawbench": {
        "path": "sayakpaul/drawbench",
        "split": "train",
        "prompt_columns": ("Prompts", "Prompt", "prompt", "prompts"),
        "category_columns": ("Category", "category"),
    },
}


def _normalize_prompt_dataset_name(name):
    return str(name).strip().lower().replace("-", "_")


def _first_present_value(item, candidate_columns):
    for column in candidate_columns:
        if column in item and item[column] is not None:
            value = str(item[column]).strip()
            if value:
                return value
    return None


def _load_prompt_dataset(config):
    dataset_name = _normalize_prompt_dataset_name(config.get("prompt_dataset", "parti"))
    preset = PROMPT_DATASET_PRESETS.get(dataset_name)
    if preset is None:
        valid_names = ", ".join(sorted(PROMPT_DATASET_PRESETS))
        raise ValueError(
            f"Invalid prompt_dataset '{dataset_name}'. Expected one of: {valid_names}."
        )

    dataset_path = config.get("prompt_dataset_path", None) or preset["path"]
    dataset_config = config.get("prompt_dataset_config", None)
    dataset_split = config.get("prompt_dataset_split", None) or preset["split"]
    prompt_columns = config.get("prompt_column", None) or preset["prompt_columns"]
    category_columns = config.get("prompt_category_column", None) or preset["category_columns"]
    if isinstance(prompt_columns, str):
        prompt_columns = (prompt_columns,)
    if isinstance(category_columns, str):
        category_columns = (category_columns,)

    if dataset_config is None:
        dataset = load_dataset(dataset_path)[dataset_split]
    else:
        dataset = load_dataset(dataset_path, dataset_config)[dataset_split]

    category_prompts = defaultdict(list)
    all_prompts_with_category = []
    skipped = 0
    for item in dataset:
        prompt = _first_present_value(item, prompt_columns)
        if prompt is None:
            skipped += 1
            continue
        category = _first_present_value(item, category_columns) or "Uncategorized"
        category_prompts[category].append(prompt)
        all_prompts_with_category.append((prompt, category))

    if not all_prompts_with_category:
        available_columns = ", ".join(dataset.column_names)
        raise ValueError(
            f"No prompts found in dataset '{dataset_path}' split '{dataset_split}'. "
            f"Tried prompt column(s): {', '.join(prompt_columns)}. "
            f"Available columns: {available_columns}."
        )

    print(
        f"Loaded {len(all_prompts_with_category)} prompts from {dataset_path} "
        f"({dataset_split}) across {len(category_prompts)} categories."
    )
    if skipped:
        print(f"Skipped {skipped} dataset row(s) without a non-empty prompt.")
    return category_prompts, all_prompts_with_category


category_prompts, all_prompts_with_category = _load_prompt_dataset(config)

N_PER_CATEGORY = config['prompt_per_categorie']  # Number of prompts to sample per category
SUBSET_SEED = config['prompt_sample_seed']
SINGLE_PROMPT_PER_SEED = config.get("single_prompt_per_seed", False)
USE_ENTIRE_DATASET = config.get("use_entire_dataset", False)
PROMPT_INDEX_RANGE = config.get("prompt_index_range", None)


def _parse_prompt_index_range(prompt_index_range, prompt_count):
    if prompt_index_range is None:
        return None
    if not isinstance(prompt_index_range, (list, tuple)) or len(prompt_index_range) != 2:
        raise ValueError("prompt_index_range must be null or a two-item list: [start, end].")

    start, end = prompt_index_range
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("prompt_index_range start and end must be integers.")
    if start < 0 or end < 0:
        raise ValueError("prompt_index_range start and end must be non-negative.")
    if start >= end:
        raise ValueError("prompt_index_range start must be smaller than end.")
    if end > prompt_count:
        raise ValueError(
            f"prompt_index_range end ({end}) exceeds the available prompt count ({prompt_count})."
        )
    return start, end


def build_selected_prompts(seed):
    if USE_ENTIRE_DATASET:
        prompt_index_range = _parse_prompt_index_range(
            PROMPT_INDEX_RANGE, len(all_prompts_with_category)
        )
        if prompt_index_range is None:
            selected_prompts_with_category = list(all_prompts_with_category)
            start = 0
        else:
            start, end = prompt_index_range
            selected_prompts_with_category = list(all_prompts_with_category[start:end])

        print(
            f"Selected {len(selected_prompts_with_category)} of {len(all_prompts_with_category)} prompts "
            f"from {len(category_prompts)} categories for seed {seed}."
        )
        if prompt_index_range is not None:
            print(
                f"Using prompt_index_range [{start}, {end}) with original prompt numbers "
                f"{start + 1}-{end}."
            )
        return [
            (prompt, category, prompt_index + 1)
            for prompt_index, (prompt, category) in enumerate(
                selected_prompts_with_category, start=start
            )
        ]

    if SINGLE_PROMPT_PER_SEED:
        rng = random.Random(seed)
        selected_prompt = rng.choice(all_prompts_with_category)
        print(
            f"Selected 1 prompt for seed {seed} from {len(all_prompts_with_category)} total prompts "
            f"across {len(category_prompts)} categories."
        )
        prompt, category = selected_prompt
        return [(prompt, category, 1)]

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
    return [
        (prompt, category, prompt_index)
        for prompt_index, (prompt, category) in enumerate(selected_prompts_with_category, start=1)
    ]



def aggregate_results(output_folder=None):
    """Generate the same summaries as the standalone aggregation script."""
    from algorithms.aggregate_results import aggregate_results as aggregate_saved_results

    aggregate_saved_results(output_folder or OUTPUT_FOLDER)


def _fill_deferred_metrics(experiment_dir, metrics):
    """Fill selected non-objective metrics before producing aggregate outputs."""
    from algorithms.fill_zero_weight_metrics import MetricBackfiller, prompt_dirs

    backfiller = MetricBackfiller(config, metrics_to_calculate=metrics)
    # This pass supplies diagnostics only; preserve optimization objectives.
    backfiller.recompute_objective = False
    try:
        for prompt_dir in prompt_dirs(Path(experiment_dir)):
            for csv_name, handler in (
                ("score_results.csv", backfiller.backfill_score_csv),
                ("fitness_results.csv", backfiller.backfill_fitness_csv),
            ):
                csv_path = prompt_dir / csv_name
                if csv_path.exists():
                    completed = handler(prompt_dir, csv_path)
                    completed.to_csv(csv_path, index=False, na_rep="nan")
                    print(f"Wrote deferred metrics: {csv_path}")
    finally:
        backfiller.engine = None
        Eigo.clear_model_cache()


if __name__ == "__main__":
    # Entry point for the script
    # Parses arguments, loads configuration, and starts the optimization process.

    optimization_config = dict(config, evaluate_zero_weight_metrics=False)
    eigo_engine = Eigo(optimization_config)
    deferred_metrics = (
        tuple(metric for metric in eigo_engine.zero_weight_metrics
              if not eigo_engine._should_evaluate_metric(metric))
        if config.get("evaluate_zero_weight_metrics", False) else ()
    )
    experiment_dir = eigo_engine.OUTPUT_FOLDER

    seed_number = 1
    for seed in seed_list:
        selected_prompts_with_category = build_selected_prompts(seed)
        for prompt, category, prompt_number in selected_prompts_with_category:
            print(f"Running seed {seed}, prompt: {prompt} (Category: {category})")
            if config['optimization_method'] == "cmaes":
                eigo_engine.run_cmaes_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "ga":
                eigo_engine.run_ga_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "adam":
                eigo_engine.run_adam_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "random_sampler":
                eigo_engine.run_random_sampler_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "zero_order":
                eigo_engine.run_zero_order_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "snes":
                eigo_engine.run_snes_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "cosyne":
                eigo_engine.run_cosyne_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            elif config['optimization_method'] == "gomea":
                eigo_engine.run_gomea_optimization(seed=seed, seed_number=seed_number, prompt=prompt, category=category, prompt_number=prompt_number)
            else:
                raise ValueError(f"Unknown optimization method: {config['optimization_method']}")
            print(f"Run with seed {seed} and prompt '{prompt}' finished!")
        seed_number += 1

    if deferred_metrics:
        print(f"Evaluating deferred metrics after all runs: {', '.join(deferred_metrics)}")
        del eigo_engine
        Eigo.clear_model_cache()
        _fill_deferred_metrics(experiment_dir, deferred_metrics)

    aggregate_results()
