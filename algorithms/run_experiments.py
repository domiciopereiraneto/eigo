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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root so local project modules can be imported.
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageOps

from eigo import Eigo

parser = argparse.ArgumentParser(description='Run optimization with configuration file')
parser.add_argument('--config', type=str, default="algorithms/config/config_run_experiments.yaml",
                   help='Path to configuration YAML file')
args = parser.parse_args()

config_path = args.config

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

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



def _first_non_empty(series):
    for value in series.dropna():
        text = str(value).strip()
        if text:
            return text
    return None


def _numeric_values(df, column):
    if column not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _result_image_path(prompt_dir, stem):
    for extension in (".jpg", ".jpeg", ".png"):
        path = os.path.join(prompt_dir, f"{stem}{extension}")
        if os.path.exists(path):
            return path
    return os.path.join(prompt_dir, f"{stem}.jpg")


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
                "aesthetic": _numeric_values(df, "aesthetic_score"),
                "clip": _numeric_values(df, "clip_score"),
                "image_reward": _numeric_values(df, "image_reward_score"),
                "hpsv2": _numeric_values(df, "hpsv2_score"),
                "pickscore": _numeric_values(df, "pickscore_score"),
                "jpeg_size": _numeric_values(df, "jpeg_size_kb"),
                "objective": _numeric_values(df, "combined_loss"),
                "vram": _numeric_values(df, "peak_vram_mb"),
                "population_metrics": None,
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
                "aesthetic": _numeric_values(df, "max_aesthetic_score"),
                "clip": _numeric_values(df, "max_clip_score"),
                "image_reward": _numeric_values(df, "max_image_reward_score"),
                "hpsv2": _numeric_values(df, "max_hpsv2_score"),
                "pickscore": _numeric_values(df, "max_pickscore_score"),
                "jpeg_size": _numeric_values(df, "min_jpeg_size_kb"),
                "objective": _numeric_values(df, "max_fitness"),
                "vram": _numeric_values(df, "peak_vram_mb"),
                "population_metrics": {
                    "aesthetic": _numeric_values(df, "avg_aesthetic_score"),
                    "clip": _numeric_values(df, "avg_clip_score"),
                    "image_reward": _numeric_values(df, "avg_image_reward_score"),
                    "hpsv2": _numeric_values(df, "avg_hpsv2_score"),
                    "pickscore": _numeric_values(df, "avg_pickscore_score"),
                    "jpeg_size": _numeric_values(df, "avg_jpeg_size_kb"),
                    "objective": _numeric_values(df, "avg_fitness"),
                },
                "objective_name": "fitness",
            })

    cleaned_runs = []
    for run in runs:
        valid = np.isfinite(run["x"]) & np.isfinite(run["time"])
        valid = valid & np.isfinite(run["aesthetic"]) & np.isfinite(run["clip"]) & np.isfinite(run["objective"])
        if valid.sum() == 0:
            continue
        metric_values = {
            metric: run[metric][valid]
            for metric in (
                "aesthetic",
                "clip",
                "image_reward",
                "hpsv2",
                "pickscore",
                "jpeg_size",
                "objective",
                "vram",
            )
        }
        population_metrics = run["population_metrics"]
        if population_metrics is not None:
            population_metrics = {
                metric: values[valid]
                for metric, values in population_metrics.items()
            }
        cleaned_runs.append({
            **run,
            "x": run["x"][valid],
            "time": run["time"][valid],
            **metric_values,
            "population_metrics": population_metrics,
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

    base_path = _result_image_path(prompt_dir, "it_0")
    best_path = _result_image_path(prompt_dir, "best_all")
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


def _grid_font(size):
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _grid_wrap_text(draw, text, font, max_width):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _grid_line_height(font):
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent + 3
    bounds = font.getbbox("Ag")
    return bounds[3] - bounds[1] + 3


def _grid_draw_centered(draw, lines, font, x0, x1, y, fill=(25, 25, 25)):
    line_height = _grid_line_height(font)
    for index, line in enumerate(lines):
        width = draw.textlength(line, font=font)
        draw.text((x0 + (x1 - x0 - width) / 2, y + index * line_height), line, font=font, fill=fill)
    return len(lines) * line_height


def _grid_metric_lines(run, index, objective_name, max_width, draw, font):
    metric_specs = [
        (objective_name.title(), "objective"),
        ("Aes", "aesthetic"),
        ("CLIP", "clip"),
        ("IR", "image_reward"),
        ("HPS", "hpsv2"),
        ("Pick", "pickscore"),
        ("JPEG KB", "jpeg_size"),
    ]
    parts = []
    for label, key in metric_specs:
        value = float(run[key][index])
        if np.isfinite(value):
            parts.append(f"{label}: {value:.4g}")

    if index == -1:
        elapsed = float(run["time"][-1])
        if np.isfinite(elapsed):
            parts.append(f"Time: {elapsed:.2f}s")
        finite_vram = run["vram"][np.isfinite(run["vram"])]
        if finite_vram.size:
            parts.append(f"Peak VRAM: {np.max(finite_vram):.1f} MB")

    lines = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}  {part}"
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = part
    if current:
        lines.append(current)
    return lines or [""]


def _grid_tile(image_path, width, height):
    tile = Image.new("RGB", (width, height), (242, 242, 242))
    with Image.open(image_path).convert("RGB") as image:
        image = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        tile.paste(image, (x, y))
    return tile


def _save_aggregate_image_grids(runs, objective_name, experiment_dir, rows_per_page=20):
    available_runs = []
    for run in runs:
        initial_path = _result_image_path(run["prompt_dir"], "it_0")
        best_path = _result_image_path(run["prompt_dir"], "best_all")
        if os.path.exists(initial_path) and os.path.exists(best_path):
            available_runs.append((run, initial_path, best_path))
    if not available_runs:
        print(f"Warning: no initial/best image pairs found for grid in {experiment_dir}")
        return []

    tile_w = 320
    tile_h = 320
    margin = 20
    gap = 16
    row_gap = 20
    grid_w = tile_w * 2 + gap
    canvas_w = grid_w + margin * 2
    prompt_font = _grid_font(16)
    metric_font = _grid_font(13)
    prompt_line_height = _grid_line_height(prompt_font)
    metric_line_height = _grid_line_height(metric_font)
    saved_paths = []
    page_count = int(np.ceil(len(available_runs) / rows_per_page))

    for page_index in range(page_count):
        page_runs = available_runs[page_index * rows_per_page:(page_index + 1) * rows_per_page]
        row_images = []
        for run, initial_path, best_path in page_runs:
            scratch = Image.new("RGB", (canvas_w, 1), "white")
            scratch_draw = ImageDraw.Draw(scratch)
            heading = run["prompt"]
            details = [value for value in (run["category"], os.path.basename(run["prompt_dir"])) if value]
            if details:
                heading = f"{heading} ({' | '.join(details)})"
            prompt_lines = _grid_wrap_text(scratch_draw, heading, prompt_font, grid_w)
            initial_lines = ["Initial"] + _grid_metric_lines(
                run, 0, objective_name, tile_w, scratch_draw, metric_font
            )
            best_lines = ["Best"] + _grid_metric_lines(
                run, -1, objective_name, tile_w, scratch_draw, metric_font
            )
            title_line_count = max(len(initial_lines), len(best_lines))
            prompt_h = len(prompt_lines) * prompt_line_height
            titles_h = title_line_count * metric_line_height
            row_h = prompt_h + 10 + titles_h + 8 + tile_h
            row_image = Image.new("RGB", (canvas_w, row_h), "white")
            draw = ImageDraw.Draw(row_image)
            _grid_draw_centered(draw, prompt_lines, prompt_font, margin, margin + grid_w, 0)
            titles_y = prompt_h + 10
            image_y = titles_y + titles_h + 8
            _grid_draw_centered(draw, initial_lines, metric_font, margin, margin + tile_w, titles_y)
            best_x = margin + tile_w + gap
            _grid_draw_centered(draw, best_lines, metric_font, best_x, best_x + tile_w, titles_y)
            row_image.paste(_grid_tile(initial_path, tile_w, tile_h), (margin, image_y))
            row_image.paste(_grid_tile(best_path, tile_w, tile_h), (best_x, image_y))
            row_images.append(row_image)

        canvas_h = margin * 2 + sum(row.height for row in row_images) + row_gap * (len(row_images) - 1)
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        y = margin
        for row_image in row_images:
            canvas.paste(row_image, (0, y))
            y += row_image.height + row_gap
        suffix = "" if page_count == 1 else f"_{page_index + 1:02d}"
        output_path = os.path.join(experiment_dir, f"aggregate_image_grid{suffix}.jpg")
        canvas.save(output_path, format="JPEG", quality=95)
        saved_paths.append(output_path)
        print(f"Saved aggregate image grid: {output_path}")
    return saved_paths


def _summary_rows(df, group_name):
    row = {"prompt": group_name, "n_runs": len(df)}
    for metric in [
        "aesthetic",
        "clip",
        "image_reward",
        "hpsv2",
        "pickscore",
        "jpeg_size",
        "objective",
        "peak_vram_mb",
        "cosine_similarity",
        "ssim",
    ]:
        values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            row[f"{metric}_avg"] = np.nan
            row[f"{metric}_std"] = np.nan
            row[f"{metric}_min"] = np.nan
            row[f"{metric}_max"] = np.nan
            continue
        row[f"{metric}_avg"] = float(np.mean(finite_values))
        row[f"{metric}_std"] = float(np.std(finite_values))
        row[f"{metric}_min"] = float(np.min(finite_values))
        row[f"{metric}_max"] = float(np.max(finite_values))
    return row


def _finite_numeric(values):
    values = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    return values[np.isfinite(values)]


def _stats_summary_row(metric, solution_type, values):
    values = _finite_numeric(values)
    return {
        "metric": metric,
        "solution_type": solution_type,
        "count": int(values.size),
        "min": float(np.min(values)) if values.size else np.nan,
        "mean": float(np.mean(values)) if values.size else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "max": float(np.max(values)) if values.size else np.nan,
        "std": float(np.std(values)) if values.size else np.nan,
    }


def _all_sample_metric_values(run, metric):
    population_metrics = run["population_metrics"]
    if population_metrics is not None:
        return population_metrics[metric]
    return run[metric]


def _final_metrics_workbook_frames(runs, objective_name):
    metric_labels = {
        "aesthetic": "aesthetic",
        "clip": "clip",
        "image_reward": "image_reward",
        "hpsv2": "hpsv2",
        "pickscore": "pickscore",
        "jpeg_size": "jpeg_size_kb",
        "objective": objective_name,
    }
    value_rows = []
    all_sample_rows = []
    for run in runs:
        population_metrics = run["population_metrics"]
        for metric, label in metric_labels.items():
            population_value = np.nan
            if population_metrics is not None:
                population_value = float(population_metrics[metric][-1])
            value_rows.append({
                "prompt": run["prompt"],
                "category": run["category"],
                "result_file": run["path"],
                "elapsed_time_seconds": float(run["time"][-1]),
                "peak_vram_mb": float(np.nanmax(run["vram"])) if np.isfinite(run["vram"]).any() else np.nan,
                "metric": label,
                "population": population_value,
                "best_solution": float(run[metric][-1]),
            })
            sample_values = _all_sample_metric_values(run, metric)
            for x_value, elapsed_time, sample_value in zip(run["x"], run["time"], sample_values):
                all_sample_rows.append({
                    "prompt": run["prompt"],
                    "category": run["category"],
                    "result_file": run["path"],
                    "x_label": run["x_label"],
                    "x": float(x_value),
                    "elapsed_time_seconds": float(elapsed_time),
                    "metric": label,
                    "value": float(sample_value),
                    "source": "generation_average" if population_metrics is not None else "sample",
                })

    values_df = pd.DataFrame(value_rows)
    all_samples_df = pd.DataFrame(all_sample_rows)
    summary_rows = []
    for metric in metric_labels.values():
        metric_rows = values_df[values_df["metric"] == metric]
        for solution_type in ("population", "best_solution"):
            summary_rows.append(_stats_summary_row(metric, solution_type, metric_rows[solution_type]))
        all_sample_values = all_samples_df.loc[all_samples_df["metric"] == metric, "value"]
        summary_rows.append(_stats_summary_row(metric, "all_samples", all_sample_values))
    elapsed_values = values_df.drop_duplicates(subset=["result_file"])["elapsed_time_seconds"]
    summary_rows.append(_stats_summary_row("elapsed_time_seconds", "run", elapsed_values))
    vram_values = values_df.drop_duplicates(subset=["result_file"])["peak_vram_mb"]
    summary_rows.append(_stats_summary_row("peak_vram_mb", "run", vram_values))
    return pd.DataFrame(summary_rows), values_df, all_samples_df


def _write_final_metrics_workbook(runs, objective_name, output_path):
    summary_df, values_df, all_samples_df = _final_metrics_workbook_frames(runs, objective_name)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        values_df.to_excel(writer, sheet_name="per_prompt_run", index=False)
        all_samples_df.to_excel(writer, sheet_name="all_samples", index=False)


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

    metric_labels = [
        ("aesthetic", "aesthetic"),
        ("clip", "clip"),
        ("image_reward", "image_reward"),
        ("hpsv2", "hpsv2"),
        ("pickscore", "pickscore"),
        ("jpeg_size", "jpeg_size_kb"),
        ("objective", objective_name),
        ("vram", "peak_vram_mb"),
    ]
    for metric, label in metric_labels:
        if not any(np.isfinite(run[metric]).any() for run in runs):
            print(f"Skipping {label} aggregation because no values were found.")
            continue
        step_values = _stack_on_axis(runs, metric, "x", x_axis)
        step_stats = _build_stats(x_axis, step_values, axis_name)
        step_stats.to_csv(os.path.join(experiment_dir, f"{label}_evolution_by_{axis_name}.csv"), index=False)
        _plot_evolution(
            step_stats,
            axis_name,
            label,
            f"{label.title()} evolution by {axis_name}",
            os.path.join(experiment_dir, f"{label}_evolution_by_{axis_name}.jpg"),
        )

        time_values = _stack_on_axis(runs, metric, "time", time_axis)
        time_stats = _build_stats(time_axis, time_values, "elapsed_time_seconds")
        time_stats.to_csv(os.path.join(experiment_dir, f"{label}_evolution_by_time.csv"), index=False)
        _plot_evolution(
            time_stats,
            "elapsed_time_seconds",
            label,
            f"{label.title()} evolution by elapsed time",
            os.path.join(experiment_dir, f"{label}_evolution_by_time.jpg"),
        )

    cosine_similarity, ssim_vals = _compute_similarity_metrics(runs)
    final_rows = []
    for run, cos_sim, ssim_value in zip(runs, cosine_similarity, ssim_vals):
        final_rows.append({
            "prompt": run["prompt"],
            "category": run["category"],
            "aesthetic": float(run["aesthetic"][-1]),
            "clip": float(run["clip"][-1]),
            "image_reward": float(run["image_reward"][-1]),
            "hpsv2": float(run["hpsv2"][-1]),
            "pickscore": float(run["pickscore"][-1]),
            "jpeg_size": float(run["jpeg_size"][-1]),
            "objective": float(run["objective"][-1]),
            "peak_vram_mb": float(np.nanmax(run["vram"])) if np.isfinite(run["vram"]).any() else np.nan,
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
    _write_final_metrics_workbook(
        runs,
        objective_name,
        os.path.join(experiment_dir, "aggregate_final_metrics.xlsx"),
    )
    _save_aggregate_image_grids(runs, objective_name, experiment_dir)
    _plot_similarity_boxplot(finals_df, os.path.join(experiment_dir, "similarity_boxplot.jpg"))
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

    aggregate_results()
