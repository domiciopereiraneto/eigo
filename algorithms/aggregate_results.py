#!/usr/bin/env python3
"""Regenerate run_experiments summaries from saved prompt-level results.

Run with --config algorithms/config/config_aggregate_results.yaml.
This module is also shared by run_experiments.py.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config/config_aggregate_results.yaml"
RUN_METRICS = (
    "aesthetic",
    "clip",
    "image_reward",
    "hpsv2",
    "pickscore",
    "jpeg_size",
    "ensemble",
    "objective",
    "vram",
)


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
                "ensemble": _numeric_values(df, "ensemble_score"),
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
                "ensemble": _numeric_values(df, "max_ensemble_score"),
                "objective": _numeric_values(df, "max_fitness"),
                "vram": _numeric_values(df, "peak_vram_mb"),
                "population_metrics": {
                    "aesthetic": _numeric_values(df, "avg_aesthetic_score"),
                    "clip": _numeric_values(df, "avg_clip_score"),
                    "image_reward": _numeric_values(df, "avg_image_reward_score"),
                    "hpsv2": _numeric_values(df, "avg_hpsv2_score"),
                    "pickscore": _numeric_values(df, "avg_pickscore_score"),
                    "jpeg_size": _numeric_values(df, "avg_jpeg_size_kb"),
                    "ensemble": _numeric_values(df, "avg_ensemble_score"),
                    "objective": _numeric_values(df, "avg_fitness"),
                },
                "objective_name": "fitness",
            })

    cleaned_runs = []
    for run in runs:
        valid = np.isfinite(run["x"]) & np.isfinite(run["time"])
        valid = valid & np.isfinite(run["aesthetic"]) & np.isfinite(run["clip"]) & np.isfinite(run["objective"])
        # The initial image (it_0 / step 0) is kept only for display; it never
        # counts towards statistics or best-individual selection.
        initial_rows = np.flatnonzero(valid & (run["x"] == 0))
        initial = None
        if initial_rows.size:
            initial = {metric: float(run[metric][initial_rows[0]]) for metric in RUN_METRICS}
        valid = valid & (run["x"] > 0)
        if valid.sum() == 0:
            continue
        metric_values = {metric: run[metric][valid] for metric in RUN_METRICS}
        population_metrics = run["population_metrics"]
        if population_metrics is not None:
            population_metrics = {
                metric: values[valid]
                for metric, values in population_metrics.items()
            }
        cleaned = {
            **run,
            "x": run["x"][valid],
            "time": run["time"][valid],
            **metric_values,
            "population_metrics": population_metrics,
            "initial": initial,
        }
        cleaned["best_so_far_idx"] = _best_so_far_indices(
            cleaned["objective"], minimize=cleaned["objective_name"] == "loss"
        )
        cleaned["best_idx"] = int(cleaned["best_so_far_idx"][-1])
        cleaned_runs.append(cleaned)
    return cleaned_runs


def _best_so_far_indices(objective, minimize):
    """Index of the best individual (by objective) seen up to each step."""
    indices = np.empty(len(objective), dtype=int)
    best = 0
    for i, value in enumerate(objective):
        if (value < objective[best]) if minimize else (value > objective[best]):
            best = i
        indices[i] = best
    return indices


def _best_value(run, metric):
    """Metric value of the run's best individual (selected by objective)."""
    return float(run[metric][run["best_idx"]])


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


def _plot_evolution(stats_df, x_col, y_name, title, output_path, population_stats=None):
    plt.figure(figsize=(10, 6))
    series = [("avg best", stats_df)]
    if population_stats is not None:
        series.insert(0, ("avg population", population_stats))
    for label, stats in series:
        x = stats[x_col].to_numpy(dtype=float)
        mean = stats["avg"].to_numpy(dtype=float)
        std = stats["std"].to_numpy(dtype=float)
        line, = plt.plot(x, mean, label=label)
        plt.fill_between(
            x, mean - std, mean + std,
            color=line.get_color(), alpha=0.25, label=f"{label} ± std",
        )
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


def _grid_metric_lines(run, which, objective_name, max_width, draw, font):
    """`which` is "initial" (it_0 values) or "best" (best individual)."""
    metric_specs = [
        (objective_name.title(), "objective"),
        ("Aes", "aesthetic"),
        ("CLIP", "clip"),
        ("IR", "image_reward"),
        ("HPS", "hpsv2"),
        ("Pick", "pickscore"),
        ("JPEG KB", "jpeg_size"),
        ("Ensemble", "ensemble"),
    ]
    parts = []
    for label, key in metric_specs:
        if which == "initial":
            value = (run["initial"] or {}).get(key, np.nan)
        else:
            value = _best_value(run, key)
        if np.isfinite(value):
            parts.append(f"{label}: {value:.4g}")

    if which == "best":
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
                run, "initial", objective_name, tile_w, scratch_draw, metric_font
            )
            best_lines = ["Best"] + _grid_metric_lines(
                run, "best", objective_name, tile_w, scratch_draw, metric_font
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
        "ensemble",
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
        "ensemble": "ensemble_score",
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
                "best_solution": _best_value(run, metric),
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
    # Step 0 (the initial image) is excluded from the runs, so axes start after it.
    x_axis = np.arange(
        int(min(np.min(run["x"]) for run in runs)),
        int(max(np.max(run["x"]) for run in runs)) + 1,
    )
    min_time = min(np.min(run["time"]) for run in runs)
    max_time = max(np.max(run["time"]) for run in runs)
    time_axis = np.linspace(min_time, max_time, 200) if max_time > min_time else np.array([max_time])

    metric_labels = [
        ("aesthetic", "aesthetic"),
        ("clip", "clip"),
        ("image_reward", "image_reward"),
        ("hpsv2", "hpsv2"),
        ("pickscore", "pickscore"),
        ("jpeg_size", "jpeg_size_kb"),
        ("ensemble", "ensemble_score"),
        ("objective", objective_name),
        ("vram", "peak_vram_mb"),
    ]
    for metric, label in metric_labels:
        if not any(np.isfinite(run[metric]).any() for run in runs):
            print(f"Skipping {label} aggregation because no values were found.")
            continue
        # The "best" series follows the best individual so far (selected by
        # objective), so every metric comes from the same individual.
        plot_runs = []
        for run in runs:
            population = run["population_metrics"]
            population_values = (
                population[metric] if population is not None and metric in population
                else run[metric]
            )
            best = run[metric] if metric == "vram" else run[metric][run["best_so_far_idx"]]
            plot_runs.append({
                **run,
                "best": np.asarray(best, dtype=float),
                "population": population_values,
            })

        for axis_key, axis, column, suffix, title_axis in (
            ("x", x_axis, axis_name, axis_name, axis_name),
            ("time", time_axis, "elapsed_time_seconds", "time", "elapsed time"),
        ):
            best_stats = _build_stats(
                axis, _stack_on_axis(plot_runs, "best", axis_key, axis), column,
            )
            population_stats = _build_stats(
                axis, _stack_on_axis(plot_runs, "population", axis_key, axis), column,
            )
            # Preserve existing best-series CSV columns and add population stats.
            output_stats = best_stats.copy()
            for stat in ("count", "avg", "std", "min", "max"):
                output_stats[f"population_{stat}"] = population_stats[stat]
            stem = os.path.join(experiment_dir, f"{label}_evolution_by_{suffix}")
            output_stats.to_csv(f"{stem}.csv", index=False)
            _plot_evolution(
                best_stats,
                column,
                label,
                f"{label.title()} best-of-run evolution by {title_axis}",
                f"{stem}.jpg",
                population_stats=population_stats,
            )

    cosine_similarity, ssim_vals = _compute_similarity_metrics(runs)
    final_rows = []
    for run, cos_sim, ssim_value in zip(runs, cosine_similarity, ssim_vals):
        final_rows.append({
            "prompt": run["prompt"],
            "category": run["category"],
            **{
                metric: _best_value(run, metric)
                for metric in RUN_METRICS if metric != "vram"
            },
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


def aggregate_results(output_folder):
    experiment_dirs = _find_experiment_dirs(output_folder)
    if not experiment_dirs:
        print(f"No experiment folders found under: {output_folder}")
        return
    for experiment_dir in experiment_dirs:
        _aggregate_one_experiment(experiment_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML configuration containing experiment_folders")
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        parser.error("Configuration must be a YAML mapping.")
    folders = config.get("experiment_folders")
    if not isinstance(folders, list) or not folders:
        parser.error("experiment_folders must be a non-empty list of folder paths.")

    # Discover everything before writing, and process overlapping selections once.
    experiments = set()
    for folder in folders:
        if not isinstance(folder, str) or not folder.strip():
            parser.error("Each experiment_folders entry must be a non-empty path string.")
        path = Path(folder).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_dir():
            parser.error(f"Experiment folder does not exist: {path}")
        discovered = _find_experiment_dirs(path)
        if not discovered:
            parser.error(f"No prompt-level result CSVs found under: {path}")
        experiments.update(discovered)

    for experiment in sorted(experiments):
        _aggregate_one_experiment(experiment)


if __name__ == "__main__":
    main()
