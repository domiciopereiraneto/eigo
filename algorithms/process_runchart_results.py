#!/usr/bin/env python3
"""Create runchart plots for EIGO experiment metrics."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = Path(__file__).resolve().parent

MPLCONFIG_DIR = Path("/tmp") / "eigo_matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


for path in (REPO_ROOT, ALGORITHMS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from process_quantitative_results import (  # noqa: E402
    CANONICAL_METRIC_COLUMNS,
    discover_experiment_dirs,
    experiment_metadata,
    load_prompt_config,
    load_yaml,
    prompt_dirs,
    read_result_csv,
    clean_result_df,
    resolve_path,
)


LOWER_IS_BETTER = {"jpeg_size_kb", "combined_loss"}
DEFAULT_METRICS = (
    "objective",
    "aesthetic_score",
    "clip_score",
    "image_reward_score",
    "hpsv2_score",
    "pickscore_score",
    "jpeg_size_kb",
)
DEFAULT_METRIC_LABELS = {
    "objective": "Objective",
    "aesthetic_score": "Aesthetic score",
    "clip_score": "CLIP score",
    "image_reward_score": "ImageReward score",
    "hpsv2_score": "HPSv2 score",
    "pickscore_score": "PickScore",
    "jpeg_size_kb": "JPEG size (KB)",
    "elapsed_time": "Elapsed time (s)",
    "peak_vram_mb": "Peak VRAM (MB)",
}

MUTED_COLORS = (
    "#6f8fd8",
    "#86a96f",
    "#e6a657",
    "#9b8cc9",
    "#c77b7b",
    "#6aa7a8",
    "#b8a35e",
    "#8f8f8f",
)


@dataclass
class PromptCurve:
    experiment_label: str
    results_folder_name: str
    csv_kind: str
    prompt_folder: str
    step: np.ndarray
    nfe: np.ndarray
    elapsed_time: np.ndarray
    value: np.ndarray


@dataclass
class ExperimentCurves:
    label: str
    metadata: dict
    csv_kinds: set
    curves: List[PromptCurve]


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_") or "metric"


def metric_label(metric: str, config: dict) -> str:
    labels = config.get("metric_labels", {}) or {}
    return str(labels.get(metric, DEFAULT_METRIC_LABELS.get(metric, metric)))


def metric_column(metric: str, csv_kind: str, df: pd.DataFrame) -> Optional[str]:
    aliases = CANONICAL_METRIC_COLUMNS.get(csv_kind, {})
    column = aliases.get(metric, metric)
    if column in df.columns:
        return column
    return None


def apply_best_so_far(metric: str, values: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return values
    out = values.astype(float, copy=True)
    finite = np.isfinite(out)
    if not finite.any():
        return out

    if metric in LOWER_IS_BETTER:
        best = math.inf
        for idx, value in enumerate(out):
            if np.isfinite(value):
                best = min(best, value)
            out[idx] = best if np.isfinite(best) else np.nan
    else:
        best = -math.inf
        for idx, value in enumerate(out):
            if np.isfinite(value):
                best = max(best, value)
            out[idx] = best if np.isfinite(best) else np.nan
    return out


def population_size(run_config: dict, metadata: dict) -> int:
    method = str(metadata.get("algorithm_key", "")).lower()
    keys_by_method = {
        "snes": ("snes_pop_size", "pop_size"),
        "cosyne": ("cosyne_pop_size", "pop_size"),
        "zeroorder": ("zero_order_pop_size", "pop_size"),
        "zero_order": ("zero_order_pop_size", "pop_size"),
        "ga": ("ga_pop_size", "pop_size"),
        "cmaes": ("pop_size",),
        "sepcmaes": ("pop_size",),
        "vdcmae": ("pop_size",),
        "randomsampler": ("num_images_to_generate", "pop_size"),
        "random_sampler": ("num_images_to_generate", "pop_size"),
    }
    for key in keys_by_method.get(method, ("pop_size",)):
        value = run_config.get(key)
        if value is None:
            continue
        try:
            pop_size = int(value)
        except (TypeError, ValueError):
            continue
        if pop_size > 0:
            return pop_size
    return 1


def denoising_steps(run_config: dict) -> int:
    for key in ("num_inference_steps", "inference_steps"):
        value = run_config.get(key)
        if value is None:
            continue
        try:
            steps = int(value)
        except (TypeError, ValueError):
            continue
        if steps > 0:
            return steps
    return 1


def nfe_from_step(step: np.ndarray, csv_kind: str, pop_size: int, diffusion_steps: int) -> np.ndarray:
    step = np.asarray(step, dtype=float)
    if csv_kind == "score":
        image_evaluations = step + 1.0
    else:
        image_evaluations = 1.0 + step * float(pop_size)
    return image_evaluations * float(diffusion_steps)


def clean_curve_arrays(
    df: pd.DataFrame,
    step_column: str,
    value_column: str,
    metric: str,
    best_so_far: bool,
    csv_kind: str,
    pop_size: int,
    diffusion_steps: int,
):
    step = pd.to_numeric(df[step_column], errors="coerce") if step_column in df.columns else pd.Series(np.arange(len(df)))
    elapsed = pd.to_numeric(df.get("elapsed_time", pd.Series([np.nan] * len(df))), errors="coerce")
    value = pd.to_numeric(df[value_column], errors="coerce")
    valid = step.notna() & value.notna()
    clean = pd.DataFrame(
        {
            "step": step[valid].to_numpy(dtype=float),
            "elapsed_time": elapsed[valid].to_numpy(dtype=float),
            "value": value[valid].to_numpy(dtype=float),
        }
    ).sort_values("step")
    clean = clean.drop_duplicates(subset=["step"], keep="last")
    if clean.empty:
        return None
    return (
        clean["step"].to_numpy(dtype=float),
        nfe_from_step(clean["step"].to_numpy(dtype=float), csv_kind, pop_size, diffusion_steps),
        clean["elapsed_time"].to_numpy(dtype=float),
        apply_best_so_far(metric, clean["value"].to_numpy(dtype=float), best_so_far),
    )


def format_experiment_label(metadata: dict, template: str) -> str:
    try:
        label = template.format(**metadata)
    except KeyError:
        label = "{algorithm} - {target_variable}".format(**metadata)
    return re.sub(r"\s+", " ", label).strip(" -")


def unique_labels(experiment_labels: List[Tuple[Path, str]]) -> Dict[Path, str]:
    counts: Dict[str, int] = {}
    for _, label in experiment_labels:
        counts[label] = counts.get(label, 0) + 1
    result = {}
    for path, label in experiment_labels:
        result[path] = f"{label} ({path.name})" if counts[label] > 1 else label
    return result


def load_experiment_curves(
    experiment_dirs: Sequence[Path],
    metric: str,
    label_template: str,
    best_so_far: bool,
) -> List[ExperimentCurves]:
    metadata_by_dir = {}
    label_pairs = []
    for experiment_dir in experiment_dirs:
        first_config = {}
        for prompt_dir in prompt_dirs(experiment_dir):
            first_config = load_prompt_config(prompt_dir)
            if first_config:
                break
        metadata = experiment_metadata(experiment_dir, first_config)
        metadata_by_dir[experiment_dir] = metadata
        label_pairs.append((experiment_dir, format_experiment_label(metadata, label_template)))

    labels_by_dir = unique_labels(label_pairs)
    experiments = []
    for experiment_dir in experiment_dirs:
        metadata = metadata_by_dir[experiment_dir]
        label = labels_by_dir[experiment_dir]
        curves = []
        csv_kinds = set()
        for prompt_dir in prompt_dirs(experiment_dir):
            loaded = read_result_csv(prompt_dir)
            if loaded is None:
                continue
            raw_df, _, csv_kind, step_column = loaded
            df = clean_result_df(raw_df, step_column)
            if df.empty:
                continue
            column = metric_column(metric, csv_kind, df)
            if column is None:
                continue
            prompt_config = load_prompt_config(prompt_dir)
            pop_size = population_size(prompt_config, metadata)
            diffusion_steps = denoising_steps(prompt_config)
            arrays = clean_curve_arrays(
                df,
                step_column,
                column,
                metric,
                best_so_far,
                csv_kind,
                pop_size,
                diffusion_steps,
            )
            if arrays is None:
                continue
            step, nfe, elapsed_time, value = arrays
            csv_kinds.add(csv_kind)
            curves.append(
                PromptCurve(
                    experiment_label=label,
                    results_folder_name=metadata["results_folder_name"],
                    csv_kind=csv_kind,
                    prompt_folder=prompt_dir.name,
                    step=step,
                    nfe=nfe,
                    elapsed_time=elapsed_time,
                    value=value,
                )
            )
        if curves:
            experiments.append(ExperimentCurves(label=label, metadata=metadata, csv_kinds=csv_kinds, curves=curves))
    return experiments


def final_elapsed_time(curve: PromptCurve) -> float:
    valid = curve.elapsed_time[np.isfinite(curve.elapsed_time) & np.isfinite(curve.value)]
    return float(np.nanmax(valid)) if valid.size else np.nan


def resolve_time_limit(experiments: Sequence[ExperimentCurves], configured) -> Optional[float]:
    if configured is None or str(configured).lower() == "null":
        return None
    if isinstance(configured, str) and configured.lower() == "shortest":
        experiment_limits = []
        for experiment in experiments:
            curve_limits = [final_elapsed_time(curve) for curve in experiment.curves]
            curve_limits = [limit for limit in curve_limits if np.isfinite(limit)]
            if curve_limits:
                experiment_limits.append(min(curve_limits))
        if not experiment_limits:
            return None
        return float(min(experiment_limits))
    limit = float(configured)
    if limit < 0:
        raise ValueError("time_limit must be >= 0, 'shortest', or null.")
    return limit


def native_nfe_stats(experiment: ExperimentCurves) -> pd.DataFrame:
    rows = []
    for curve in experiment.curves:
        valid = np.isfinite(curve.nfe) & np.isfinite(curve.value)
        rows.extend(
            {"nfe": float(nfe), "value": float(value)}
            for nfe, value in zip(curve.nfe[valid], curve.value[valid])
        )
    if not rows:
        return pd.DataFrame(columns=["nfe", "mean", "std", "count"])
    values = pd.DataFrame(rows)
    return (
        values.groupby("nfe", as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", lambda series: series.std(ddof=0)),
            count=("value", "count"),
        )
        .sort_values("nfe")
        .reset_index(drop=True)
    )


def native_time_stats(experiment: ExperimentCurves, time_limit: Optional[float]) -> pd.DataFrame:
    rows = []
    for curve in experiment.curves:
        valid = np.isfinite(curve.nfe) & np.isfinite(curve.elapsed_time) & np.isfinite(curve.value)
        if time_limit is not None:
            valid &= curve.elapsed_time <= time_limit
        rows.extend(
            {"nfe": float(nfe), "elapsed_time": float(elapsed), "value": float(value)}
            for nfe, elapsed, value in zip(curve.nfe[valid], curve.elapsed_time[valid], curve.value[valid])
        )
    if not rows:
        return pd.DataFrame(columns=["elapsed_time_seconds", "mean", "std", "count", "nfe"])
    values = pd.DataFrame(rows)
    return (
        values.groupby("nfe", as_index=False)
        .agg(
            elapsed_time_seconds=("elapsed_time", "mean"),
            mean=("value", "mean"),
            std=("value", lambda series: series.std(ddof=0)),
            count=("value", "count"),
        )
        .sort_values("elapsed_time_seconds")
        .reset_index(drop=True)
    )


def write_stats_csv(rows: List[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, na_rep="nan")


def plot_stats(
    series: Dict[str, pd.DataFrame],
    x_column: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
    width: float,
    height: float,
    xscale: str = "linear",
    base10_ticks: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height))
    for color, (label, stats_df) in zip(MUTED_COLORS, series.items()):
        x = stats_df[x_column].to_numpy(dtype=float)
        mean = stats_df["mean"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(mean)
        if not valid.any():
            continue
        xv = x[valid]
        mv = mean[valid]
        marker_every = max(1, len(xv) // 12)
        ax.plot(
            xv,
            mv,
            marker="o",
            color=color,
            markersize=5,
            markevery=marker_every,
            linewidth=2.8,
            label=label,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale(xscale)
    if base10_ticks and xscale == "linear":
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, color="#d0d0d0", linewidth=0.7)
    if xscale == "log":
        ax.grid(True, which="minor", color="#e6e6e6", linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc")
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_runcharts(config: dict) -> List[Path]:
    folder_values = config.get("experiment_folders", config.get("source_dirs", []))
    if not folder_values:
        raise ValueError("Config must define experiment_folders or source_dirs.")
    metrics = config.get("metrics", DEFAULT_METRICS)
    if isinstance(metrics, str):
        metrics = [metrics]
    if not metrics:
        raise ValueError("metrics must contain at least one metric.")

    experiment_dirs = discover_experiment_dirs([resolve_path(path) for path in folder_values])
    if not experiment_dirs:
        raise FileNotFoundError("No experiment folders with results_* CSVs were found.")

    out_dir = resolve_path(config.get("save_folder", "results_runcharts")) / "runcharts"
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.get("clean_output", True)):
        for pattern in (
            "*_by_progress.csv",
            "*_by_progress.png",
            "*_by_nfe.csv",
            "*_by_nfe.png",
            "*_by_time.csv",
            "*_by_time.png",
        ):
            for path in out_dir.glob(pattern):
                path.unlink()
    label_template = str(config.get("label_template", "{algorithm} - {target_variable}"))
    best_so_far = bool(config.get("best_so_far", True))
    width = float(config.get("figure_width", 10.5))
    height = float(config.get("figure_height", 6.2))
    nfe_xscale = str(config.get("nfe_xscale", "log")).lower()
    if nfe_xscale not in {"linear", "log"}:
        raise ValueError("nfe_xscale must be 'linear' or 'log'.")

    written = []
    for metric in metrics:
        metric = str(metric)
        experiments = load_experiment_curves(experiment_dirs, metric, label_template, best_so_far)
        if not experiments:
            print(f"Warning: no curves found for metric '{metric}'.")
            continue

        nfe_series = {}
        nfe_rows = []
        for experiment in experiments:
            stats_df = native_nfe_stats(experiment)
            nfe_series[experiment.label] = stats_df
            for row in stats_df.itertuples(index=False):
                nfe_rows.append(
                    {
                        "metric": metric,
                        "experiment": experiment.label,
                        "results_folder_name": experiment.metadata["results_folder_name"],
                        "nfe": row.nfe,
                        "mean": row.mean,
                        "std": row.std,
                        "count": row.count,
                    }
                )

        safe_metric = safe_name(metric)
        nfe_csv = out_dir / f"{safe_metric}_by_nfe.csv"
        nfe_png = out_dir / f"{safe_metric}_by_nfe.png"
        write_stats_csv(nfe_rows, nfe_csv)
        plot_stats(
            nfe_series,
            "nfe",
            "Inference Compute (NFE)",
            metric_label(metric, config),
            f"{metric_label(metric, config)} by NFE",
            nfe_png,
            width,
            height,
            nfe_xscale,
            True,
        )
        written.extend([nfe_csv, nfe_png])

        resolved_limit = resolve_time_limit(experiments, config.get("time_limit", None))
        time_series = {}
        time_rows = []
        for experiment in experiments:
            stats_df = native_time_stats(experiment, resolved_limit)
            time_series[experiment.label] = stats_df
            for row in stats_df.itertuples(index=False):
                time_rows.append(
                    {
                        "metric": metric,
                        "experiment": experiment.label,
                        "results_folder_name": experiment.metadata["results_folder_name"],
                        "time_limit_seconds": resolved_limit,
                        "elapsed_time_seconds": row.elapsed_time_seconds,
                        "nfe": row.nfe,
                        "mean": row.mean,
                        "std": row.std,
                        "count": row.count,
                    }
                )
        time_csv = out_dir / f"{safe_metric}_by_time.csv"
        time_png = out_dir / f"{safe_metric}_by_time.png"
        write_stats_csv(time_rows, time_csv)
        plot_stats(
            time_series,
            "elapsed_time_seconds",
            "Elapsed time (s)",
            metric_label(metric, config),
            f"{metric_label(metric, config)} by elapsed time",
            time_png,
            width,
            height,
        )
        written.extend([time_csv, time_png])

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create runchart plots for experiment metrics.")
    parser.add_argument(
        "--config",
        default="algorithms/config/config_process_runchart_results.yaml",
        help="YAML config containing experiment folders, metrics, and plotting options.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    written = build_runcharts(config)
    print(f"Wrote {len(written)} runchart files.")
    if written:
        print(f"Output folder: {written[0].parent}")


if __name__ == "__main__":
    main()
