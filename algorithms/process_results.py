#!/usr/bin/env python3
"""
Consolidated result processing for EIGO experiments.

The script discovers EIGO experiment folders, parses AdamW/population/random result
CSVs, and produces summary tables, evolution plots, image grids, prompt/category
analysis, and image-distance summaries from one configuration file.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    score_col: str
    fitness_col: str


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("aesthetic_score", "Aesthetic score", "aesthetic_score", "max_aesthetic_score"),
    MetricSpec("clip_score", "CLIP score", "clip_score", "max_clip_score"),
    MetricSpec("image_reward_score", "ImageReward score", "image_reward_score", "max_image_reward_score"),
    MetricSpec("hpsv2_score", "HPSv2 score", "hpsv2_score", "max_hpsv2_score"),
    MetricSpec("pickscore_score", "PickScore", "pickscore_score", "max_pickscore_score"),
)

OBJECTIVE_KEY = "objective"
OBJECTIVE_LABEL = "Fitness / combined score"

WEIGHT_KEYS = {
    "aesw": "aesthetic_score",
    "clipw": "clip_score",
    "irw": "image_reward_score",
    "hpsw": "hpsv2_score",
    "psw": "pickscore_score",
}

DEFAULT_LABELS = {
    "adam": "AdamW",
    "cmaes": "CMA-ES",
    "sepcmaes": "sep-CMA-ES",
    "vdcmae": "VD-CMA",
    "ga": "GA",
    "randomsampler": "Random sampler",
}


@dataclass
class PromptRun:
    run_dir: Path
    prompt_dir: Path
    csv_path: Path
    csv_kind: str
    step_name: str
    df: pd.DataFrame
    prompt: str
    category: str
    prompt_id: str
    seed: Optional[int]


@dataclass
class ExperimentRun:
    path: Path
    name: str
    method_key: str
    method_label: str
    weights: Dict[str, int]
    weights_label: str
    seed: Optional[int]
    backend: str
    model_tag: str
    prompt_runs: List[PromptRun]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path_like: str | os.PathLike) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def first_non_empty(series: pd.Series) -> Optional[str]:
    for value in series.dropna():
        text = str(value).strip()
        if text:
            return text
    return None


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def pct_diff(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return 100.0 * (value - baseline) / abs(baseline)


def summarise(values: Iterable[float]) -> Dict[str, float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def parse_weights(name: str) -> Dict[str, int]:
    weights: Dict[str, int] = {}
    for short, metric in WEIGHT_KEYS.items():
        match = re.search(rf"(?:^|_){short}(\d+)(?:_|$)", name.lower())
        if match:
            weights[metric] = int(match.group(1))

    if weights:
        return weights

    old = re.search(r"(?:^|_)a(\d+)_b(\d+)(?:_|$)", name.lower())
    if old:
        return {
            "aesthetic_score": int(old.group(1)),
            "clip_score": int(old.group(2)),
            "image_reward_score": 0,
            "hpsv2_score": 0,
            "pickscore_score": 0,
        }

    return {}


def parse_seed(name: str, weights: Dict[str, int]) -> Optional[int]:
    if weights:
        weight_markers = "|".join(re.escape(k) for k in ["aesw", "a", "clipw"])
        match = re.search(rf"_(\d+)_(?:{weight_markers})", name.lower())
        if match:
            return int(match.group(1))
    candidates = [int(m.group(1)) for m in re.finditer(r"_(\d+)(?:_|$)", name)]
    return candidates[-1] if candidates else None


def parse_method(name: str) -> str:
    lowered = name.lower()
    for key in ("randomsampler", "sepcmaes", "vdcmae", "cmaes", "adam", "ga"):
        if lowered.startswith(key) or f"_{key}_" in lowered:
            return key
    return lowered.split("_", 1)[0] if lowered else "unknown"


def parse_backend_and_model(name: str) -> Tuple[str, str]:
    lowered = name.lower()
    match = re.search(
        r"_clip_[^_]+_(sdxl|flux|pixart|lcm)_([^_]+)_\d+_(?:aesw|a)",
        lowered,
    )
    if match:
        return match.group(1), match.group(2)
    return "", ""


def weights_label(weights: Dict[str, int], metric_order: Sequence[MetricSpec]) -> str:
    if not weights:
        return "unweighted"
    parts = []
    short_by_metric = {v: k for k, v in WEIGHT_KEYS.items()}
    for metric in metric_order:
        val = int(weights.get(metric.key, 0))
        if val != 0 or metric.key in weights:
            parts.append(f"{short_by_metric[metric.key]}={val / 100.0:g}")
    return ", ".join(parts) if parts else "unweighted"


def weights_sort_key(label: str) -> Tuple:
    nums = [float(x) for x in re.findall(r"=([-+]?\d*\.?\d+)", label)]
    return tuple([-n for n in nums]) if nums else (math.inf,)


def prompt_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def read_prompt_csv(prompt_dir: Path) -> Optional[Tuple[pd.DataFrame, Path, str, str]]:
    score_csv = prompt_dir / "score_results.csv"
    fitness_csv = prompt_dir / "fitness_results.csv"
    if score_csv.exists():
        df = pd.read_csv(score_csv)
        return df, score_csv, "score", "iteration"
    if fitness_csv.exists():
        df = pd.read_csv(fitness_csv)
        return df, fitness_csv, "fitness", "generation"
    return None


def clean_prompt_df(df: pd.DataFrame, kind: str, step_name: str) -> Optional[pd.DataFrame]:
    if df.empty or step_name not in df.columns:
        return None
    df = df.copy()
    df[step_name] = pd.to_numeric(df[step_name], errors="coerce")
    df = df[df[step_name].notna()].sort_values(step_name)
    df = df.drop_duplicates(subset=[step_name], keep="last").reset_index(drop=True)
    if df.empty:
        return None

    if "elapsed_time" not in df.columns:
        df["elapsed_time"] = np.nan
    if kind == "score":
        if "combined_score" not in df.columns and "combined_loss" in df.columns:
            df["combined_score"] = 1.0 - numeric_series(df, "combined_loss")
    return df


def load_prompt_run(run_dir: Path, prompt_dir: Path, seed: Optional[int]) -> Optional[PromptRun]:
    loaded = read_prompt_csv(prompt_dir)
    if loaded is None:
        return None
    raw_df, csv_path, kind, step_name = loaded
    df = clean_prompt_df(raw_df, kind, step_name)
    if df is None:
        return None
    prompt = first_non_empty(df["prompt"]) if "prompt" in df.columns else None
    category = first_non_empty(df["category"]) if "category" in df.columns else None
    return PromptRun(
        run_dir=run_dir,
        prompt_dir=prompt_dir,
        csv_path=csv_path,
        csv_kind=kind,
        step_name=step_name,
        df=df,
        prompt=prompt or prompt_dir.name,
        category=category or "Unknown",
        prompt_id=prompt_dir.name,
        seed=seed,
    )


def is_experiment_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("results_") and read_prompt_csv(child) is not None:
            return True
    return False


def discover_run_dirs(config: dict) -> List[Path]:
    explicit = [resolve_path(p) for p in config.get("source_dirs", [])]
    run_dirs = [p for p in explicit if is_experiment_dir(p)]

    skipped_explicit = [p for p in explicit if not is_experiment_dir(p)]
    for skipped in skipped_explicit:
        print(f"Warning: source_dir is not an experiment folder with prompt CSVs: {skipped}")

    discover_cfg = config.get("discover", {}) or {}
    roots = [resolve_path(p) for p in discover_cfg.get("roots", [])]
    recursive = bool(discover_cfg.get("recursive", True))
    max_depth = discover_cfg.get("max_depth", None)
    include = [re.compile(p) for p in discover_cfg.get("include", [])]
    exclude = [re.compile(p) for p in discover_cfg.get("exclude", [])]

    def allowed(path: Path) -> bool:
        rel = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        if include and not any(p.search(rel) for p in include):
            return False
        if exclude and any(p.search(rel) for p in exclude):
            return False
        return True

    for root in roots:
        if not root.exists():
            print(f"Warning: discovery root does not exist: {root}")
            continue
        if is_experiment_dir(root) and allowed(root):
            run_dirs.append(root)
        if not recursive:
            continue
        base_depth = len(root.parts)
        for current, dirnames, _ in os.walk(root):
            cur = Path(current)
            if max_depth is not None and len(cur.parts) - base_depth > int(max_depth):
                dirnames[:] = []
                continue
            if is_experiment_dir(cur) and allowed(cur):
                run_dirs.append(cur)
                dirnames[:] = []

    unique = []
    seen = set()
    for path in run_dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique, key=lambda p: str(p))


def load_experiment(run_dir: Path, config: dict, metric_order: Sequence[MetricSpec]) -> Optional[ExperimentRun]:
    name = run_dir.name
    weights = parse_weights(name)
    seed = parse_seed(name, weights)
    method_key = parse_method(name)
    labels = {**DEFAULT_LABELS, **(config.get("method_labels", {}) or {})}
    backend, model_tag = parse_backend_and_model(name)

    prompt_runs = []
    for child in sorted(run_dir.iterdir(), key=prompt_sort_key):
        if not child.is_dir() or not child.name.startswith("results_"):
            continue
        loaded = load_prompt_run(run_dir, child, seed)
        if loaded is not None:
            prompt_runs.append(loaded)

    if not prompt_runs:
        return None

    return ExperimentRun(
        path=run_dir,
        name=name,
        method_key=method_key,
        method_label=labels.get(method_key, method_key),
        weights=weights,
        weights_label=weights_label(weights, metric_order),
        seed=seed,
        backend=backend,
        model_tag=model_tag,
        prompt_runs=prompt_runs,
    )


def metric_column(prompt_run: PromptRun, metric: MetricSpec) -> str:
    return metric.score_col if prompt_run.csv_kind == "score" else metric.fitness_col


def objective_column(prompt_run: PromptRun) -> str:
    if prompt_run.csv_kind == "score":
        return "combined_score"
    return "max_fitness"


def get_metric_values(prompt_run: PromptRun, metric: MetricSpec) -> pd.Series:
    return numeric_series(prompt_run.df, metric_column(prompt_run, metric))


def get_objective_values(prompt_run: PromptRun) -> pd.Series:
    return numeric_series(prompt_run.df, objective_column(prompt_run))


def final_value(prompt_run: PromptRun, metric: MetricSpec) -> float:
    vals = get_metric_values(prompt_run, metric).dropna()
    return float(vals.iloc[-1]) if not vals.empty else np.nan


def baseline_value(prompt_run: PromptRun, metric: MetricSpec) -> float:
    vals = get_metric_values(prompt_run, metric).dropna()
    return float(vals.iloc[0]) if not vals.empty else np.nan


def final_objective(prompt_run: PromptRun) -> float:
    vals = get_objective_values(prompt_run).dropna()
    return float(vals.iloc[-1]) if not vals.empty else np.nan


def baseline_objective(prompt_run: PromptRun) -> float:
    vals = get_objective_values(prompt_run).dropna()
    return float(vals.iloc[0]) if not vals.empty else np.nan


def elapsed_final(prompt_run: PromptRun) -> float:
    vals = numeric_series(prompt_run.df, "elapsed_time").dropna()
    return float(vals.iloc[-1]) if not vals.empty else np.nan


def enabled_metrics(config: dict) -> List[MetricSpec]:
    selected = config.get("metrics", "auto")
    if selected == "auto" or selected is None:
        return list(METRICS)
    selected_set = set(selected)
    by_key = {m.key: m for m in METRICS}
    missing = selected_set - set(by_key)
    if missing:
        raise ValueError(f"Unknown metrics in config: {sorted(missing)}")
    return [m for m in METRICS if m.key in selected_set]


def write_summary_tables(runs: Sequence[ExperimentRun], out_dir: Path, metrics: Sequence[MetricSpec]) -> None:
    run_rows = []
    for run in runs:
        row = {
            "method": run.method_label,
            "method_key": run.method_key,
            "weights": run.weights_label,
            "seed": run.seed,
            "backend": run.backend,
            "model": run.model_tag,
            "n_prompts": len(run.prompt_runs),
            "folder": str(run.path.relative_to(REPO_ROOT) if run.path.is_relative_to(REPO_ROOT) else run.path),
        }
        for metric in metrics:
            finals = [final_value(p, metric) for p in run.prompt_runs]
            bases = [baseline_value(p, metric) for p in run.prompt_runs]
            fs = summarise(finals)
            bs = summarise(bases)
            row[f"{metric.key}_baseline_mean"] = bs["mean"]
            row[f"{metric.key}_mean"] = fs["mean"]
            row[f"{metric.key}_std"] = fs["std"]
            row[f"{metric.key}_max"] = fs["max"]
            row[f"{metric.key}_diff_to_baseline_pct"] = pct_diff(fs["mean"], bs["mean"])
        obj = summarise([final_objective(p) for p in run.prompt_runs])
        obj_base = summarise([baseline_objective(p) for p in run.prompt_runs])
        elapsed = summarise([elapsed_final(p) for p in run.prompt_runs])
        row["objective_baseline_mean"] = obj_base["mean"]
        row["objective_mean"] = obj["mean"]
        row["objective_std"] = obj["std"]
        row["objective_max"] = obj["max"]
        row["objective_diff_to_baseline_pct"] = pct_diff(obj["mean"], obj_base["mean"])
        row["elapsed_time_mean"] = elapsed["mean"]
        row["elapsed_time_std"] = elapsed["std"]
        run_rows.append(row)

    by_run = pd.DataFrame(run_rows).sort_values(["weights", "method", "seed"], na_position="last")

    group_cols = ["method", "method_key", "weights", "backend", "model"]
    numeric_cols = [c for c in by_run.columns if c not in group_cols + ["folder"]]
    agg_spec = {c: "mean" for c in numeric_cols if c != "seed"}
    agg_spec["seed"] = "count"
    by_method = by_run.groupby(group_cols, dropna=False, as_index=False).agg(agg_spec)
    by_method = by_method.rename(columns={"seed": "n_seeds"})

    out_path = out_dir / "summary_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        by_run.to_excel(writer, sheet_name="by_run", index=False)
        by_method.to_excel(writer, sheet_name="by_method", index=False)
    by_run.to_csv(out_dir / "summary_results_by_run.csv", index=False)
    by_method.to_csv(out_dir / "summary_results_by_method.csv", index=False)
    print(f"Saved: {out_path}")


def resample_to_percent(values: Sequence[float], target_percent: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.full_like(target_percent, np.nan, dtype=float)
    if arr.size == 1:
        return np.full_like(target_percent, arr[0], dtype=float)
    src = np.linspace(0.0, 100.0, arr.size)
    return np.interp(target_percent, src, arr)


def stack_stats(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.vstack(curves)
    count = np.sum(np.isfinite(mat), axis=0).astype(float)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=0)
    return mean, std, count


def plot_series_ci(ax, x, mean, std, count, label):
    valid = np.isfinite(x) & np.isfinite(mean)
    if not valid.any():
        return
    xv = x[valid]
    mv = mean[valid]
    sv = std[valid]
    cv = count[valid]
    line = ax.plot(xv, mv, label=label)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = 1.96 * np.where(cv > 0, sv / np.sqrt(cv), np.nan)
    lower = mv - ci
    upper = mv + ci
    ci_valid = np.isfinite(lower) & np.isfinite(upper)
    if ci_valid.sum() > 1:
        ax.fill_between(xv[ci_valid], lower[ci_valid], upper[ci_valid], color=line.get_color(), alpha=0.15)


def collect_curves(run: ExperimentRun, metric_key: str, metric: Optional[MetricSpec], x_percent: np.ndarray) -> List[np.ndarray]:
    curves = []
    for prompt_run in run.prompt_runs:
        if metric_key == OBJECTIVE_KEY:
            values = get_objective_values(prompt_run)
        elif metric is not None:
            values = get_metric_values(prompt_run, metric)
        else:
            continue
        values = values.dropna().to_numpy(dtype=float)
        if values.size:
            curves.append(resample_to_percent(values, x_percent))
    return curves


def create_evolution_plots(runs: Sequence[ExperimentRun], out_dir: Path, metrics: Sequence[MetricSpec], config: dict) -> None:
    plot_dir = out_dir / "evolution_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    points = int(config.get("evolution_points", 101))
    x_percent = np.linspace(0.0, 100.0, points)

    plot_metrics: List[Tuple[str, str, Optional[MetricSpec]]] = [
        (m.key, m.label, m) for m in metrics
    ] + [(OBJECTIVE_KEY, OBJECTIVE_LABEL, None)]

    grouped: Dict[Tuple[str, str], List[ExperimentRun]] = {}
    for run in runs:
        grouped.setdefault((run.weights_label, run.method_label), []).append(run)

    for metric_key, label, metric in plot_metrics:
        rows = []
        by_weight: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for (w_label, method_label), group_runs in grouped.items():
            curves = []
            for run in group_runs:
                curves.extend(collect_curves(run, metric_key, metric, x_percent))
            if not curves:
                continue
            mean, std, count = stack_stats(curves)
            by_weight.setdefault(w_label, {})[method_label] = (mean, std, count)
            for x, avg, sd, n in zip(x_percent, mean, std, count):
                rows.append({
                    "weights": w_label,
                    "method": method_label,
                    "metric": metric_key,
                    "iteration_percent": x,
                    "mean": avg,
                    "std": sd,
                    "count": n,
                })

        if rows:
            pd.DataFrame(rows).to_csv(plot_dir / f"{metric_key}_evolution.csv", index=False)

        for w_label, methods in sorted(by_weight.items(), key=lambda item: weights_sort_key(item[0])):
            fig, ax = plt.subplots(figsize=(10, 6))
            for method_label, (mean, std, count) in sorted(methods.items()):
                plot_series_ci(ax, x_percent, mean, std, count, method_label)
            ax.set_xlabel("Optimization progress (%)")
            ax.set_ylabel(label)
            ax.set_title(f"{label} evolution ({w_label})")
            ax.grid(alpha=0.3)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
            fig.tight_layout()
            safe_w = re.sub(r"[^a-zA-Z0-9]+", "_", w_label).strip("_") or "unweighted"
            out_path = plot_dir / f"{metric_key}_evolution_{safe_w}.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved evolution plots under: {plot_dir}")


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def font_height(font: ImageFont.ImageFont) -> int:
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent + 3
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1] + 3


def draw_centered_lines(draw, lines, font, x0, x1, y, fill=(0, 0, 0)) -> int:
    height = font_height(font)
    for idx, line in enumerate(lines):
        width = draw.textlength(line, font=font)
        draw.text((x0 + (x1 - x0 - width) / 2, y + idx * height), line, font=font, fill=fill)
    return len(lines) * height


def image_grid_scores(prompt_run: PromptRun, metrics: Sequence[MetricSpec], initial: bool = False) -> Dict[str, float]:
    metric_by_key = {metric.key: metric for metric in metrics}
    aesthetic_metric = metric_by_key.get("aesthetic_score")
    clip_metric = metric_by_key.get("clip_score")
    return {
        "fitness": baseline_objective(prompt_run) if initial else final_objective(prompt_run),
        "aesthetic": (
            baseline_value(prompt_run, aesthetic_metric)
            if initial and aesthetic_metric is not None
            else final_value(prompt_run, aesthetic_metric)
            if aesthetic_metric is not None
            else np.nan
        ),
        "clip": (
            baseline_value(prompt_run, clip_metric)
            if initial and clip_metric is not None
            else final_value(prompt_run, clip_metric)
            if clip_metric is not None
            else np.nan
        ),
    }


def image_grid_score_text(scores: Dict[str, float]) -> str:
    parts = [
        ("Aes", scores.get("aesthetic", np.nan)),
        ("CLIP", scores.get("clip", np.nan)),
        ("Fit", scores.get("fitness", np.nan)),
    ]
    return "  ".join(f"{name}: {value:.3g}" for name, value in parts if np.isfinite(value))


def best_score_color(scores: Dict[str, float], row_scores: Sequence[Dict[str, float]]) -> Tuple[int, int, int]:
    def best(metric: str) -> float:
        values = [item.get(metric, np.nan) for item in row_scores]
        values = [value for value in values if np.isfinite(value)]
        return max(values) if values else np.nan

    if np.isfinite(scores.get("fitness", np.nan)) and np.isclose(scores["fitness"], best("fitness")):
        return (102, 0, 153)
    if np.isfinite(scores.get("aesthetic", np.nan)) and np.isclose(scores["aesthetic"], best("aesthetic")):
        return (200, 0, 0)
    if np.isfinite(scores.get("clip", np.nan)) and np.isclose(scores["clip"], best("clip")):
        return (0, 0, 200)
    return (0, 0, 0)


def create_image_grids(runs: Sequence[ExperimentRun], out_dir: Path, metrics: Sequence[MetricSpec], config: dict) -> None:
    grid_cfg = config.get("image_grid", {}) or {}
    max_rows = int(grid_cfg.get("max_rows", 24))
    max_cols = int(grid_cfg.get("max_method_columns", 8))
    tile_size = grid_cfg.get("tile_size", [256, 256])
    tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    margin = 18
    gap = 12
    row_gap = 18
    prompt_font = load_font(int(grid_cfg.get("prompt_font_size", 15)))
    title_font = load_font(int(grid_cfg.get("title_font_size", 13)))
    out_grid = out_dir / "image_grids"
    out_grid.mkdir(parents=True, exist_ok=True)

    by_weight: Dict[str, List[ExperimentRun]] = {}
    for run in runs:
        by_weight.setdefault(run.weights_label, []).append(run)

    saved = 0
    for w_label, group_runs in sorted(by_weight.items(), key=lambda item: weights_sort_key(item[0])):
        selected_runs = sorted(group_runs, key=lambda r: (r.method_label, r.seed or -1, r.name))[:max_cols]
        by_prompt: Dict[str, Dict[str, PromptRun]] = {}
        prompt_text: Dict[str, str] = {}
        for run in selected_runs:
            for prompt_run in run.prompt_runs:
                key = prompt_run.prompt
                by_prompt.setdefault(key, {})[run.name] = prompt_run
                prompt_text[key] = prompt_run.prompt

        rows = []
        for key in sorted(by_prompt.keys(), key=str.lower)[:max_rows]:
            entries = []
            for run in selected_runs:
                prompt_run = by_prompt[key].get(run.name)
                if prompt_run is None:
                    entries.append(None)
                    continue
                base = prompt_run.prompt_dir / "it_0.png"
                best = prompt_run.prompt_dir / "best_all.png"
                entries.append(prompt_run if base.exists() and best.exists() else None)
            if any(entries):
                rows.append((key, entries))

        if not rows:
            continue

        cols = len(selected_runs) + 1
        grid_w = cols * tile_w + (cols - 1) * gap
        canvas_w = grid_w + 2 * margin
        x_positions = [margin + idx * (tile_w + gap) for idx in range(cols)]
        row_images = []

        for row_idx, (prompt_key, entries) in enumerate(rows):
            present = [entry for entry in entries if entry is not None]
            baseline_source = present[0]
            baseline_img = baseline_source.prompt_dir / "it_0.png"
            imgs = [baseline_img] + [
                entry.prompt_dir / "best_all.png" if entry is not None else None for entry in entries
            ]

            prompt_canvas = Image.new("RGB", (canvas_w, 100), (255, 255, 255))
            prompt_draw = ImageDraw.Draw(prompt_canvas)
            prompt_lines = wrap_text(prompt_draw, prompt_text.get(prompt_key, prompt_key), prompt_font, grid_w)
            prompt_h = len(prompt_lines) * font_height(prompt_font)
            title_lines = 2 if row_idx == 0 else 1
            title_h = title_lines * font_height(title_font)
            row_h = prompt_h + 6 + title_h + 6 + tile_h
            row_img = Image.new("RGB", (canvas_w, row_h), (255, 255, 255))
            draw = ImageDraw.Draw(row_img)
            draw_centered_lines(draw, prompt_lines, prompt_font, margin, margin + grid_w, 0)
            y_titles = prompt_h + 6
            y_img = y_titles + title_h + 6

            labels = ["Initial"] + [f"{r.method_label}" + (f" s={r.seed}" if r.seed is not None else "") for r in selected_runs]
            row_scores = [image_grid_scores(baseline_source, metrics, initial=True)]
            for entry in entries:
                row_scores.append(image_grid_scores(entry, metrics) if entry is not None else {})
            score_texts = [image_grid_score_text(scores) for scores in row_scores]
            title_colors = [best_score_color(scores, row_scores) for scores in row_scores]

            for col_idx, img_path in enumerate(imgs):
                lines = [labels[col_idx], score_texts[col_idx]] if row_idx == 0 else [score_texts[col_idx]]
                draw_centered_lines(
                    draw,
                    [line[:80] for line in lines],
                    title_font,
                    x_positions[col_idx],
                    x_positions[col_idx] + tile_w,
                    y_titles,
                    fill=title_colors[col_idx],
                )
                if img_path is None or not img_path.exists():
                    continue
                with Image.open(img_path).convert("RGB") as img:
                    img = img.resize((tile_w, tile_h), Image.LANCZOS)
                    row_img.paste(img, (x_positions[col_idx], y_img))
            row_images.append(row_img)

        canvas_h = sum(img.height for img in row_images) + row_gap * (len(row_images) - 1) + 2 * margin
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        y = margin
        for row_img in row_images:
            canvas.paste(row_img, (0, y))
            y += row_img.height + row_gap
        safe_w = re.sub(r"[^a-zA-Z0-9]+", "_", w_label).strip("_") or "unweighted"
        out_path = out_grid / f"generated_image_comparison_grid_{safe_w}.png"
        canvas.save(out_path)
        saved += 1
        print(f"Saved: {out_path}")

    if saved == 0:
        print("Warning: no image grids were created; missing it_0.png/best_all.png pairs.")


def create_prompt_category_tables(runs: Sequence[ExperimentRun], out_dir: Path, metrics: Sequence[MetricSpec]) -> None:
    rows = []
    for run in runs:
        for prompt_run in run.prompt_runs:
            row = {
                "method": run.method_label,
                "method_key": run.method_key,
                "weights": run.weights_label,
                "seed": run.seed,
                "prompt": prompt_run.prompt,
                "category": prompt_run.category,
                "folder": str(prompt_run.prompt_dir.relative_to(REPO_ROOT) if prompt_run.prompt_dir.is_relative_to(REPO_ROOT) else prompt_run.prompt_dir),
                "objective": final_objective(prompt_run),
                "baseline_objective": baseline_objective(prompt_run),
                "objective_diff_to_baseline_pct": pct_diff(final_objective(prompt_run), baseline_objective(prompt_run)),
                "elapsed_time": elapsed_final(prompt_run),
            }
            for metric in metrics:
                final = final_value(prompt_run, metric)
                base = baseline_value(prompt_run, metric)
                row[metric.key] = final
                row[f"baseline_{metric.key}"] = base
                row[f"{metric.key}_diff_to_baseline_pct"] = pct_diff(final, base)
            rows.append(row)

    if not rows:
        print("Warning: no prompt rows found for prompt/category analysis.")
        return

    prompt_df = pd.DataFrame(rows)
    group_cols = ["prompt", "category", "method", "method_key", "weights"]
    value_cols = [
        c for c in prompt_df.columns
        if c not in group_cols + ["seed", "folder"] and pd.api.types.is_numeric_dtype(prompt_df[c])
    ]
    per_prompt = prompt_df.groupby(group_cols, dropna=False, as_index=False).agg(
        **{f"{c}_mean": (c, "mean") for c in value_cols},
        **{f"{c}_std": (c, lambda s: s.std(ddof=0)) for c in value_cols},
        n_seeds=("seed", "count"),
    )

    cat_cols = ["category", "method", "method_key", "weights"]
    per_category = per_prompt.groupby(cat_cols, dropna=False, as_index=False).agg(
        **{c: (c, "mean") for c in per_prompt.columns if c.endswith("_mean")},
        n_prompts=("prompt", "count"),
    )

    out_path = out_dir / "prompt_category_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        prompt_df.to_excel(writer, sheet_name="raw_prompt_runs", index=False)
        per_prompt.to_excel(writer, sheet_name="per_prompt", index=False)
        per_category.to_excel(writer, sheet_name="per_category", index=False)
    prompt_df.to_csv(out_dir / "prompt_results_raw.csv", index=False)
    per_prompt.to_csv(out_dir / "results_per_prompt.csv", index=False)
    per_category.to_csv(out_dir / "results_per_category.csv", index=False)
    print(f"Saved: {out_path}")
    plot_category_bars(per_category, out_dir, metrics)


def plot_category_bars(per_category: pd.DataFrame, out_dir: Path, metrics: Sequence[MetricSpec]) -> None:
    plot_dir = out_dir / "plots_by_category"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [(f"{m.key}_mean", m.label) for m in metrics if f"{m.key}_mean" in per_category.columns]
    if "objective_mean" in per_category.columns:
        metric_cols.append(("objective_mean", OBJECTIVE_LABEL))

    for weights in sorted(per_category["weights"].dropna().unique().tolist(), key=weights_sort_key):
        sub = per_category[per_category["weights"] == weights]
        categories = sorted(sub["category"].dropna().unique().tolist(), key=str.lower)
        methods = sorted(sub["method"].dropna().unique().tolist(), key=str.lower)
        if not categories or not methods:
            continue
        x = np.arange(len(categories))
        width = 0.8 / max(1, len(methods))
        safe_w = re.sub(r"[^a-zA-Z0-9]+", "_", weights).strip("_") or "unweighted"
        for col, label in metric_cols:
            fig, ax = plt.subplots(figsize=(max(10, len(categories) * 0.8), 6))
            for idx, method in enumerate(methods):
                vals = []
                for category in categories:
                    row = sub[(sub["category"] == category) & (sub["method"] == method)]
                    vals.append(float(row.iloc[0][col]) if not row.empty and pd.notna(row.iloc[0][col]) else np.nan)
                ax.bar(x + idx * width, vals, width, label=method)
            ax.set_title(f"{label} by category ({weights})")
            ax.set_xlabel("Category")
            ax.set_ylabel(label)
            ax.set_xticks(x + (len(methods) - 1) * width / 2)
            ax.set_xticklabels(categories, rotation=90)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
            fig.tight_layout()
            out_path = plot_dir / f"per_category_{col}_{safe_w}.png"
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)


def load_gray_image(path: Path, max_side: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    with Image.open(path).convert("L") as img:
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)


def compute_ssim(path_a: Path, path_b: Path, max_side: int) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception as exc:
        print(f"Warning: skimage unavailable for SSIM ({exc}).")
        return np.nan
    arr_a = load_gray_image(path_a, max_side)
    arr_b = load_gray_image(path_b, max_side)
    if arr_a is None or arr_b is None:
        return np.nan
    h = min(arr_a.shape[0], arr_b.shape[0])
    w = min(arr_a.shape[1], arr_b.shape[1])
    if h < 8 or w < 8:
        return np.nan
    return safe_float(ssim(arr_a[:h, :w], arr_b[:h, :w], data_range=255))


def load_clip_bundle(device: Optional[str] = None):
    import torch
    import clip

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def clip_embedding(path: Path, bundle) -> Optional[np.ndarray]:
    import torch

    model, preprocess, device = bundle
    if not path.exists():
        return None
    with Image.open(path).convert("RGB") as img:
        tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(tensor).float().cpu().numpy().flatten()
    return vec


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return np.nan
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return np.nan
    return float(np.dot(a, b) / den)


def create_distance_tables(runs: Sequence[ExperimentRun], out_dir: Path, config: dict) -> None:
    dist_cfg = config.get("distance", {}) or {}
    compute_clip = bool(dist_cfg.get("clip_similarity", False))
    compute_ssim_enabled = bool(dist_cfg.get("ssim", True))
    max_side = int(dist_cfg.get("ssim_max_side", 256))

    rows = []
    clip_bundle = None
    if compute_clip:
        try:
            clip_bundle = load_clip_bundle(dist_cfg.get("clip_device"))
        except Exception as exc:
            print(f"Warning: CLIP similarity unavailable ({exc}).")
            clip_bundle = None

    for run in runs:
        precomp = run.path / "aggregate_prompt_similarity_values.csv"
        precomp_df = pd.read_csv(precomp) if precomp.exists() else None
        for idx, prompt_run in enumerate(run.prompt_runs):
            base = prompt_run.prompt_dir / "it_0.png"
            best = prompt_run.prompt_dir / "best_all.png"
            cosine = np.nan
            ssim_value = np.nan
            if precomp_df is not None and idx < len(precomp_df):
                if "cosine_similarity" in precomp_df.columns:
                    cosine = safe_float(precomp_df.iloc[idx]["cosine_similarity"])
                if "ssim" in precomp_df.columns:
                    ssim_value = safe_float(precomp_df.iloc[idx]["ssim"])
            if compute_clip and clip_bundle is not None and not np.isfinite(cosine):
                cosine = cosine_similarity(clip_embedding(base, clip_bundle), clip_embedding(best, clip_bundle))
            if compute_ssim_enabled and not np.isfinite(ssim_value):
                ssim_value = compute_ssim(base, best, max_side)
            rows.append({
                "method": run.method_label,
                "method_key": run.method_key,
                "weights": run.weights_label,
                "seed": run.seed,
                "prompt": prompt_run.prompt,
                "category": prompt_run.category,
                "cosine_similarity": cosine,
                "ssim": ssim_value,
                "folder": str(prompt_run.prompt_dir.relative_to(REPO_ROOT) if prompt_run.prompt_dir.is_relative_to(REPO_ROOT) else prompt_run.prompt_dir),
            })

    if not rows:
        print("Warning: no image-distance rows found.")
        return

    values = pd.DataFrame(rows)
    group_cols = ["method", "method_key", "weights"]
    summary = values.groupby(group_cols, dropna=False, as_index=False).agg(
        cosine_similarity_mean=("cosine_similarity", "mean"),
        cosine_similarity_std=("cosine_similarity", lambda s: s.std(ddof=0)),
        cosine_similarity_max=("cosine_similarity", "max"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", lambda s: s.std(ddof=0)),
        ssim_max=("ssim", "max"),
        n_prompts=("prompt", "count"),
    )

    out_path = out_dir / "distance_summary.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        values.to_excel(writer, sheet_name="per_prompt", index=False)
    values.to_csv(out_dir / "distance_values.csv", index=False)
    summary.to_csv(out_dir / "distance_summary.csv", index=False)
    print(f"Saved: {out_path}")
    plot_distance_boxplots(values, out_dir)


def plot_distance_boxplots(values: pd.DataFrame, out_dir: Path) -> None:
    for metric, label in (("cosine_similarity", "CLIP cosine similarity"), ("ssim", "SSIM")):
        if metric not in values.columns or not values[metric].notna().any():
            continue
        df = values.copy()
        df["group"] = df["weights"].astype(str)
        groups = sorted(df["group"].dropna().unique().tolist(), key=weights_sort_key)
        methods = sorted(df["method"].dropna().unique().tolist(), key=str.lower)
        x = np.arange(len(groups))
        width = 0.8 / max(1, len(methods))
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for idx, method in enumerate(methods):
            data = []
            positions = []
            for group in groups:
                vals = pd.to_numeric(
                    df[(df["group"] == group) & (df["method"] == method)][metric],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                data.append(vals)
                positions.append(float(x[groups.index(group)] + idx * width))
            if not data:
                continue
            bp = ax.boxplot(data, positions=positions, widths=width * 0.85, patch_artist=True, showfliers=True)
            color = cmap(idx % 10)
            for box in bp["boxes"]:
                box.set(facecolor=color, alpha=0.35, edgecolor=color)
            for item in bp["whiskers"] + bp["caps"] + bp["medians"]:
                item.set(color=color)
            ax.plot([], [], color=color, linewidth=8, alpha=0.35, label=method)
        ax.set_title(f"{label} to initial image")
        ax.set_xlabel("Weights")
        ax.set_ylabel(label)
        ax.set_xticks(x + (len(methods) - 1) * width / 2)
        ax.set_xticklabels(groups, rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        fig.tight_layout()
        out_path = out_dir / f"{metric}_boxplot_by_weight.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def run_pipeline(config: dict) -> None:
    metrics = enabled_metrics(config)
    out_dir = resolve_path(config.get("save_folder", "results_processed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(config)
    if not run_dirs:
        raise FileNotFoundError(
            "No experiment folders found. Configure source_dirs or discover.roots in the process-results config."
        )

    runs = []
    for run_dir in run_dirs:
        run = load_experiment(run_dir, config, metrics)
        if run is not None:
            runs.append(run)

    if not runs:
        raise FileNotFoundError("No valid prompt-level score_results.csv or fitness_results.csv files were parsed.")

    manifest = pd.DataFrame([
        {
            "folder": str(run.path.relative_to(REPO_ROOT) if run.path.is_relative_to(REPO_ROOT) else run.path),
            "method": run.method_label,
            "method_key": run.method_key,
            "weights": run.weights_label,
            "seed": run.seed,
            "backend": run.backend,
            "model": run.model_tag,
            "n_prompts": len(run.prompt_runs),
        }
        for run in runs
    ])
    manifest.to_csv(out_dir / "run_manifest.csv", index=False)
    print(f"Parsed {len(runs)} experiment folders and {sum(len(r.prompt_runs) for r in runs)} prompt runs.")

    outputs = config.get("outputs", {}) or {}
    if outputs.get("summary", True):
        write_summary_tables(runs, out_dir, metrics)
    if outputs.get("evolution_plots", True):
        create_evolution_plots(runs, out_dir, metrics, config)
    if outputs.get("image_grid", True):
        create_image_grids(runs, out_dir, metrics, config)
    if outputs.get("prompt_category_analysis", True):
        create_prompt_category_tables(runs, out_dir, metrics)
    if outputs.get("distance_table", True):
        create_distance_tables(runs, out_dir, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process EIGO experiment results.")
    parser.add_argument(
        "--config",
        type=str,
        default="algorithms/config/config_process_results.yaml",
        help="Path to the process-results YAML config.",
    )
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    run_pipeline(config)


if __name__ == "__main__":
    main()
