#!/usr/bin/env python3
"""Compile quantitative tables from completed EIGO experiment folders.

The script discovers prompt-level results, selects one row per prompt CSV, and
aggregates metrics into long/wide CSV files plus an Excel workbook.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the local environment
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_METHOD_LABELS = {
    "adam": "AdamW",
    "cmaes": "CMA-ES",
    "sepcmaes": "sep-CMA-ES",
    "vdcmae": "VD-CMA",
    "snes": "SNES",
    "cosyne": "CoSyNE",
    "ga": "GA",
    "zero_order": "Zero order",
    "zeroorder": "Zero order",
    "randomsampler": "Random sampler",
    "random_sampler": "Random sampler",
}

WEIGHT_TO_METRIC = {
    "aesw": "aesthetic_score",
    "clipw": "clip_score",
    "irw": "image_reward_score",
    "hpsw": "hpsv2_score",
    "psw": "pickscore_score",
    "jpgw": "jpeg_size_kb",
}

CANONICAL_METRIC_COLUMNS = {
    "score": {
        "objective": "combined_score",
        "aesthetic_score": "aesthetic_score",
        "clip_score": "clip_score",
        "image_reward_score": "image_reward_score",
        "hpsv2_score": "hpsv2_score",
        "pickscore_score": "pickscore_score",
        "jpeg_size_kb": "jpeg_size_kb",
        "elapsed_time": "elapsed_time",
        "peak_vram_mb": "peak_vram_mb",
    },
    "fitness": {
        "objective": "max_fitness",
        "aesthetic_score": "max_aesthetic_score",
        "clip_score": "max_clip_score",
        "image_reward_score": "max_image_reward_score",
        "hpsv2_score": "max_hpsv2_score",
        "pickscore_score": "max_pickscore_score",
        "jpeg_size_kb": "min_jpeg_size_kb",
        "elapsed_time": "elapsed_time",
        "peak_vram_mb": "peak_vram_mb",
    },
}

METADATA_COLUMNS = {
    "generation",
    "iteration",
    "prompt",
    "category",
    "elapsed_time",
    "peak_vram_mb",
}


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read config files. Install project requirements "
            "or run `pip install PyYAML`."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(path_like: str | os.PathLike) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def first_non_empty(df: pd.DataFrame, column: str) -> Optional[str]:
    if column not in df.columns:
        return None
    for value in df[column].dropna():
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return None


def read_result_csv(prompt_dir: Path) -> Optional[Tuple[pd.DataFrame, Path, str, str]]:
    score_csv = prompt_dir / "score_results.csv"
    fitness_csv = prompt_dir / "fitness_results.csv"
    if score_csv.exists():
        return pd.read_csv(score_csv), score_csv, "score", "iteration"
    if fitness_csv.exists():
        return pd.read_csv(fitness_csv), fitness_csv, "fitness", "generation"
    return None


def clean_result_df(df: pd.DataFrame, step_column: str) -> pd.DataFrame:
    if step_column not in df.columns:
        return df.reset_index(drop=True)
    out = df.copy()
    out[step_column] = pd.to_numeric(out[step_column], errors="coerce")
    out = out[out[step_column].notna()].sort_values(step_column)
    out = out.drop_duplicates(subset=[step_column], keep="last")
    return out.reset_index(drop=True)


def parse_weights(name: str) -> Dict[str, int]:
    weights: Dict[str, int] = {}
    lowered = name.lower()
    for short_name, metric in WEIGHT_TO_METRIC.items():
        match = re.search(rf"(?:^|_){short_name}(-?\d+)(?:_|$)", lowered)
        if match:
            weights[metric] = int(match.group(1))
    old_style = re.search(r"(?:^|_)a(\d+)_b(\d+)(?:_|$)", lowered)
    if old_style and not weights:
        weights["aesthetic_score"] = int(old_style.group(1))
        weights["clip_score"] = int(old_style.group(2))
    return weights


def parse_method(name: str, run_config: dict) -> str:
    lowered = name.lower()
    for key in (
        "randomsampler",
        "random_sampler",
        "sepcmaes",
        "vdcmae",
        "zeroorder",
        "zero_order",
        "cmaes",
        "cosyne",
        "snes",
        "adam",
        "ga",
    ):
        if lowered.startswith(key) or f"_{key}_" in lowered:
            return key

    configured = run_config.get("optimization_method")
    if configured:
        configured_key = str(configured).lower().replace("-", "_")
        if configured_key == "cmaes":
            variant = str(run_config.get("cmaes_variant", "")).lower()
            if variant == "sep":
                return "sepcmaes"
            if variant == "vd":
                return "vdcmae"
        return configured_key
    return lowered.split("_", 1)[0] if lowered else "unknown"


def infer_model(name: str, run_config: dict) -> str:
    model_id = run_config.get("model_id")
    if model_id:
        return str(model_id)
    match = re.search(
        r"_clip_[^_]+_(sdxl|flux|pixart|lcm|sana|sana_sprint|sd)_([^_]+)_\d+_",
        name.lower(),
    )
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return ""


def infer_backend(name: str, run_config: dict) -> str:
    for key in ("resolved_model_backend", "model_backend"):
        value = run_config.get(key)
        if value:
            return str(value)
    match = re.search(r"_clip_[^_]+_(sdxl|flux|pixart|lcm|sana|sana_sprint|sd)_", name.lower())
    return match.group(1) if match else ""


def infer_optimization_target(run_config: dict) -> str:
    return str(
        run_config.get("resolved_optimization_target")
        or run_config.get("optimization_target")
        or ""
    )


def infer_target_variable(name: str, run_config: dict, weights: Dict[str, int]) -> str:
    nonzero = [metric for metric, weight in weights.items() if weight != 0]
    if len(nonzero) == 1:
        return nonzero[0]
    if len(nonzero) > 1:
        return "+".join(nonzero)

    for metric, config_key in (
        ("aesthetic_score", "aesthetic_score_weight"),
        ("clip_score", "clip_score_weight"),
        ("image_reward_score", "image_reward_score_weight"),
        ("hpsv2_score", "hpsv2_score_weight"),
        ("pickscore_score", "pickscore_score_weight"),
        ("jpeg_size_kb", "jpeg_size_weight"),
    ):
        try:
            if float(run_config.get(config_key, 0.0)) != 0.0:
                nonzero.append(metric)
        except (TypeError, ValueError):
            pass
    if len(nonzero) == 1:
        return nonzero[0]
    if len(nonzero) > 1:
        return "+".join(nonzero)

    lowered = name.lower()
    for token, metric in (
        ("imagereward", "image_reward_score"),
        ("image_reward", "image_reward_score"),
        ("hpsv2", "hpsv2_score"),
        ("pickscore", "pickscore_score"),
        ("clip", "clip_score"),
        ("aesthetic", "aesthetic_score"),
        ("jpeg", "jpeg_size_kb"),
        ("jpg", "jpeg_size_kb"),
    ):
        if re.search(rf"(?:^|_){token}(?:_|$)", lowered):
            return metric
    return ""


def load_prompt_config(prompt_dir: Path) -> dict:
    if yaml is None:
        return {}
    config_path = prompt_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        return load_yaml(config_path)
    except Exception as exc:
        print(f"Warning: could not read {config_path}: {exc}")
        return {}


def is_prompt_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("results_") and read_result_csv(path) is not None


def prompt_dirs(experiment_dir: Path) -> List[Path]:
    return [
        path for path in sorted(experiment_dir.iterdir(), key=prompt_sort_key)
        if is_prompt_dir(path)
    ]


def prompt_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def is_experiment_dir(path: Path) -> bool:
    return path.is_dir() and any(prompt_dirs(path))


def discover_experiment_dirs(paths: Sequence[Path]) -> List[Path]:
    experiment_dirs = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Experiment folder does not exist: {path}")
        if is_experiment_dir(path):
            experiment_dirs.add(path.resolve())
            continue
        for root, _, files in os.walk(path):
            if "score_results.csv" not in files and "fitness_results.csv" not in files:
                continue
            prompt_dir = Path(root)
            if prompt_dir.name.startswith("results_"):
                experiment_dirs.add(prompt_dir.parent.resolve())
    return sorted(experiment_dirs, key=lambda item: str(item))


def selected_row(df: pd.DataFrame, kind: str, row_selector: str) -> pd.Series:
    if df.empty:
        raise ValueError("Cannot select a row from an empty dataframe.")
    selector = row_selector.lower().strip()
    if selector == "last":
        return df.iloc[-1]
    if selector == "first":
        return df.iloc[0]
    if selector == "best":
        objective = "combined_score" if kind == "score" else "max_fitness"
        if objective not in df.columns:
            return df.iloc[-1]
        values = pd.to_numeric(df[objective], errors="coerce")
        if values.notna().any():
            return df.loc[values.idxmax()]
        return df.iloc[-1]
    raise ValueError("row_selector must be one of: last, first, best")


def normalize_metrics(config_metrics, kind: str, df: pd.DataFrame) -> Dict[str, str]:
    columns = df.columns
    available = set(columns)
    if config_metrics is None or config_metrics == "canonical":
        return {
            metric: column
            for metric, column in CANONICAL_METRIC_COLUMNS[kind].items()
            if column in available
        }
    if config_metrics == "auto":
        selected = {}
        for column in columns:
            if column in METADATA_COLUMNS:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.notna().any():
                selected[column] = column
        return selected
    if isinstance(config_metrics, str):
        config_metrics = [config_metrics]
    if not isinstance(config_metrics, list):
        raise ValueError("metrics must be 'canonical', 'auto', or a list of metric names/columns.")

    aliases = CANONICAL_METRIC_COLUMNS[kind]
    selected = {}
    for metric in config_metrics:
        metric_name = str(metric)
        column = aliases.get(metric_name, metric_name)
        if column in available:
            selected[metric_name] = column
        else:
            print(f"Warning: metric column not found for {metric_name}: {column}")
    return selected


def stats(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "min": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(numeric.count()),
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)) if numeric.count() > 1 else 0.0,
        "median": float(numeric.median()),
        "max": float(numeric.max()),
    }


def experiment_metadata(experiment_dir: Path, run_config: dict) -> dict:
    name = experiment_dir.name
    weights = parse_weights(name)
    method_key = parse_method(name, run_config)
    return {
        "algorithm": DEFAULT_METHOD_LABELS.get(method_key, method_key),
        "algorithm_key": method_key,
        "model": infer_model(name, run_config),
        "model_backend": infer_backend(name, run_config),
        "target_variable": infer_target_variable(name, run_config, weights),
        "optimization_target": infer_optimization_target(run_config),
        "results_folder_name": name,
        "results_folder_path": str(experiment_dir),
    }


def collect_prompt_values(
    experiment_dir: Path,
    row_selector: str,
    config_metrics,
) -> Tuple[pd.DataFrame, dict]:
    rows = []
    prompt_configs = []
    for prompt_dir in prompt_dirs(experiment_dir):
        loaded = read_result_csv(prompt_dir)
        if loaded is None:
            continue
        raw_df, csv_path, kind, step_column = loaded
        df = clean_result_df(raw_df, step_column)
        if df.empty:
            print(f"Warning: empty result CSV skipped: {csv_path}")
            continue

        prompt_config = load_prompt_config(prompt_dir)
        if prompt_config:
            prompt_configs.append(prompt_config)
        row = selected_row(df, kind, row_selector)
        metric_columns = normalize_metrics(config_metrics, kind, df)
        values = {
            "prompt_folder": prompt_dir.name,
            "csv_path": str(csv_path),
            "csv_kind": kind,
            "selected_row": row_selector,
            "step_column": step_column,
            "step": row.get(step_column, np.nan),
            "prompt": first_non_empty(df, "prompt") or "",
            "category": first_non_empty(df, "category") or "",
        }
        for metric, column in metric_columns.items():
            values[metric] = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        rows.append(values)

    run_config = prompt_configs[0] if prompt_configs else {}
    return pd.DataFrame(rows), run_config


def summarize_experiment(prompt_values: pd.DataFrame, metadata: dict) -> Tuple[List[dict], dict]:
    base = dict(metadata)
    base["prompt_count"] = int(len(prompt_values))
    metric_names = [
        column
        for column in prompt_values.columns
        if column not in {
            "prompt_folder",
            "csv_path",
            "csv_kind",
            "selected_row",
            "step_column",
            "step",
            "prompt",
            "category",
        }
        and pd.api.types.is_numeric_dtype(prompt_values[column])
    ]

    long_rows = []
    wide_row = dict(base)
    for metric in metric_names:
        metric_stats = stats(prompt_values[metric])
        long_row = {**base, "metric": metric, **metric_stats}
        long_rows.append(long_row)
        for stat_name, value in metric_stats.items():
            wide_row[f"{metric}_{stat_name}"] = value
    return long_rows, wide_row


def output_paths(config: dict) -> Tuple[Path, Path]:
    if config.get("output_excel"):
        excel_path = resolve_path(config["output_excel"])
        out_dir = excel_path.parent
    else:
        out_dir = resolve_path(config.get("save_folder", "results_quantitative"))
        excel_path = out_dir / "quantitative_results.xlsx"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, excel_path


def autosize_excel_columns(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    for idx, column in enumerate(df.columns, start=1):
        series = df[column].astype(str)
        max_width = max([len(str(column)), *series.map(len).head(200).tolist()])
        worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = min(
            max(max_width + 2, 10),
            60,
        )


def build_quantitative_results(config: dict) -> Tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder_values = config.get("experiment_folders", config.get("source_dirs", []))
    if not folder_values:
        raise ValueError("Config must define experiment_folders or source_dirs.")

    experiment_dirs = discover_experiment_dirs([resolve_path(path) for path in folder_values])
    if not experiment_dirs:
        raise FileNotFoundError("No experiment folders with results_* CSVs were found.")

    row_selector = str(config.get("row_selector", "last"))
    config_metrics = config.get("metrics", "canonical")
    long_rows = []
    wide_rows = []
    prompt_value_frames = []

    for experiment_dir in experiment_dirs:
        prompt_values, run_config = collect_prompt_values(experiment_dir, row_selector, config_metrics)
        if prompt_values.empty:
            print(f"Warning: no prompt values parsed for {experiment_dir}")
            continue
        metadata = experiment_metadata(experiment_dir, run_config)
        exp_long, exp_wide = summarize_experiment(prompt_values, metadata)
        long_rows.extend(exp_long)
        wide_rows.append(exp_wide)
        prompt_values = prompt_values.copy()
        for key, value in reversed(metadata.items()):
            prompt_values.insert(0, key, value)
        prompt_value_frames.append(prompt_values)

    if not long_rows:
        raise FileNotFoundError("No quantitative rows were produced.")

    summary_long = pd.DataFrame(long_rows)
    summary_wide = pd.DataFrame(wide_rows)
    prompt_values = pd.concat(prompt_value_frames, ignore_index=True) if prompt_value_frames else pd.DataFrame()

    out_dir, excel_path = output_paths(config)
    summary_long.to_csv(out_dir / "quantitative_summary_long.csv", index=False, na_rep="nan")
    summary_wide.to_csv(out_dir / "quantitative_summary_wide.csv", index=False, na_rep="nan")
    prompt_values.to_csv(out_dir / "quantitative_prompt_values.csv", index=False, na_rep="nan")

    if importlib.util.find_spec("openpyxl") is None:
        raise RuntimeError(
            "openpyxl is required to write the Excel workbook. CSV summaries were "
            f"written to {out_dir}. Install project requirements or run "
            "`pip install openpyxl`."
        )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_wide.to_excel(writer, sheet_name="summary", index=False)
        summary_long.to_excel(writer, sheet_name="summary_by_metric", index=False)
        prompt_values.to_excel(writer, sheet_name="prompt_values", index=False)
        autosize_excel_columns(writer, "summary", summary_wide)
        autosize_excel_columns(writer, "summary_by_metric", summary_long)
        autosize_excel_columns(writer, "prompt_values", prompt_values)

    return excel_path, summary_long, summary_wide, prompt_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile min/mean/std/median/max metrics for experiment folders."
    )
    parser.add_argument(
        "--config",
        default="algorithms/config/config_process_quantitative_results.yaml",
        help="YAML config containing experiment_folders/source_dirs and output settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    excel_path, summary_long, _, prompt_values = build_quantitative_results(config)
    experiments = summary_long["results_folder_name"].nunique()
    metrics = summary_long["metric"].nunique()
    prompts = len(prompt_values)
    print(f"Wrote {excel_path}")
    print(f"Experiments: {experiments}; metrics: {metrics}; prompt rows: {prompts}")


if __name__ == "__main__":
    main()
