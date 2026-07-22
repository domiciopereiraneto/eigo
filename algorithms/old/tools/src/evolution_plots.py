import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_W_PAT = re.compile(r"_a(\d+)_b(\d+)")


def _parse_weights(name: str):
    m = _W_PAT.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _detect_mode(folder: Path):
    for prompt_dir in sorted(folder.iterdir()):
        if not prompt_dir.is_dir() or not prompt_dir.name.startswith("results_"):
            continue
        if (prompt_dir / "score_results.csv").exists():
            return "iteration"
        if (prompt_dir / "fitness_results.csv").exists():
            return "generation"
    return None


def _load_prompt_runs(folder: Path):
    mode = _detect_mode(folder)
    if mode is None:
        return None, []

    runs = []
    for prompt_dir in sorted(folder.iterdir()):
        if not prompt_dir.is_dir() or not prompt_dir.name.startswith("results_"):
            continue

        if mode == "iteration":
            csv_path = prompt_dir / "score_results.csv"
            step_col = "iteration"
            aes_col = "aesthetic_score"
            clip_col = "clip_score"
        else:
            csv_path = prompt_dir / "fitness_results.csv"
            step_col = "generation"
            aes_col = "max_aesthetic_score"
            clip_col = "max_clip_score"

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        needed = {step_col, aes_col, clip_col, "elapsed_time"}
        if not needed.issubset(df.columns):
            continue

        step = pd.to_numeric(df[step_col], errors="coerce")
        aes = pd.to_numeric(df[aes_col], errors="coerce")
        clip = pd.to_numeric(df[clip_col], errors="coerce")
        elapsed = pd.to_numeric(df["elapsed_time"], errors="coerce")
        valid = step.notna() & aes.notna() & clip.notna() & elapsed.notna()
        if not valid.any():
            continue

        clean = pd.DataFrame(
            {"step": step[valid], "aes": aes[valid], "clip": clip[valid], "elapsed": elapsed[valid]}
        ).sort_values("step")
        clean = clean.drop_duplicates(subset=["step"], keep="last")
        if clean.empty:
            continue

        runs.append(
            {
                "aes": clean["aes"].to_numpy(dtype=float),
                "clip": clean["clip"].to_numpy(dtype=float),
                "elapsed": clean["elapsed"].to_numpy(dtype=float),
            }
        )

    return mode, runs


def _resample_to_percent(arr: np.ndarray, target_percent: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.full_like(target_percent, np.nan, dtype=float)
    src_percent = np.linspace(0.0, 100.0, arr.size)
    return np.interp(target_percent, src_percent, arr)


def _stack_stats(curves: List[np.ndarray]):
    mat = np.vstack(curves)
    count = np.sum(np.isfinite(mat), axis=0).astype(float)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=0)
    return mean, std, count


def _ci95(mean: np.ndarray, std: np.ndarray, count: np.ndarray):
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = np.where(count > 0, std / np.sqrt(count), np.nan)
    ci = 1.96 * sem
    return mean - ci, mean + ci


def _build_method_metrics(folder: Path, a_int: int, b_int: int, aesthetic_max: float, clip_max: float):
    mode, runs = _load_prompt_runs(folder)
    if mode is None or not runs:
        return None

    min_len = min(len(r["aes"]) for r in runs)
    if min_len < 1:
        return None

    x_percent = np.linspace(0.0, 100.0, min_len)
    aes_curves = [_resample_to_percent(r["aes"], x_percent) for r in runs]
    clip_curves = [_resample_to_percent(r["clip"], x_percent) for r in runs]
    time_curves = [_resample_to_percent(r["elapsed"], x_percent) for r in runs]

    a_frac = a_int / 100.0
    b_frac = b_int / 100.0
    fit_curves = [
        a_frac * (ac / aesthetic_max) + b_frac * (cc / clip_max)
        for ac, cc in zip(aes_curves, clip_curves)
    ]

    aes_mean, aes_std, aes_count = _stack_stats(aes_curves)
    clip_mean, clip_std, clip_count = _stack_stats(clip_curves)
    time_mean, time_std, time_count = _stack_stats(time_curves)
    fit_mean, fit_std, fit_count = _stack_stats(fit_curves)

    return {
        "mode": mode,
        "x": x_percent,
        "aes_mean": aes_mean,
        "aes_std": aes_std,
        "aes_count": aes_count,
        "clip_mean": clip_mean,
        "clip_std": clip_std,
        "clip_count": clip_count,
        "time_mean": time_mean,
        "time_std": time_std,
        "time_count": time_count,
        "fitness_mean": fit_mean,
        "fitness_std": fit_std,
        "fitness_count": fit_count,
    }


def _plot_series_with_ci(ax, x, mean, std, count, label):
    valid = np.isfinite(x) & np.isfinite(mean)
    if valid.sum() == 0:
        return
    xv = x[valid]
    mv = mean[valid]
    sv = std[valid]
    cv = count[valid]

    lower, upper = _ci95(mv, sv, cv)
    line = ax.plot(xv, mv, "--", label=label)[0]
    c = line.get_color()
    ci_valid = np.isfinite(lower) & np.isfinite(upper)
    if ci_valid.sum() > 1:
        ax.fill_between(xv[ci_valid], lower[ci_valid], upper[ci_valid], color=c, alpha=0.15)


def _plot_block(block: Dict[str, Dict], labels: Dict[str, str], save_dir: Path, a_int: int, b_int: int):
    if not block:
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    prefixes = [p for p in labels.keys() if p in block]
    if not prefixes:
        return []

    leg = dict(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for p in prefixes:
        m = block[p]["metrics"]
        _plot_series_with_ci(ax, m["x"], m["aes_mean"], m["aes_std"], m["aes_count"], labels[p])
    ax.set_ylim(1, 10)
    ax.set_xlabel("Iteration (%)")
    ax.set_ylabel("Aesthetic score")
    ax.set_title(f"Evolution of Aesthetic Score (a={a_int}, b={b_int})")
    ax.grid(alpha=0.3)
    ax.legend(**leg)
    fig.tight_layout()
    out_aes = save_dir / f"aesthetic_evolution_a{a_int}_b{b_int}.jpg"
    fig.savefig(out_aes, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out_aes)

    fig, ax = plt.subplots(figsize=(10, 6))
    for p in prefixes:
        m = block[p]["metrics"]
        _plot_series_with_ci(ax, m["x"], m["clip_mean"], m["clip_std"], m["clip_count"], labels[p])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Iteration (%)")
    ax.set_ylabel("CLIP score")
    ax.set_title(f"Evolution of CLIP Score (a={a_int}, b={b_int})")
    ax.grid(alpha=0.3)
    ax.legend(**leg)
    fig.tight_layout()
    out_clip = save_dir / f"clip_evolution_a{a_int}_b{b_int}.jpg"
    fig.savefig(out_clip, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out_clip)

    fig, ax = plt.subplots(figsize=(10, 6))
    for p in prefixes:
        m = block[p]["metrics"]
        _plot_series_with_ci(ax, m["x"], m["time_mean"], m["time_std"], m["time_count"], labels[p])
    ax.set_xlabel("Iteration (%)")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title(f"Elapsed Time per Iteration (a={a_int}, b={b_int})")
    ax.grid(alpha=0.3)
    ax.legend(**leg)
    fig.tight_layout()
    out_time = save_dir / f"elapsed_time_a{a_int}_b{b_int}.jpg"
    fig.savefig(out_time, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out_time)

    fig, ax = plt.subplots(figsize=(10, 6))
    for p in prefixes:
        m = block[p]["metrics"]
        _plot_series_with_ci(ax, m["x"], m["fitness_mean"], m["fitness_std"], m["fitness_count"], labels[p])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Iteration (%)")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Evolution of Fitness (a={a_int}, b={b_int})")
    ax.grid(alpha=0.3)
    ax.legend(**leg)
    fig.tight_layout()
    out_fit = save_dir / f"fitness_evolution_a{a_int}_b{b_int}.jpg"
    fig.savefig(out_fit, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out_fit)

    return paths


def create_evolution_plots(
    source_dirs: List[str],
    save_folder: str,
    algo_labels: List[Tuple[str, str]],
    aesthetic_max: float,
    clip_max: float,
) -> None:
    out_root = Path(save_folder) / "evolution_plots"
    out_root.mkdir(parents=True, exist_ok=True)

    labels = {p: l for p, l in algo_labels}
    runs = []

    for source_dir in source_dirs:
        folder = Path(source_dir)
        if not folder.exists() or not folder.is_dir():
            continue

        name = folder.name
        ab = _parse_weights(name)
        if not ab:
            continue

        prefix = next((p for p in labels.keys() if name.startswith(p)), None)
        if prefix is None:
            continue

        metrics = _build_method_metrics(folder, ab[0], ab[1], aesthetic_max, clip_max)
        if metrics is None:
            continue

        runs.append({"a": ab[0], "b": ab[1], "prefix": prefix, "metrics": metrics})

    if not runs:
        print("No prompt-level CSV runs found for the configured prefixes.")
        return None

    blocks: Dict[Tuple[int, int], Dict[str, Dict]] = {}
    for run in runs:
        key = (run["a"], run["b"])
        d = blocks.setdefault(key, {})
        if run["prefix"] not in d:
            d[run["prefix"]] = {"metrics": run["metrics"]}

    for (a_int, b_int), block in sorted(blocks.items(), key=lambda x: (-x[0][0], x[0][1])):
        _plot_block(block, labels, out_root, a_int, b_int)

    print(f"Plots saved under: {out_root}")
    return None
