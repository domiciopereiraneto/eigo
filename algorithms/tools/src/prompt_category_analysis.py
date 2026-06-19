import re
from pathlib import Path

import pandas as pd


def create_prompt_category_results_comparison(source_dirs, save_path):
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for source_dir in source_dirs:
        run_dir = Path(source_dir)
        if not run_dir.is_dir():
            continue

        approach, a_scaled, b_scaled = build_label(run_dir)
        seed = parse_seed(run_dir.name)
        for prompt_dir in sorted_prompt_dirs(run_dir):
            row = extract_prompt_entry(prompt_dir, run_dir, approach, a_scaled, b_scaled, seed)
            if row is not None:
                entries.append(row)

    if not entries:
        raise SystemExit("No prompt-level CSV files parsed under the provided source_dirs.")

    entries_df = pd.DataFrame(entries)

    key_cols = ["approach", "a", "b", "prompt", "category"]
    value_cols = [
        "aesthetic_score",
        "clip_score",
        "fitness",
        "elapsed_time",
        "baseline_aesthetic_score",
        "baseline_clip_score",
        "baseline_fitness",
    ]
    per_prompt_stats = consolidate_by_seed(entries_df, key_cols, value_cols)
    per_prompt_stats, _ = add_diffs(per_prompt_stats)

    res_per_prompt = build_results_per_prompt(per_prompt_stats)
    res_per_cat = build_results_per_category(res_per_prompt)

    rp_path = out_dir / "results_per_prompt.xlsx"
    rc_path = out_dir / "results_per_category.xlsx"
    res_per_prompt.to_excel(rp_path, index=False)
    res_per_cat.to_excel(rc_path, index=False)
    print(f"Saved: {rp_path}")
    print(f"Saved: {rc_path}")

    plots_dir = out_dir / "plots_by_weight"
    cplots = _plot_per_category_vertical(res_per_cat, plots_dir)
    for pth in cplots:
        print(f"Saved: {pth}")
    return None


def parse_seed(name: str):
    m = re.search(r"_(\d+)_a(\d+)_b(\d+)", name)
    return int(m.group(1)) if m else None


def sorted_prompt_dirs(run_dir: Path):
    pat = re.compile(r"(\d+)$")
    dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("results_")]
    return sorted(dirs, key=lambda p: int(pat.search(p.name).group(1)) if pat.search(p.name) else 10**9)


def parse_adam_steps(text: str):
    m = re.search(r"(\d+)adamsteps", text.lower())
    return int(m.group(1)) if m else None


def parse_weights_numeric(text: str):
    t = text.lower()
    ma = re.search(r"(?:^|[_-])a(\d+)(?:_|$)", t)
    mb = re.search(r"(?:^|[_-])b(\d+)(?:_|$)", t)
    a_scaled = float(ma.group(1)) / 100.0 if ma else None
    b_scaled = float(mb.group(1)) / 100.0 if mb else None
    return a_scaled, b_scaled


def base_approach(text: str):
    t = text.lower()
    if "hybridsep" in t or "hybrid" in t:
        return "Hybrid Adam/sep-CMA-ES"
    if "sepcmaes" in t or "sep-cma-es" in t or "sep_cma_es" in t:
        return "sep-CMA-ES"
    if "adam" in t:
        return "Adam"
    if any(k in t for k in ["baseline", "noopt", "no_opt", "no-optimization", "turbo"]):
        return "SDXL Turbo (no optimization)"
    return "Unknown"


def build_label(folder: Path):
    name = folder.name
    base = base_approach(name)
    steps = parse_adam_steps(name)
    a_scaled, b_scaled = parse_weights_numeric(name)
    approach = f"{base} ({steps} adam steps)" if steps is not None else base
    return approach, a_scaled, b_scaled


def extract_prompt_entry(prompt_dir: Path, run_dir: Path, approach: str, a_scaled, b_scaled, seed):
    score_csv = prompt_dir / "score_results.csv"
    fitness_csv = prompt_dir / "fitness_results.csv"

    if score_csv.exists():
        df = pd.read_csv(score_csv)
        step_col = "iteration"
        aes_col = "aesthetic_score"
        clip_col = "clip_score"
        fit_col = "combined_score"
    elif fitness_csv.exists():
        df = pd.read_csv(fitness_csv)
        step_col = "generation"
        aes_col = "max_aesthetic_score"
        clip_col = "max_clip_score"
        fit_col = "max_fitness"
    else:
        return None

    needed = {step_col, aes_col, clip_col, fit_col, "elapsed_time"}
    if df.empty or not needed.issubset(df.columns):
        return None

    df = df.sort_values(step_col).drop_duplicates(subset=[step_col], keep="last")
    first = df.iloc[0]
    last = df.iloc[-1]

    prompt = str(first["prompt"]).strip() if "prompt" in df.columns and pd.notna(first["prompt"]) else prompt_dir.name
    category = (
        str(first["category"]).strip() if "category" in df.columns and pd.notna(first["category"]) else "Unknown"
    )

    return {
        "seed": seed,
        "prompt": prompt,
        "category": category,
        "approach": approach,
        "a": a_scaled,
        "b": b_scaled,
        "folder": str(prompt_dir.relative_to(run_dir)),
        "aesthetic_score": float(last[aes_col]),
        "clip_score": float(last[clip_col]),
        "fitness": float(last[fit_col]),
        "elapsed_time": float(last["elapsed_time"]),
        "baseline_aesthetic_score": float(first[aes_col]),
        "baseline_clip_score": float(first[clip_col]),
        "baseline_fitness": float(first[fit_col]),
    }


def consolidate_by_seed(entries_df: pd.DataFrame, key_cols, value_cols):
    grouped = entries_df.groupby(key_cols, as_index=False, dropna=False)
    agg = {}
    for c in value_cols:
        agg[c + "_mean"] = (c, "mean")
        agg[c + "_std"] = (c, lambda s: s.std(ddof=0))
    out = grouped.agg(**agg)

    folder_series = (
        entries_df.groupby(key_cols, dropna=False)["folder"]
        .agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
        .reset_index()["folder"]
    )
    out["folder"] = folder_series.values
    return out


def add_diffs(df):
    def _pct(v, b):
        if b is None or pd.isna(b) or b == 0:
            return None
        return 100.0 * (v - b) / abs(b)

    cols = []
    if "aesthetic_score_mean" in df.columns and "baseline_aesthetic_score_mean" in df.columns:
        df["aesthetic_diff_to_baseline_pct"] = [
            _pct(v, b) for v, b in zip(df["aesthetic_score_mean"], df["baseline_aesthetic_score_mean"])
        ]
        cols.append("aesthetic_diff_to_baseline_pct")
    if "clip_score_mean" in df.columns and "baseline_clip_score_mean" in df.columns:
        df["clip_diff_to_baseline_pct"] = [
            _pct(v, b) for v, b in zip(df["clip_score_mean"], df["baseline_clip_score_mean"])
        ]
        cols.append("clip_diff_to_baseline_pct")
    if "fitness_mean" in df.columns and "baseline_fitness_mean" in df.columns:
        df["fitness_diff_to_baseline_pct"] = [
            _pct(v, b) for v, b in zip(df["fitness_mean"], df["baseline_fitness_mean"])
        ]
        cols.append("fitness_diff_to_baseline_pct")
    return df, cols


def build_results_per_prompt(entries_df: pd.DataFrame):
    cols = [
        "prompt",
        "category",
        "approach",
        "a",
        "b",
        "aesthetic_score_mean",
        "aesthetic_score_std",
        "clip_score_mean",
        "clip_score_std",
        "fitness_mean",
        "fitness_std",
        "aesthetic_diff_to_baseline_pct",
        "clip_diff_to_baseline_pct",
        "fitness_diff_to_baseline_pct",
        "elapsed_time_mean",
        "elapsed_time_std",
        "folder",
    ]
    cols = [c for c in cols if c in entries_df.columns]
    merged = entries_df[cols].copy()
    merged = merged.sort_values(["category", "prompt", "approach", "a", "b"]).reset_index(drop=True)
    return merged


def build_results_per_category(per_prompt_df: pd.DataFrame):
    value_cols = [c for c in per_prompt_df.columns if c.endswith("_mean")] + [
        c for c in per_prompt_df.columns if c.endswith("_diff_to_baseline_pct")
    ]
    grouped = per_prompt_df.groupby(["category", "approach", "a", "b"], dropna=False)
    out = grouped.agg({c: "mean" for c in value_cols})
    for c in value_cols:
        stdname = c.replace("_mean", "_std_over_prompts")
        out[stdname] = grouped[c].std(ddof=0)
    out = out.reset_index().sort_values(["category", "approach", "a", "b"]).reset_index(drop=True)
    return out


def _select_category_metrics(df):
    cols = []
    if "aesthetic_score_mean" in df.columns:
        cols.append(("aesthetic_score_mean", "Aesthetic score (mean)"))
    if "clip_score_mean" in df.columns:
        cols.append(("clip_score_mean", "CLIP score (mean)"))
    if "fitness_mean" in df.columns:
        cols.append(("fitness_mean", "Fitness (mean)"))
    return cols


def _plot_per_category_vertical(df, out_dir):
    import numpy as np
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _select_category_metrics(df)
    approach_order = sorted(df["approach"].dropna().unique().tolist())
    weights = df[["a", "b"]].drop_duplicates().sort_values(["a", "b"]).itertuples(index=False, name=None)
    paths = []

    for a_val, b_val in weights:
        sub = df[(df["a"] == a_val) & (df["b"] == b_val)].copy()
        cats = sorted(sub["category"].unique().tolist(), key=lambda x: str(x).lower())
        approaches = [ap for ap in approach_order if ap in sub["approach"].unique()]
        w = max(10, 0.8 * len(cats))
        for col, label in metrics:
            fig, ax = plt.subplots(figsize=(w + 3, 6))
            x = np.arange(len(cats))
            width = 0.8 / max(1, len(approaches))
            for j, ap in enumerate(approaches):
                vals = []
                for c in cats:
                    row = sub[(sub["category"] == c) & (sub["approach"] == ap)]
                    vals.append(float(row.iloc[0][col]) if not row.empty else np.nan)
                vals = np.array(vals, dtype=float)
                ax.bar(x + j * width, vals, width, label=ap)
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        ax.text(x[i] + j * width, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            ax.set_title(f"{label} comparison per category — a={a_val:.1f}, b={b_val:.1f}")
            ax.set_xlabel("Category")
            ax.set_ylabel(label)
            ax.set_xticks(x + (len(approaches) - 1) * width / 2)
            ax.set_xticklabels(cats, rotation=90)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
            fig.tight_layout()
            outp = out_dir / f"per_category_{col}_a{a_val:.1f}_b{b_val:.1f}.jpg"
            fig.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(outp)
    return paths
