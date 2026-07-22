from pathlib import Path
import re
import numpy as np
import pandas as pd

def create_summary_table(results_dirs, save_folder, labels, aesthetic_norm, clip_norm):
    OUT_NAME      = "summary_results.xlsx" 

    build_summary(results_dirs, save_folder, OUT_NAME, aesthetic_norm, clip_norm, labels)

def parse_weights(name: str):
    W_PAT    = re.compile(r"_a(\d+)_b(\d+)")
    m = W_PAT.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def parse_seed(name: str):
    SEED_PAT = re.compile(r"_(\d+)_a(\d+)_b(\d+)")
    m = SEED_PAT.search(name)
    if not m:
        return None
    return int(m.group(1))

def summarise(vals: pd.Series):
    v = pd.Series(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=0)), float(v.max())

def _extract_arrays_from_prompt_csv(prompt_dir: Path):
    score_csv = prompt_dir / "score_results.csv"
    fitness_csv = prompt_dir / "fitness_results.csv"

    if score_csv.exists():
        df = pd.read_csv(score_csv)
        step_col = "iteration"
        aes_col = "aesthetic_score"
        clip_col = "clip_score"
    elif fitness_csv.exists():
        df = pd.read_csv(fitness_csv)
        step_col = "generation"
        aes_col = "max_aesthetic_score"
        clip_col = "max_clip_score"
    else:
        return None

    if df.empty:
        return None

    if step_col in df.columns:
        df = df.sort_values(step_col)

    aes = pd.to_numeric(df[aes_col], errors="coerce")
    cli = pd.to_numeric(df[clip_col], errors="coerce")
    time = pd.to_numeric(df["elapsed_time"], errors="coerce")
    valid = aes.notna() & cli.notna() & time.notna()
    if not valid.any():
        return None

    v = df.loc[valid]
    return {
        "aes_first": float(v.iloc[0][aes_col]),
        "cli_first": float(v.iloc[0][clip_col]),
        "time_first": float(v.iloc[0]["elapsed_time"]),
        "aes_last": float(v.iloc[-1][aes_col]),
        "cli_last": float(v.iloc[-1][clip_col]),
        "time_last": float(v.iloc[-1]["elapsed_time"]),
    }

def load_prompt_level_arrays(experiment_dir: Path):
    aes_first, cli_first, time_first = [], [], []
    aes_last, cli_last, time_last = [], [], []

    for prompt_dir in sorted(experiment_dir.iterdir()):
        if not prompt_dir.is_dir() or not prompt_dir.name.startswith("results_"):
            continue
        vals = _extract_arrays_from_prompt_csv(prompt_dir)
        if vals is None:
            continue
        aes_first.append(vals["aes_first"])
        cli_first.append(vals["cli_first"])
        time_first.append(vals["time_first"])
        aes_last.append(vals["aes_last"])
        cli_last.append(vals["cli_last"])
        time_last.append(vals["time_last"])

    if not aes_last:
        return None

    return (
        pd.Series(aes_first, dtype=float).reset_index(drop=True),
        pd.Series(cli_first, dtype=float).reset_index(drop=True),
        pd.Series(time_first, dtype=float).reset_index(drop=True),
        pd.Series(aes_last, dtype=float).reset_index(drop=True),
        pd.Series(cli_last, dtype=float).reset_index(drop=True),
        pd.Series(time_last, dtype=float).reset_index(drop=True),
    )

def build_summary(results_dir, save_folder, out_name, aesthetic_norm, clip_norm, algo_labels):
    # Scan runs
    runs = []
    baseline_source = None

    for folder in results_dir:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        name = folder.name
        arr = load_prompt_level_arrays(folder)
        if arr is None:
            continue
        aes_first, cli_first, time_first, aes_last, cli_last, time_last = arr

        # capture baseline source once from the first found run
        if baseline_source is None:
            baseline_source = (aes_first, cli_first, time_first)
        a_b = parse_weights(name)
        if not a_b:
            continue
        a, b = a_b
        seed = parse_seed(name)
        runs.append({
            "folder": name,
            "seed": seed,
            "a": a, "b": b,
            "aes_last": aes_last,
            "cli_last": cli_last,
            "time_last": time_last
        })

    if not runs:
        raise FileNotFoundError("No runs with prompt CSVs found under the given directories.")

    # Unique weights
    weight_pairs = sorted({(r["a"], r["b"]) for r in runs}, key=lambda x: (-x[0], x[1]))
    seeds = [r["seed"] for r in runs if r["seed"] is not None]
    seed_val = seeds[0] if seeds else ""

    aes0, cli0, time0 = baseline_source  # baseline arrays (iteration 0)

    # Headers (two-row style)
    hdr_top = [
        '', '', '', '',
        'LAION Aesthetic Predictor V2 Score [1-10]', '', '', '',
        'CLIP [-1, 1]', '', '', '',
        'Fitness [0-1]', '', '', '',
        'Elapsed Time (s)',
        ''
    ]
    hdr_sub = [
        'Algorithm', 'a', 'b', 'Seed',
        'Avg.', 'Std.', 'Max', 'Diff. to baseline (%)',
        'Avg.', 'Std.', 'Max', 'Diff. to baseline (%)',
        'Avg.', 'Std.', 'Max', 'Diff. to baseline (%)',
        'Avg.',
        'Folder'
    ]
    rows = [hdr_top, hdr_sub]

    def append_line(alg_label, a_frac, b_frac, seed, aes_vals, cli_vals, time_vals, folder_name_or_nan,
                    base_aes_mean, base_cli_mean, base_comb_mean):
        aes_mean, aes_std, aes_max = summarise(aes_vals)
        cli_mean, cli_std, cli_max = summarise(cli_vals)
        time_mean, _, _ = summarise(time_vals)
        comb_vals = a_frac*(aes_vals/aesthetic_norm) + b_frac*(cli_vals/clip_norm)
        comb_mean, comb_std, comb_max = summarise(comb_vals)

        # Convert diffs to percentages
        def pct_diff(v, base):
            if base == 0 or np.isnan(base):
                return np.nan
            return 100.0 * (v - base) / base

        r = [
            alg_label,
            round(a_frac, 2), round(b_frac, 2), seed,
            round(aes_mean, 2), round(aes_std, 2), round(aes_max, 2), round(pct_diff(aes_mean, base_aes_mean), 2),
            round(cli_mean, 4), round(cli_std, 4), round(cli_max, 4), round(pct_diff(cli_mean, base_cli_mean), 4),
            round(comb_mean, 4), round(comb_std, 4), round(comb_max, 4), round(pct_diff(comb_mean, base_comb_mean), 4),
            round(time_mean, 2),
            folder_name_or_nan
        ]
        rows.append(r)

    # Emit blocks per (a,b)
    for a_int, b_int in weight_pairs:
        a_frac, b_frac = a_int/100.0, b_int/100.0
        base_comb_vals = a_frac*(aes0/aesthetic_norm) + b_frac*(cli0/clip_norm)
        base_aes_mean, _, _ = summarise(aes0)
        base_cli_mean, _, _ = summarise(cli0)
        base_comb_mean, _, _ = summarise(base_comb_vals)

        # Baseline row
        append_line(algo_labels[0][1], a_frac, b_frac, seed_val, aes0, cli0, time0, np.nan,
                    base_aes_mean, base_cli_mean, base_comb_mean)

        # Algorithms in fixed order if present
        for prefix, label in algo_labels[1:]:
            hit = next((r for r in runs if r["folder"].startswith(prefix)
                        and r["a"] == a_int and r["b"] == b_int), None)
            if not hit:
                continue
            append_line(label, a_frac, b_frac, hit["seed"], hit["aes_last"], hit["cli_last"],
                        hit["time_last"] , hit["folder"], base_aes_mean, base_cli_mean, base_comb_mean)

    # Write Excel with two-row header
    out_df = pd.DataFrame(rows)
    out_path = f"{save_folder}/{out_name}"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="summary", index=False, header=False)
    print(f"Saved to {out_path}")
