from pathlib import Path
import re
import numpy as np
import pandas as pd

def create_summary_table(results_dirs, save_folder, labels, aesthetic_norm, clip_norm):
    EXCEL_NAME    = "aggregated_score_results.xlsx"
    OUT_NAME      = "summary_results.xlsx" 

    build_summary(results_dirs, save_folder, EXCEL_NAME, OUT_NAME, aesthetic_norm, clip_norm, labels)

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

def load_first_row_arrays(xls_path: Path, is_adam: bool):
    """Return baseline per-prompt arrays (iteration 0) for aesthetic and clip."""
    df = pd.read_excel(xls_path)
    first = df.iloc[0]
    if is_adam:
        aes = pd.Series([float(first[c]) for c in df.columns if str(c).startswith("aesthetic_score_")])
        cli = pd.Series([float(first[c]) for c in df.columns if str(c).startswith("clip_score_")])
    else:
        aes = pd.Series([float(first[c]) for c in df.columns if str(c).startswith("max_aesthetic_score_")])
        cli = pd.Series([float(first[c]) for c in df.columns if str(c).startswith("max_clip_score_")])
    time = pd.Series([float(first[c]) for c in df.columns if str(c).startswith("elapsed_time_")])

    return aes.reset_index(drop=True), cli.reset_index(drop=True), time.reset_index(drop=True)

def last_row_arrays(xls_path: Path, is_adam: bool):
    """Return final per-prompt arrays for aesthetic and clip from the last row."""
    df = pd.read_excel(xls_path)
    last = df.iloc[-1]
    if is_adam:
        aes = pd.Series([float(last[c]) for c in df.columns if str(c).startswith("aesthetic_score_")])
        cli = pd.Series([float(last[c]) for c in df.columns if str(c).startswith("clip_score_")])
    else:
        aes = pd.Series([float(last[c]) for c in df.columns if str(c).startswith("max_aesthetic_score_")])
        cli = pd.Series([float(last[c]) for c in df.columns if str(c).startswith("max_clip_score_")])
    time = pd.Series([float(last[c]) for c in df.columns if str(c).startswith("elapsed_time_")])

    return aes.reset_index(drop=True), cli.reset_index(drop=True), time.reset_index(drop=True)

def build_summary(results_dir, save_folder, excel_name, out_name, aesthetic_norm, clip_norm, algo_labels):
    # Scan runs
    runs = []
    baseline_source = None

    for folder in results_dir:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        # Find the Excel file
        matches = list(folder.rglob(excel_name))
        if not matches:
            continue
        xls = matches[0]
        name = folder.name
        is_adam = name.startswith("adam")

        # capture baseline source once from the first found run
        if baseline_source is None:
            baseline_source = load_first_row_arrays(xls, is_adam=is_adam)

        a_b = parse_weights(name)
        if not a_b:
            continue
        a, b = a_b
        seed = parse_seed(name)
        aes_last, cli_last, time_last = last_row_arrays(xls, is_adam=is_adam)
        runs.append({
            "folder": name,
            "seed": seed,
            "a": a, "b": b,
            "aes_last": aes_last,
            "cli_last": cli_last,
            "time_last": time_last
        })

    if not runs:
        raise FileNotFoundError(f"No runs with '{excel_name}' found under the given directories.")

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