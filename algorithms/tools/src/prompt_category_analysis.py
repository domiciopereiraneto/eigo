import re
from pathlib import Path
import pandas as pd

def create_prompt_category_results_comparison(source_dirs, save_path):
    out_dir = Path(save_path); out_dir.mkdir(parents=True, exist_ok=True)
    prompts_df = read_selected_prompts(Path(source_dirs[0]) / "selected_prompts.txt")
    entries = []

    for source_dir in source_dirs:
        root = Path(source_dir)
        for xlsx in scan_results(root):
            df = pd.read_excel(xlsx)
            schema = detect_schema(df.columns, xlsx.parent.name)
            approach, a_scaled, b_scaled = build_label(xlsx.parent)
            try:
                folder_rel = str(xlsx.parent.relative_to(root))
            except ValueError:
                folder_rel = xlsx.parent.name

            if schema == "adam":
                entries.extend(extract_entries_from_adam(df, approach, a_scaled, b_scaled, folder_rel))
            elif schema == "hybrid":
                entries.extend(extract_entries_from_hybrid(df, approach, a_scaled, b_scaled, folder_rel))
            else:
                continue

    if not entries:
        raise SystemExit("No aggregated_score_results*.xlsx files parsed under the provided source_dirs.")

    entries_df = pd.DataFrame(entries)
    value_cols = [c for c in ["aesthetic_score", "clip_score", "fitness", "elapsed_time", "baseline_aesthetic_score", "baseline_clip_score", "baseline_fitness"] if c in entries_df.columns]

    key_cols = ["approach", "a", "b", "prompt_idx"]
    per_prompt_stats = consolidate_by_seed(entries_df, key_cols, value_cols)

    # compute diff-to-baseline (%) at per-prompt level
    def _add_diffs(df):
        def _pct(v, b):
            try:
                if b is None or pd.isna(b) or b == 0:
                    return None
                return 100.0 * (v - b) / abs(b)
            except Exception:
                return None
        cols = []
        if "aesthetic_score_mean" in df.columns and "baseline_aesthetic_score_mean" in df.columns:
            df["aesthetic_diff_to_baseline_pct"] = [_pct(v, b) for v, b in zip(df["aesthetic_score_mean"], df["baseline_aesthetic_score_mean"])]
            cols.append("aesthetic_diff_to_baseline_pct")
        if "clip_score_mean" in df.columns and "baseline_clip_score_mean" in df.columns:
            df["clip_diff_to_baseline_pct"] = [_pct(v, b) for v, b in zip(df["clip_score_mean"], df["baseline_clip_score_mean"])]
            cols.append("clip_diff_to_baseline_pct")
        if "fitness_mean" in df.columns and "baseline_fitness_mean" in df.columns:
            df["fitness_diff_to_baseline_pct"] = [_pct(v, b) for v, b in zip(df["fitness_mean"], df["baseline_fitness_mean"])]
            cols.append("fitness_diff_to_baseline_pct")
        return df, cols

    per_prompt_stats, diff_cols = _add_diffs(per_prompt_stats)

    res_per_prompt = build_results_per_prompt(per_prompt_stats, prompts_df)
    res_per_cat    = build_results_per_category(res_per_prompt)

    rp_path = Path(save_path) / "results_per_prompt.xlsx"
    rc_path = Path(save_path) / "results_per_category.xlsx"
    res_per_prompt.to_excel(rp_path, index=False)
    res_per_cat.to_excel(rc_path, index=False)
    print(f"Saved: {rp_path}")
    print(f"Saved: {rc_path}")

    # plots
    plots_dir = Path(save_path) / "plots_by_weight"
    # Only category plots per request
    cplots = _plot_per_category_vertical(res_per_cat, plots_dir)
    for pth in cplots:
        print(f"Saved: {pth}")
    return None

def read_selected_prompts(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                cat, prompt = line.split("\t", 1)
            elif "," in line:
                cat, prompt = line.split(",", 1)
            else:
                cat, prompt = "Unknown", line
            rows.append({"prompt_idx": i, "category": cat.strip(), "prompt": prompt.strip()})
    return pd.DataFrame(rows)

def detect_schema(cols, folder_name: str):
    # Prefer column-based detection
    for c in cols:
        if re.match(r"combined_score_\d+_\d+$", str(c)):
            return "adam"
    for c in cols:
        if re.match(r"avg_fitness_\d+_\d+$", str(c)):
            return "hybrid"
    # Fallback: infer from folder pattern
    low = folder_name.lower()
    if any(k in low for k in ["hybrid", "hybridsep", "sepcmaes", "sep-cma-es", "sep_cma_es"]):
        return "hybrid"
    if "adam" in low:
        return "adam"
    return "unknown"

def parse_adam_steps(text: str):
    m = re.search(r"(\d+)adamsteps", text.lower())
    return int(m.group(1)) if m else None

def parse_weights_numeric(text: str):
    t = text.lower()
    ma = re.search(r'(?:^|[_-])a(\d+)(?:_|$)', t)
    mb = re.search(r'(?:^|[_-])b(\d+)(?:_|$)', t)
    a_scaled = float(ma.group(1))/100.0 if ma else None
    b_scaled = float(mb.group(1))/100.0 if mb else None
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
    if steps is not None:
        approach = f"{base} ({steps} adam steps)"
    else:
        approach = base
    return approach, a_scaled, b_scaled

# ---------- extractors ----------

def extract_entries_from_adam(df: pd.DataFrame, approach: str, a_scaled, b_scaled, folder_rel: str):
    lr = df.iloc[-1]
    fr = df.iloc[0]
    pat = re.compile(r"(combined_score|aesthetic_score|clip_score|elapsed_time)_(\d+)_(\d+)$")
    records = {}
    for col in df.columns:
        m = pat.match(str(col))
        if not m:
            continue
        metric, seed, idx = m.group(1), int(m.group(2)), int(m.group(3))
        key = (seed, idx)
        if key not in records:
            records[key] = {
                "seed": seed, "prompt_idx": idx,
                "approach": approach, 
                "a": a_scaled, "b": b_scaled, "folder": folder_rel
            }
        records[key][metric] = float(lr[col])
        # baseline from first row
        base_key = f'baseline_{metric}' if metric != 'combined_score' else 'baseline_fitness'
        if metric == 'combined_score':
            records[key]['baseline_fitness'] = float(fr[col])
        else:
            records[key][base_key] = float(fr[col])
    out = []
    for v in records.values():
        out.append({
            "seed": v["seed"],
            "prompt_idx": v["prompt_idx"],
            "approach": v["approach"],
            "a": v["a"],
            "b": v["b"],
            "folder": v["folder"],
            "aesthetic_score": v.get("aesthetic_score", None),
            "clip_score": v.get("clip_score", None),
            "fitness": v.get("combined_score", None),
            "elapsed_time": v.get("elapsed_time", None),
            "baseline_aesthetic_score": v.get("baseline_aesthetic_score", None),
            "baseline_clip_score": v.get("baseline_clip_score", None),
            "baseline_fitness": v.get("baseline_fitness", None),
        })
    return out

def extract_entries_from_hybrid(df: pd.DataFrame, approach: str, a_scaled, b_scaled, folder_rel: str):
    lr = df.iloc[-1]
    fr = df.iloc[0]
    pat = re.compile(r"(avg|std|max)_(fitness|aesthetic_score|clip_score)_(\d+)_(\d+)$")
    time_pat = re.compile(r"elapsed_time_(\d+)_(\d+)$")
    records = {}
    for col in df.columns:
        m = pat.match(str(col))
        if m:
            agg, metric, seed, idx = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            key = (seed, idx)
            rec = records.setdefault(key, {
                "seed": seed, "prompt_idx": idx,
                "approach": approach, 
                "a": a_scaled, "b": b_scaled, "folder": folder_rel
            })
            rec[f"{agg}_{metric}"] = float(lr[col])
            if agg == 'avg':
                rec[f'baseline_{metric}'] = float(fr[col])
            continue
        tm = time_pat.match(str(col))
        if tm:
            seed, idx = int(tm.group(1)), int(tm.group(2))
            key = (seed, idx)
            rec = records.setdefault(key, {
                "seed": seed, "prompt_idx": idx,
                "approach": approach, 
                "a": a_scaled, "b": b_scaled, "folder": folder_rel
            })
            rec["elapsed_time"] = float(lr[col])
    out = []
    for rec in records.values():
        out.append({
            "seed": rec["seed"],
            "prompt_idx": rec["prompt_idx"],
            "approach": rec["approach"],
            
            "a": rec["a"],
            "b": rec["b"],
            "folder": rec["folder"],
            "fitness": rec.get("avg_fitness", None),
            "aesthetic_score": rec.get("avg_aesthetic_score", None),
            "clip_score": rec.get("avg_clip_score", None),
            "elapsed_time": rec.get("elapsed_time", None),
            "baseline_aesthetic_score": rec.get("baseline_aesthetic_score", None),
            "baseline_clip_score": rec.get("baseline_clip_score", None),
            "baseline_fitness": rec.get("baseline_fitness", None),
        })
    return out

# ---------- aggregation ----------

def consolidate_by_seed(entries_df: pd.DataFrame, key_cols, value_cols):
    try:
        grouped = entries_df.groupby(key_cols, as_index=False, dropna=False)
    except TypeError:
        grouped = entries_df.groupby(key_cols, as_index=False)
    agg = {}
    for c in value_cols:
        agg[c + "_mean"] = (c, "mean")
        agg[c + "_std"]  = (c, lambda s: s.std(ddof=0))
    out = grouped.agg(**agg)
    try:
        folder_map = entries_df.groupby(key_cols, dropna=False)["folder"].agg(lambda s: s.iloc[0] if s.nunique()==1 else None).reset_index()["folder"]
    except TypeError:
        folder_map = entries_df.groupby(key_cols)["folder"].agg(lambda s: s.iloc[0] if s.nunique()==1 else None).reset_index()["folder"]
    out["folder"] = folder_map.values
    return out

def build_results_per_prompt(entries_df: pd.DataFrame, prompts_df: pd.DataFrame):
    merged = entries_df.merge(prompts_df, on="prompt_idx", how="left")
    cols = ["prompt", "category", "approach", "a", "b",
            "aesthetic_score_mean", "aesthetic_score_std",
            "clip_score_mean", "clip_score_std",
            "fitness_mean", "fitness_std", "aesthetic_diff_to_baseline_pct", "clip_diff_to_baseline_pct", "fitness_diff_to_baseline_pct",
            "elapsed_time_mean", "elapsed_time_std",
            "folder"]
    cols = [c for c in cols if c in merged.columns]
    merged = merged[cols]
    merged = merged.sort_values(["category", "prompt", "approach", "a", "b"]).reset_index(drop=True)
    return merged

def build_results_per_category(per_prompt_df: pd.DataFrame):
    value_cols = [c for c in per_prompt_df.columns if c.endswith("_mean")] + [c for c in per_prompt_df.columns if c.endswith("_diff_to_baseline_pct")]
    try:
        grouped = per_prompt_df.groupby(["category", "approach", "a", "b"], dropna=False)
    except TypeError:
        grouped = per_prompt_df.groupby(["category", "approach", "a", "b"])
    out = grouped.agg({c: "mean" for c in value_cols})
    for c in value_cols:
        stdname = c.replace("_mean", "_std_over_prompts")
        out[stdname] = grouped[c].std(ddof=0)
    out = out.reset_index().sort_values(["category", "approach", "a", "b"]).reset_index(drop=True)
    return out

# ---------- scanning and run ----------

# ---------- plotting ----------

def _select_prompt_metrics(df):
    cols = []
    if "clip_diff_to_baseline_pct" in df.columns:
        cols.append(("clip_diff_to_baseline_pct", "CLIP Δ to baseline (%)"))
    if "fitness_diff_to_baseline_pct" in df.columns:
        cols.append(("fitness_diff_to_baseline_pct", "Fitness Δ to baseline (%)"))
    if not cols:
        if "clip_score_mean" in df.columns:
            cols.append(("clip_score_mean","CLIP mean"))
        if "fitness_mean" in df.columns:
            cols.append(("fitness_mean","Fitness mean"))
    return cols

def _select_category_metrics(df):
    cols = []
    if 'aesthetic_score_mean' in df.columns:
        cols.append(('aesthetic_score_mean', 'Aesthetic score (mean)'))
    if 'clip_score_mean' in df.columns:
        cols.append(('clip_score_mean', 'CLIP score (mean)'))
    if 'fitness_mean' in df.columns:
        cols.append(('fitness_mean', 'Fitness (mean)'))
    return cols

def _plot_per_prompt_horizontal(df, out_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _select_prompt_metrics(df)
    approach_order = sorted(df["approach"].dropna().unique().tolist())
    weights = df[["a","b"]].drop_duplicates().sort_values(["a","b"]).itertuples(index=False, name=None)
    paths = []
    for a_val, b_val in weights:
        sub = df[(df["a"] == a_val) & (df["b"] == b_val)].copy()
        prompts = sorted(sub["prompt"].unique().tolist(), key=lambda x: str(x).lower())
        approaches = [ap for ap in approach_order if ap in sub["approach"].unique()]
        H = max(6, 0.35 * len(prompts))
        for col, label in metrics:
            fig, ax = plt.subplots(figsize=(14, H))
            y = np.arange(len(prompts))
            bar_h = 0.8 / max(1, len(approaches))
            for j, ap in enumerate(approaches):
                vals = []
                for pr in prompts:
                    row = sub[(sub["prompt"] == pr) & (sub["approach"] == ap)]
                    vals.append(float(row.iloc[0][col]) if not row.empty else np.nan)
                vals = np.array(vals, dtype=float)
                ax.barh(y + j*bar_h, vals, height=bar_h, label=ap)
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        ax.text(v, y[i] + j*bar_h, f"{v:.2f}", va="center", ha="left", fontsize=7)
            ax.set_title(f"Per-prompt comparison — a={a_val:.1f}, b={b_val:.1f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Prompt")
            ax.set_yticks(y + (len(approaches)-1)*bar_h/2)
            ax.set_yticklabels(prompts)
            ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            fig.tight_layout()
            outp = out_dir / f"per_prompt_horizontal_{col}_a{a_val:.1f}_b{b_val:.1f}.png"
            fig.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(outp)
    return paths

def _plot_per_category_vertical(df, out_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _select_category_metrics(df)
    approach_order = sorted(df["approach"].dropna().unique().tolist())
    weights = df[["a","b"]].drop_duplicates().sort_values(["a","b"]).itertuples(index=False, name=None)
    paths = []
    for a_val, b_val in weights:
        sub = df[(df["a"] == a_val) & (df["b"] == b_val)].copy()
        cats = sorted(sub["category"].unique().tolist(), key=lambda x: str(x).lower())
        approaches = [ap for ap in approach_order if ap in sub["approach"].unique()]
        W = max(10, 0.8 * len(cats))
        for col, label in metrics:
            fig, ax = plt.subplots(figsize=(W+3, 6))
            x = np.arange(len(cats))
            width = 0.8 / max(1, len(approaches))
            for j, ap in enumerate(approaches):
                vals = []
                for c in cats:
                    row = sub[(sub["category"] == c) & (sub["approach"] == ap)]
                    vals.append(float(row.iloc[0][col]) if not row.empty else np.nan)
                vals = np.array(vals, dtype=float)
                ax.bar(x + j*width, vals, width, label=ap)
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        ax.text(x[i] + j*width, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            ax.set_title(f"{label} comparison per category — a={a_val:.1f}, b={b_val:.1f}")
            ax.set_xlabel("Category")
            ax.set_ylabel(label)
            ax.set_xticks(x + (len(approaches)-1)*width/2)
            ax.set_xticklabels(cats, rotation=90)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            fig.tight_layout()
            outp = out_dir / f"per_category_{col}_a{a_val:.1f}_b{b_val:.1f}.png"
            fig.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(outp)
    return paths

def scan_results(root: Path):
    files = []
    for p in root.rglob("aggregated_score_results*.xlsx"):
        # skip our outputs if they accidentally match pattern
        if "results_per_" in p.name:
            continue
        files.append(p)
    return files