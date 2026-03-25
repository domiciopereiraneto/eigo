
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Heavy deps are imported lazily inside functions
#   torch, clip, cv2, skimage are only loaded when computing distances


# ------------------------- parsing helpers -------------------------

_W_PAT    = re.compile(r"_a(\d+)_b(\d+)")
_SEED_PAT = re.compile(r"_(\d+)_a(\d+)_b(\d+)")
_PROMPT_DIR_PAT = re.compile(r"^results_.*_(\d+)$")

def parse_weights(name: str) -> Tuple[int, int] | None:
    m = _W_PAT.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def parse_seed(name: str) -> int | None:
    m = _SEED_PAT.search(name)
    if not m:
        return None
    return int(m.group(1))

def find_prompt_dirs(run_dir: Path) -> List[Path]:
    out = []
    for p in run_dir.iterdir():
        if p.is_dir() and _PROMPT_DIR_PAT.match(p.name):
            out.append(p)
    # natural sort by numeric id at end
    out.sort(key=lambda p: int(_PROMPT_DIR_PAT.match(p.name).group(1)))
    return out

def summarise(vals: Iterable[float]) -> Tuple[float, float, float]:
    v = pd.Series(list(vals), dtype=float)
    return float(v.mean()), float(v.std(ddof=0)), float(v.max())


# ----------------------- distance computation ----------------------

def _load_clip(device: str = None):
    import torch
    import clip
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def _img_to_vec(img_path: Path, model, preprocess, device: str):
    from PIL import Image
    import torch
    with Image.open(img_path).convert("RGB") as im:
        tensor = preprocess(im).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(tensor)
    return vec.cpu().numpy().flatten()

def _load_gray(img_path: Path, max_side: int = 256):
    import cv2
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    h, w = im.shape
    scale = max_side / max(h, w)
    if scale < 1.0:
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return im

def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    num = float(np.dot(u, v))
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    if den == 0.0:
        return np.nan
    return float(num / den)

def _compute_distances_for_run(run_dir, model, preprocess, device):
    """Return per‑prompt (cosine_similarity, ssim) lists for a single algorithm run folder."""
    from skimage.metrics import structural_similarity as ssim

    cos_vals, ssim_vals = [], []

    for pdir in find_prompt_dirs(run_dir):
        base = pdir / "it_0.png"
        best = pdir / "best_all.png"
        if not base.exists() or not best.exists():
            continue
        # CLIP cosine similarity
        v_base = _img_to_vec(base, model, preprocess, device)
        v_best = _img_to_vec(best, model, preprocess, device)
        cos_vals.append(_cosine_similarity(v_best, v_base))
        # SSIM
        g_base = _load_gray(base)
        g_best = _load_gray(best)
        try:
            ssim_vals.append(float(ssim(g_best, g_base, data_range=255)))
        except Exception:
            # fallback if shapes differ beyond resize or other issues
            mn = min(g_best.shape[0], g_base.shape[0])
            nn = min(g_best.shape[1], g_base.shape[1])
            ssim_vals.append(float(ssim(g_best[:mn, :nn], g_base[:mn, :nn], data_range=255)))
    return cos_vals, ssim_vals

def _load_precomputed_distances_for_run(run_dir: Path):
    csv_path = run_dir / "aggregate_prompt_similarity_values.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    if "cosine_similarity" not in df.columns or "ssim" not in df.columns:
        return None

    cos_sim = pd.to_numeric(df["cosine_similarity"], errors="coerce").to_numpy(dtype=float)
    ssim_vals = pd.to_numeric(df["ssim"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(cos_sim) & np.isfinite(ssim_vals)
    if valid.sum() == 0:
        return None

    return cos_sim[valid].tolist(), ssim_vals[valid].tolist()


# ------------------------- public API ------------------------------

def create_distance_table_and_plots(
    results_dirs: List[str],
    save_folder: str,
    algo_labels: List[Tuple[str, str]],
) -> None:
    """
    Build an Excel like 'summary_results.xlsx' but for CLIP‑cosine similarity to baseline
    and SSIM to baseline. Also emit two grouped box plots across weight pairs.
    
    Parameters
    ----------
    results_dirs : list of str
        Roots to search recursively for algorithm runs. Each run folder must include
        weights in its name: *_aXX_bYY* and prompt subfolders *results_*_<ID>/ with
        it_0.png and best_all.png.
    save_folder : str
        Output directory.
    algo_labels : list of (prefix, label)
        Same format used by summary_table.py. The first tuple is assumed to be the baseline
        label; it will appear in the table as identity similarity (cos=1, ssim=1).
    """
    out_dir = Path(save_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect run folders
    runs: List[Dict] = []
    for root in results_dirs:
        root = Path(root)
        if not root.exists():
            continue

        ab = parse_weights(root.name)
        if not ab:
            continue
        a, b = ab

        name = root.name

        seed = parse_seed(name)

        runs.append({"path": root, "name": name, "a": ab[0], "b": ab[1], "seed": seed})

    if not runs:
        raise FileNotFoundError("No run folders matching pattern *_aXX_bYY found under provided results_dirs.")

    # Map algorithm by prefix to label
    prefix_to_label = {p: lab for p, lab in algo_labels}
    prefixes = list(prefix_to_label.keys())

    # Organize runs by (a,b) then by algorithm prefix
    grid: Dict[Tuple[int,int], Dict[str, Dict]] = {}
    for r in runs:
        for pref in prefixes:
            if r["name"].startswith(pref):
                grid.setdefault((r["a"], r["b"]), {})[pref] = r
                break

    # Prepare Excel table rows
    hdr_top = [
        "", "", "", "",
        "Cosine similarity to baseline [-1, 1]", "", "",
        "SSIM to baseline [-1, 1]", "", "",
        ""
    ]
    hdr_sub = [
        "Algorithm", "a", "b", "Seed",
        "Avg.", "Std.", "Max.",
        "Avg.", "Std.", "Max.",
        "Folder"
    ]
    rows = [hdr_top, hdr_sub]

    # DataFrame for plotting distributions
    plot_rows = []  # dicts with a,b,algorithm,cosine_similarity,ssim

    # Sorted weight pairs like summary_table
    weight_pairs = sorted(grid.keys(), key=lambda x: (-x[0], x[1]))

    clip_bundle = None
    for a_int, b_int in weight_pairs:
        # Baseline row (cos=1, ssim=1)
        rows.append([
            algo_labels[0][1], a_int/100.0, b_int/100.0, "",
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            np.nan
        ])

        for pref, label in algo_labels[1:]:
            run = grid.get((a_int, b_int), {}).get(pref)
            if not run:
                continue
            precomp = _load_precomputed_distances_for_run(run["path"])
            if precomp is not None:
                cos_vals, ssim_vals = precomp
            else:
                if clip_bundle is None:
                    clip_bundle = _load_clip()
                model, preprocess, device = clip_bundle
                cos_vals, ssim_vals = _compute_distances_for_run(run["path"], model, preprocess, device)
            if not cos_vals or not ssim_vals:
                continue
            cos_mean, cos_std, cos_max = summarise(cos_vals)
            ssim_mean, ssim_std, ssim_max = summarise(ssim_vals)

            rows.append([
                label, a_int/100.0, b_int/100.0, run["seed"] if run["seed"] is not None else "",
                round(cos_mean, 6), round(cos_std, 6), round(cos_max, 6),
                round(ssim_mean, 6), round(ssim_std, 6), round(ssim_max, 6),
                run["name"]
            ])
            for cv, sv in zip(cos_vals, ssim_vals):
                plot_rows.append({
                    "a": a_int/100.0, "b": b_int/100.0,
                    "algorithm": label,
                    "cosine_similarity": float(cv),
                    "ssim": float(sv)
                })

    # Write Excel
    out_xlsx = out_dir / "distance_summary.xlsx"
    df_out = pd.DataFrame(rows)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="distance_summary", index=False, header=False)

    # Create grouped box plots across weight pairs
    if plot_rows:
        plot_df = pd.DataFrame(plot_rows)
        _grouped_boxplots(plot_df, out_dir, value_col="cosine_similarity",
                      title="Cosine Similarity to Baseline by Weighting and Algorithm",
                      ylabel="Cosine Similarity [-1, 1]")
        _grouped_boxplots(plot_df, out_dir, value_col="ssim",
                      title="SSIM to Baseline by Weighting and Algorithm",
                      ylabel="SSIM [-1, 1]")

    print(f"Saved: {out_xlsx}")


def _grouped_boxplots(df: pd.DataFrame, out_dir: Path, value_col: str, title: str, ylabel: str):
    """One plot. Groups=weight pairs, boxes=algorithms. Saves PNG next to Excel."""
    # Build x labels as 'a=0.5, b=0.5'
    df = df.copy()
    df["group"] = df.apply(lambda r: f"a={r['a']:.1f}, b={r['b']:.1f}", axis=1)
    groups = sorted(df["group"].unique().tolist(),
                    key=lambda s: (-float(s.split(",")[0].split("=")[1]), float(s.split(",")[1].split("=")[1])))
    algos = sorted(df["algorithm"].unique().tolist())

    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(algos))

    cmap = plt.get_cmap("tab10")
    for j, algo in enumerate(algos):
        data = []
        positions = []
        for g in groups:
            vals = pd.to_numeric(
                df[(df["group"] == g) & (df["algorithm"] == algo)][value_col],
                errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            data.append(vals)
            positions.append(float(x[groups.index(g)] + j*width))
        if not data:
            continue
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=True
        )
        color = cmap(j % 10)
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.35, edgecolor=color)
        for item in bp["whiskers"] + bp["caps"] + bp["medians"]:
            item.set(color=color)
        ax.plot([], [], color=color, linewidth=8, alpha=0.35, label=algo)

    ax.set_title(title)
    ax.set_xlabel("Weight Combination")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + (len(algos)-1)*width/2)
    ax.set_xticklabels(groups)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    # place legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout()
    # filename by metric
    metric = "cosine" if "cos" in value_col else "ssim"
    out_path = out_dir / f"{metric}_boxplot_by_weight.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
