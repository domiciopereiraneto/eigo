import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_W_PAT    = re.compile(r"_a(\d+)_b(\d+)")

def _parse_weights(name: str):
    m = _W_PAT.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def _scan_aggregated_xlsx(root: Path):
    files = []
    for p in root.rglob('aggregated_score_results*.xlsx'):
        # skip our outputs if they accidentally match pattern
        if 'results_per_' in p.name:
            continue
        files.append(p)
    return files

def _is_evolutionary(folder_name: str) -> bool:
    low = folder_name.lower()
    # Adam is iterative. Everything else we consider evolutionary here.
    return not low.startswith('adam')

def _load_df_metrics(df: pd.DataFrame, evolutionary: bool, target_len: int = 101):
    '''Return dict with x, aes_mean, aes_std, clip_mean, clip_std, elapsed_mean, fitness_mean.

    Metrics are interpolated per column onto a common x grid, then averaged.
    '''
    out = {}

    def _interp_columns(cols: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if cols.empty:
            return np.array([], dtype=float), np.array([], dtype=float)
        xq = np.linspace(0, 1, target_len)
        series_list = []
        for c in cols.columns:
            vals = pd.to_numeric(cols[c], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) < 2:
                continue
            xp = np.linspace(0, 1, len(vals))
            series_list.append(np.interp(xq, xp, vals))
        if not series_list:
            return np.array([], dtype=float), np.array([], dtype=float)
        data = np.vstack(series_list)
        return data.mean(axis=0), data.std(axis=0, ddof=0)

    if evolutionary:
        aes = df.filter(like='max_aesthetic_score_')
        cli = df.filter(like='max_clip_score_')
        fit = df.filter(like='max_fitness_')
    else:
        aes = df.filter(like='aesthetic_score_')
        cli = df.filter(like='clip_score_')
        fit = df.filter(like='combined_score_')

    out['aes_mean'], out['aes_std'] = _interp_columns(aes)
    out['clip_mean'], out['clip_std'] = _interp_columns(cli)
    out['fitness_mean'], _ = _interp_columns(fit)

    tm = df.filter(like='elapsed_time_')
    out['elapsed_mean'], _ = _interp_columns(tm) if not tm.empty else (np.array([], dtype=float), np.array([], dtype=float))

    # x as percent
    n = target_len if len(out['aes_mean']) == target_len else len(out['fitness_mean'])
    out['x_percent'] = np.linspace(0, 100, n)
    return out

def _normalize_to_min_len(series_list: List[np.ndarray]) -> List[np.ndarray]:
    k = min(len(s) for s in series_list if s is not None and len(s) > 0)
    out = []
    for s in series_list:
        if s is None or len(s) == 0:
            out.append(None)
            continue
        if len(s) == k:
            out.append(s.astype(float))
        else:
            # linear interpolation onto [0, k-1]
            xp = np.linspace(0, 1, len(s))
            xq = np.linspace(0, 1, k)
            out.append(np.interp(xq, xp, s).astype(float))
    return out

def _plot_block(block: Dict[str, Dict], labels: Dict[str,str], save_dir: Path,
                a_int: int, b_int: int, aesthetic_max: float, clip_max: float):
    '''block: prefix -> {'evo':bool, 'metrics':dict}'''
    if not block:
        return []
    paths = []
    save_dir.mkdir(parents=True, exist_ok=True)

    # collect arrays and normalize length per metric
    prefixes = [p for p in labels.keys() if p in block]  # keep configured order
    if not prefixes:
        return []

    # build arrays per metric
    arrays = {
        'x': [block[p]['metrics']['x_percent'] for p in prefixes],
        'aes': [block[p]['metrics']['aes_mean'] for p in prefixes],
        'clip': [block[p]['metrics']['clip_mean'] for p in prefixes],
        'time': [block[p]['metrics']['elapsed_mean'] for p in prefixes],
        'fitness': [block[p]['metrics']['fitness_mean'] for p in prefixes],
    }
    x_norm, aes_norm, clip_norm, time_norm, fitness_norm = [_normalize_to_min_len(v) for v in arrays.values()]

    # common legend placement (below plot)
    leg = dict(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               ncol=min(4, len(prefixes)), frameon=False)

    # Aesthetic evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, p in enumerate(prefixes):
        if x_norm[i] is None:
            continue
        ax.plot(x_norm[i], aes_norm[i], '-', label=labels[p])
    ax.set_ylim(1, 10)
    ax.set_xlabel('Iteration (%)')
    ax.set_ylabel('Aesthetic score')
    ax.set_title(f'Evolution of Aesthetic Score (a={a_int}, b={b_int})')
    ax.grid(alpha=0.3)
    ax.legend(**leg)
    fig.tight_layout()
    out_aes = save_dir / f'aesthetic_evolution_a{a_int}_b{b_int}.png'
    fig.savefig(out_aes, dpi=300, bbox_inches='tight')
    plt.close(fig)
    paths.append(out_aes)

    # CLIP evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, p in enumerate(prefixes):
        if x_norm[i] is None:
            continue
        ax.plot(x_norm[i], clip_norm[i], '-', label=labels[p])
    ax.set_xlim(left=0)
    ax.set_ylim(0.5, 0.85)
    ax.set_xlabel('Iteration (%)')
    ax.set_ylabel('CLIP score')
    ax.set_title(f'Evolution of CLIP Score (a={a_int}, b={b_int})')
    ax.grid(alpha=0.3)
    ax.legend(**leg)
    fig.tight_layout()
    out_clip = save_dir / f'clip_evolution_a{a_int}_b{b_int}.png'
    fig.savefig(out_clip, dpi=300, bbox_inches='tight')
    plt.close(fig)
    paths.append(out_clip)

    # Elapsed time
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, p in enumerate(prefixes):
        if x_norm[i] is None:
            continue
        ax.plot(x_norm[i], time_norm[i], '-', label=labels[p])
    ax.set_xlabel('Iteration (%)')
    ax.set_ylabel('Elapsed time (s)')
    ax.set_title(f'Elapsed Time per Iteration (a={a_int}, b={b_int})')
    ax.grid(alpha=0.3)
    ax.legend(**leg)
    fig.tight_layout()
    out_time = save_dir / f'elapsed_time_a{a_int}_b{b_int}.png'
    fig.savefig(out_time, dpi=300, bbox_inches='tight')
    plt.close(fig)
    paths.append(out_time)

    # Fitness evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, p in enumerate(prefixes):
        if x_norm[i] is None:
            continue
        ax.plot(x_norm[i], fitness_norm[i], '-', label=labels[p])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Iteration (%)')
    ax.set_ylabel('Fitness')
    ax.set_title(f'Evolution of Fitness (a={a_int}, b={b_int})')
    ax.grid(alpha=0.3)
    ax.legend(**leg)
    fig.tight_layout()
    out_fitness = save_dir / f'fitness_evolution_a{a_int}_b{b_int}.png'
    fig.savefig(out_fitness, dpi=300, bbox_inches='tight')
    plt.close(fig)
    paths.append(out_fitness)

    return paths

def _plot_combined_metric(runs: List[Dict], out_dir: Path, metric_key: str,
                          title: str, ylabel: str, weight_count: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        metrics = r['metrics']
        x = metrics['x_percent']
        y = metrics.get(metric_key)
        if y is None or len(y) == 0:
            continue
        ax.plot(x, y, '-', label=r['label'])
    ax.set_xlabel('Iteration (%)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric_key == "fitness_mean":
        ax.set_xlim(left=0)
        ax.set_ylim(0.5, 0.85)
    ax.grid(alpha=0.3)
    ncol = max(1, min(weight_count, len(runs)))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=ncol, frameon=False)
    fig.tight_layout()
    out_path = out_dir / f'{metric_key}_all_weights.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path

def create_evolution_plots(source_dirs: List[str],
                           save_folder: str,
                           algo_labels: List[Tuple[str, str]],
                           aesthetic_max: float,
                           clip_max: float,
                           method_names: Optional[List[str]] = None) -> None:
    '''
    Generate evolution comparison plots per weight combination using the aggregated
    XLSX files produced by each run.

    Parameters
    ----------
    source_dirs : list of str
        Roots that contain run folders.
    save_folder : str
        Output directory for plots.
    algo_labels : list of (prefix, label)
        Same list used elsewhere. Baseline is ignored here if no XLSX exists.
    aesthetic_max : float
        Max value used to normalize aesthetic score to [0,1].
    clip_max : float
        Max value used to normalize CLIP score to [0,1].
    '''
    out_root = Path(save_folder) / 'evolution_plots'
    out_root.mkdir(parents=True, exist_ok=True)

    # map prefix -> label
    labels = {p:l for p,l in algo_labels}

    # gather all runs with aggregated XLSX
    runs = []  # dicts: {'a':int,'b':int,'prefix':str,'xlsx':Path,'evo':bool}
    for root in source_dirs:
        r = Path(root)
        if not r.exists():
            continue
        for xlsx in _scan_aggregated_xlsx(r):
            folder = xlsx.parent.name
            ab = _parse_weights(folder)
            if not ab:
                continue
            # match the configured prefixes
            pref = next((p for p in labels.keys() if folder.startswith(p)), None)
            if pref is None:
                continue
            runs.append({
                'a': ab[0], 'b': ab[1],
                'prefix': pref,
                'xlsx': xlsx,
                'evo': _is_evolutionary(folder)
            })

    if not runs:
        print('No aggregated_score_results*.xlsx files found for the configured prefixes.')
        return None

    # group by weight pair
    blocks: Dict[Tuple[int,int], Dict[str, Dict]] = {}
    for r in runs:
        key = (r['a'], r['b'])
        d = blocks.setdefault(key, {})
        # prefer first found if duplicates
        if r['prefix'] not in d:
            df = pd.read_excel(r['xlsx'])
            metrics = _load_df_metrics(df, evolutionary=r['evo'])
            d[r['prefix']] = {'evo': r['evo'], 'metrics': metrics}

    # make plots per weight pair
    for (a_int, b_int), block in sorted(blocks.items(), key=lambda x: (-x[0][0], x[0][1])):
        _plot_block(block, labels, out_root, a_int, b_int, aesthetic_max, clip_max)

    # Combined plots across all weights (one line per configured method/source_dir)
    if source_dirs:
        combined_runs = []
        weight_keys = set()
        algo_order = [p for p, _ in algo_labels]
        for idx, src in enumerate(source_dirs):
            root = Path(src)
            if not root.exists():
                continue
            xlsx_list = _scan_aggregated_xlsx(root)
            if not xlsx_list:
                continue
            xlsx = xlsx_list[0]
            folder = xlsx.parent.name
            ab = _parse_weights(folder)
            if not ab:
                continue
            weight_keys.add(ab)
            evo = _is_evolutionary(folder)
            df = pd.read_excel(xlsx)
            metrics = _load_df_metrics(df, evolutionary=evo)
            label = None
            if method_names:
                if len(method_names) == len(source_dirs) + 1:
                    label_idx = idx + 1
                else:
                    label_idx = idx
                if label_idx < len(method_names):
                    label = method_names[label_idx]
            if label is None:
                label = folder
            algo_rank = next((i for i, p in enumerate(algo_order) if folder.startswith(p)), len(algo_order))
            combined_runs.append({
                "label": label,
                "metrics": metrics,
                "weight": ab,
                "algo_rank": algo_rank,
            })

        if combined_runs:
            weight_count = len(weight_keys)
            def _w_key(w):
                return (-w[0], w[1])
            combined_runs.sort(key=lambda r: (r["algo_rank"], _w_key(r["weight"])))
            combined_dir = out_root / "all_weights"
            _plot_combined_metric(
                combined_runs,
                combined_dir,
                "aes_mean",
                "Aesthetic Score (All Weights)",
                "Aesthetic score",
                weight_count,
            )
            _plot_combined_metric(
                combined_runs,
                combined_dir,
                "clip_mean",
                "CLIP Score (All Weights)",
                "CLIP score",
                weight_count,
            )
            _plot_combined_metric(
                combined_runs,
                combined_dir,
                "fitness_mean",
                "Fitness (All Weights)",
                "Fitness",
                weight_count,
            )
            _plot_combined_metric(
                combined_runs,
                combined_dir,
                "elapsed_mean",
                "Elapsed Time (All Weights)",
                "Elapsed time (s)",
                weight_count,
            )
    print(f'Plots saved under: {out_root}')
    return None
