"""Run Cartesian hyperparameter sweeps for EIGO optimizers.

The grid config defines one or more prompts, enabled optimizers, and per-method
parameter lists. Every method-specific cartesian product is executed and recorded
with its effective parameters and final metrics.
"""
import argparse
import csv
import copy
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import yaml
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
DEFAULT_CONFIG = 'algorithms/config/config_eigo_grid_search.yaml'

def load_yaml(path: Path):
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError('Configuration YAML must load to a dictionary.')
    return data

def _coerce_scalar(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == '' or stripped.lower() == 'nan':
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return value
        if numeric.is_integer():
            return int(numeric)
        return numeric
    return value

def extract_final_results(produced_folder, method):
    produced_path = Path(produced_folder)
    if method in {'gomea', 'ga', 'random_sampler'}:
        csv_path = produced_path / 'fitness_results.csv'
        metric_keys = ['best_fitness', 'evaluations', 'generation', 'elapsed_time', 'peak_vram_mb', 'avg_fitness', 'max_fitness', 'avg_aesthetic_score', 'max_aesthetic_score', 'avg_clip_score', 'max_clip_score', 'avg_image_reward_score', 'max_image_reward_score', 'avg_hpsv2_score', 'max_hpsv2_score', 'avg_pickscore_score', 'max_pickscore_score', 'avg_jpeg_size_kb', 'min_jpeg_size_kb']
    else:
        return None
    if not csv_path.exists():
        return {'status': 'missing_csv', 'csv_path': str(csv_path)}
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {'status': 'empty_csv', 'csv_path': str(csv_path)}
    final_row = rows[-1]
    final_results = {'status': 'ok', 'csv_path': str(csv_path)}
    for key in metric_keys:
        if key in final_row:
            final_results[key] = _coerce_scalar(final_row[key])
    return final_results

def normalize_grid(method_grid):
    """Ensure each grid entry is represented as key -> list of candidate values."""
    if method_grid is None:
        return {}
    if not isinstance(method_grid, dict):
        raise ValueError('Each method grid must be a dictionary of parameter -> list/value.')
    normalized = {}
    for key, value in method_grid.items():
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Grid parameter '{key}' has an empty list.")
            normalized[key] = value
        else:
            normalized[key] = [value]
    return normalized

def expand_grid(grid_dict):
    """Create cartesian-product combinations from a dictionary of lists."""
    if not grid_dict:
        return [{}]
    keys = list(grid_dict.keys())
    value_lists = [grid_dict[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(keys, values)))
    return combos

def get_prompt_list(config):
    if 'selected_prompts' in config:
        prompts = config['selected_prompts']
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("'selected_prompts' must be a non-empty list of prompt strings.")
        if not all((isinstance(prompt, str) and prompt.strip() for prompt in prompts)):
            raise ValueError("'selected_prompts' must only contain non-empty strings.")
        return prompts
    if 'selected_prompt' in config:
        prompt = config['selected_prompt']
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("'selected_prompt' must be a non-empty string.")
        return [prompt]
    raise ValueError("Missing required key 'selected_prompts' or 'selected_prompt' in config.")

def validate_base_config(config):
    get_prompt_list(config)
    if 'results_folder' not in config:
        raise ValueError("Missing required key 'results_folder' in config.")

def build_run_queue(config):
    grid = config.get('grid') or {}
    if not isinstance(grid, dict) or set(grid) - {'ga', 'gomea', 'random_sampler'}:
        raise ValueError('grid must contain only ga, gomea, and random_sampler sections')
    queue = [(method, combo) for method in ('ga', 'gomea', 'random_sampler') if config.get('test_' + method, method == 'ga') for combo in expand_grid(normalize_grid(grid.get(method, {})))]
    if not queue:
        raise ValueError('Enable test_ga, test_gomea, or test_random_sampler')
    return queue

def create_run_config(base_config, method, overrides, run_id, prompt):
    run_config = copy.deepcopy(base_config)
    run_config['optimization_method'] = method
    run_config['selected_prompt'] = prompt
    run_config.update(overrides)
    base_results_folder = str(base_config.get('results_folder', 'results'))
    run_config['results_folder'] = os.path.join(base_results_folder, run_id)
    return run_config

def save_run_parameters(results_folder, run_id, method, overrides, run_config):
    """Persist the exact parameter combination used for a run."""
    output_dir = Path(results_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'run_id': run_id, 'method': method, 'selected_prompt': run_config.get('selected_prompt'), 'overrides': overrides, 'effective_parameters': {key: run_config.get(key) for key in sorted(overrides.keys())}}
    params_path = output_dir / 'grid_search_parameters.yaml'
    with params_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return params_path

def run_grid_search(config, dry_run=False):
    validate_base_config(config)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_list = get_prompt_list(config)
    run_queue = build_run_queue(config)
    summary_rows = []
    print(f'Prompt count: {len(prompt_list)}')
    print(f'Total runs: {len(prompt_list) * len(run_queue)}')
    total_runs = len(prompt_list) * len(run_queue)
    run_index = 0
    for prompt_idx, prompt in enumerate(prompt_list, start=1):
        for method, overrides in run_queue:
            run_index += 1
            run_id = f'run_{run_index:03d}_prompt_{prompt_idx:03d}_{method}_{timestamp}'
            run_config = create_run_config(config, method, overrides, run_id, prompt)
            print('-' * 80)
            print(f'Run {run_index}/{total_runs}')
            print(f'Prompt {prompt_idx}/{len(prompt_list)}: {prompt}')
            print(f'Method: {method}')
            print(f'Overrides: {json.dumps(overrides, ensure_ascii=True)}')
            print(f"Results folder: {run_config['results_folder']}")
            run_params_path = save_run_parameters(run_config['results_folder'], run_id, method, overrides, run_config)
            print(f'Run parameters saved to: {run_params_path}')
            row = {'run_index': run_index, 'run_id': run_id, 'prompt_index': prompt_idx, 'selected_prompt': prompt, 'method': method, 'overrides': overrides, 'results_folder': run_config['results_folder'], 'status': 'scheduled'}
            if dry_run:
                row['status'] = 'dry_run'
                summary_rows.append(row)
                continue
            try:
                from eigo import Eigo
                eigo_engine = Eigo(run_config)
                if method == 'ga':
                    produced_folder = eigo_engine.run_ga_optimization()
                else:
                    if method == 'gomea':
                        produced_folder = eigo_engine.run_gomea_optimization()
                    elif method == 'random_sampler':
                        produced_folder = eigo_engine.run_random_sampler_optimization()
                    else:
                        raise ValueError(f'Unsupported method: {method}')
                row['status'] = 'success'
                row['produced_folder'] = produced_folder
                row['final_results'] = extract_final_results(produced_folder, method)
                save_run_parameters(produced_folder, run_id, method, overrides, run_config)
            except Exception as exc:
                row['status'] = 'failed'
                row['error'] = str(exc)
                summary_rows.append(row)
                print(f'Run failed: {exc}')
                continue
            summary_rows.append(row)
    return summary_rows

def save_summary(config, summary_rows):
    output_dir = Path(str(config.get('results_folder', 'results')))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'grid_search_summary.yaml'
    payload = {'selected_prompts': get_prompt_list(config), 'test_ga': bool(config.get('test_ga', False)), 'test_gomea': bool(config.get('test_gomea', False)), 'test_random_sampler': bool(config.get('test_random_sampler', False)), 'total_runs': len(summary_rows), 'runs': summary_rows}
    with summary_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f'Summary saved to: {summary_path}')

def build_parser():
    parser = argparse.ArgumentParser(description='Run grid search for EIGO with one or more prompts and method-specific parameter lists.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help='Path to the grid-search config YAML.')
    parser.add_argument('--dry-run', action='store_true', help='Print planned runs without executing optimization.')
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_yaml(Path(args.config).resolve())
    summary_rows = run_grid_search(config, dry_run=args.dry_run)
    save_summary(config, summary_rows)
if __name__ == '__main__':
    main()
