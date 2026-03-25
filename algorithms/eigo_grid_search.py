#!/usr/bin/env python3
"""Run EIGO grid search for a single prompt across Adam and/or CMA-ES parameter sets."""

import argparse
import copy
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add parent directory to Python path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)


DEFAULT_CONFIG = "algorithms/config/config_eigo_grid_search.yaml"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Configuration YAML must load to a dictionary.")
    return data


def normalize_grid(method_grid):
    """Ensure each grid entry is represented as key -> list of candidate values."""
    if method_grid is None:
        return {}
    if not isinstance(method_grid, dict):
        raise ValueError("Each method grid must be a dictionary of parameter -> list/value.")

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


def validate_base_config(config):
    required_keys = ["selected_prompt", "results_folder"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config.")


def build_run_queue(config):
    test_adam = bool(config.get("test_adam", True))
    test_cmaes = bool(config.get("test_cmaes", True))

    if not test_adam and not test_cmaes:
        raise ValueError("At least one of 'test_adam' or 'test_cmaes' must be true.")

    grid_config = config.get("grid", {})
    if grid_config is None:
        grid_config = {}
    if not isinstance(grid_config, dict):
        raise ValueError("'grid' must be a dictionary with optional 'adam' and 'cmaes' sections.")

    run_queue = []

    if test_adam:
        adam_grid = normalize_grid(grid_config.get("adam", {}))
        adam_combos = expand_grid(adam_grid)
        for combo in adam_combos:
            run_queue.append(("adam", combo))

    if test_cmaes:
        cmaes_grid = normalize_grid(grid_config.get("cmaes", {}))
        cmaes_combos = expand_grid(cmaes_grid)
        for combo in cmaes_combos:
            run_queue.append(("cmaes", combo))

    return run_queue


def create_run_config(base_config, method, overrides, run_id):
    run_config = copy.deepcopy(base_config)

    run_config["optimization_method"] = method
    run_config.update(overrides)

    base_results_folder = str(base_config.get("results_folder", "results"))
    run_config["results_folder"] = os.path.join(base_results_folder, run_id)

    return run_config


def save_run_parameters(results_folder, run_id, method, overrides, run_config):
    """Persist the exact parameter combination used for a run."""
    output_dir = Path(results_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "method": method,
        "overrides": overrides,
        "effective_parameters": {
            key: run_config.get(key)
            for key in sorted(overrides.keys())
        },
    }

    params_path = output_dir / "grid_search_parameters.yaml"
    with params_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return params_path


def run_grid_search(config, dry_run=False):
    validate_base_config(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_queue = build_run_queue(config)

    summary_rows = []
    print(f"Single prompt: {config['selected_prompt']}")
    print(f"Total runs: {len(run_queue)}")

    for idx, (method, overrides) in enumerate(run_queue, start=1):
        run_id = f"run_{idx:03d}_{method}_{timestamp}"
        run_config = create_run_config(config, method, overrides, run_id)

        print("-" * 80)
        print(f"Run {idx}/{len(run_queue)}")
        print(f"Method: {method}")
        print(f"Overrides: {json.dumps(overrides, ensure_ascii=True)}")
        print(f"Results folder: {run_config['results_folder']}")

        run_params_path = save_run_parameters(
            run_config["results_folder"],
            run_id,
            method,
            overrides,
            run_config,
        )
        print(f"Run parameters saved to: {run_params_path}")

        row = {
            "run_index": idx,
            "run_id": run_id,
            "method": method,
            "overrides": overrides,
            "results_folder": run_config["results_folder"],
            "status": "scheduled",
        }

        if dry_run:
            row["status"] = "dry_run"
            summary_rows.append(row)
            continue

        try:
            from eigo import Eigo

            eigo_engine = Eigo(run_config)
            if method == "adam":
                produced_folder = eigo_engine.run_adam_optimization()
            elif method == "cmaes":
                produced_folder = eigo_engine.run_cmaes_optimization()
            else:
                raise ValueError(f"Unsupported method: {method}")

            row["status"] = "success"
            row["produced_folder"] = produced_folder
            save_run_parameters(
                produced_folder,
                run_id,
                method,
                overrides,
                run_config,
            )
        except Exception as exc:  # pylint: disable=broad-except
            row["status"] = "failed"
            row["error"] = str(exc)
            summary_rows.append(row)
            print(f"Run failed: {exc}")
            continue

        summary_rows.append(row)

    return summary_rows


def save_summary(config, summary_rows):
    output_dir = Path(str(config.get("results_folder", "results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "grid_search_summary.yaml"

    payload = {
        "selected_prompt": config.get("selected_prompt"),
        "test_adam": bool(config.get("test_adam", True)),
        "test_cmaes": bool(config.get("test_cmaes", True)),
        "total_runs": len(summary_rows),
        "runs": summary_rows,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"Summary saved to: {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run grid search for EIGO with one prompt and method-specific parameter lists."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to the grid-search config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing optimization.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_yaml(Path(args.config).resolve())
    summary_rows = run_grid_search(config, dry_run=args.dry_run)
    save_summary(config, summary_rows)


if __name__ == "__main__":
    main()
