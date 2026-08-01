#!/usr/bin/env python3
"""Run Optuna studies sequentially from scheduled config overrides.

Each schedule entry is recursively merged into the base Optuna config. Use this
for paper-scale optimizer-objective tuning where every study has its own result
folder and SQLite storage file.
"""

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively replace values in base with values from updates."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_schedule(data):
    """Accept either a top-level list or a {'runs': [...]} mapping."""
    if isinstance(data, list):
        runs = data
    elif isinstance(data, dict) and isinstance(data.get("runs"), list):
        runs = data["runs"]
    else:
        raise ValueError(
            "Schedule YAML must be either a list of dictionaries or a dictionary "
            "with a 'runs' list."
        )

    if not runs:
        raise ValueError("Schedule is empty. Add at least one run override dictionary.")

    for idx, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"Run #{idx} is not a dictionary.")

    return runs


def run_schedule(args):
    base_config_path = Path(args.base_config).resolve()
    schedule_path = Path(args.schedule).resolve()
    script_path = Path(args.script).resolve()

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    base_config = load_yaml(base_config_path)
    if not isinstance(base_config, dict):
        raise ValueError("Base config must load to a dictionary.")

    runs = parse_schedule(load_yaml(schedule_path))
    print(f"Loaded {len(runs)} scheduled run(s).")

    with tempfile.TemporaryDirectory(prefix="optuna_schedule_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for idx, override in enumerate(runs, start=1):
            run_config = deep_update(copy.deepcopy(base_config), override)
            run_config_path = tmpdir_path / f"run_{idx:03d}.yaml"

            with run_config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(run_config, f, sort_keys=False)

            cmd = [args.python, str(script_path), "--config", str(run_config_path)]

            print("-" * 80)
            print(f"Run {idx}/{len(runs)}")
            print(f"Overrides: {override}")
            print(f"Command: {' '.join(cmd)}")

            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                message = f"Run {idx} failed with exit code {result.returncode}."
                if args.continue_on_error:
                    print(message + " Continuing.")
                else:
                    raise RuntimeError(message)

    print("Schedule finished.")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run eigo_optuna_search.py sequentially using a base config and a "
            "schedule of overrides."
        )
    )
    parser.add_argument(
        "--base-config",
        default="algorithms/config/config_eigo_optuna_search.yaml",
        help="Path to the base Optuna-search config YAML.",
    )
    parser.add_argument(
        "--schedule",
        default="algorithms/config/optuna_schedule.yaml",
        help="Path to schedule YAML (top-level list or {'runs': [...]}).",
    )
    parser.add_argument(
        "--script",
        default="algorithms/eigo_optuna_search.py",
        help="Path to the Optuna-search script executed for each run.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for each scheduled run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the run configs and print commands without executing them.",
    )
    return parser


def main():
    run_schedule(build_parser().parse_args())


if __name__ == "__main__":
    main()
