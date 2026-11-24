#!/usr/bin/env python3
"""Run batch experiments sequentially from scheduled config overrides.

Each schedule entry is recursively merged into the base run_experiments.py config,
written to a temporary YAML file, and executed as one batch job.

A schedule entry may set ``config_file`` to the path of another config YAML. That
file is merged on top of the base config first, and the entry's remaining keys are
then merged on top of it, so inline parameters always take precedence. Relative
paths are resolved against the current working directory, falling back to the
schedule file's directory.
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
    """Recursively merge updates into base and return base."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


CONFIG_FILE_KEY = "config_file"


def resolve_config_file(path_str, schedule_dir: Path) -> Path:
    """Resolve a run's config_file path (cwd first, then schedule directory)."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, schedule_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Run config_file not found: {path_str} (tried: {', '.join(str(c) for c in candidates)})"
    )


def build_run_config(base_config: dict, override: dict, schedule_dir: Path) -> dict:
    """Merge base config <- optional config_file <- inline overrides."""
    run_config = copy.deepcopy(base_config)
    override = dict(override)
    config_file = override.pop(CONFIG_FILE_KEY, None)
    if config_file:
        config_path = resolve_config_file(config_file, schedule_dir)
        file_config = load_yaml(config_path) or {}
        if not isinstance(file_config, dict):
            raise ValueError(f"Config file must load to a dictionary: {config_path}")
        deep_update(run_config, copy.deepcopy(file_config))
    return deep_update(run_config, copy.deepcopy(override))


def parse_schedule(data):
    """Accept either a top-level list or {'runs': [...]} format."""
    if isinstance(data, list):
        runs = data
    elif isinstance(data, dict) and isinstance(data.get("runs"), list):
        runs = data["runs"]
    else:
        raise ValueError(
            "Schedule YAML must be either a list of dictionaries or a dictionary with a 'runs' list."
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

    schedule_data = load_yaml(schedule_path)
    runs = parse_schedule(schedule_data)

    print(f"Loaded {len(runs)} scheduled run(s).")

    with tempfile.TemporaryDirectory(prefix="experiments_schedule_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for idx, override in enumerate(runs, start=1):
            run_config = build_run_config(base_config, override, schedule_path.parent)
            run_config_path = tmpdir_path / f"run_{idx:03d}.yaml"

            with run_config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(run_config, f, sort_keys=False)

            cmd = [
                args.python,
                str(script_path),
                "--config",
                str(run_config_path),
            ]

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
            "Run run_experiments.py sequentially using a base config and a schedule of overrides."
        )
    )
    parser.add_argument(
        "--base-config",
        default="algorithms/config/config_run_experiments.yaml",
        help="Path to the base config YAML.",
    )
    parser.add_argument(
        "--schedule",
        default="algorithms/config/experiments_schedule.yaml",
        help="Path to schedule YAML (top-level list or {'runs': [...]}).",
    )
    parser.add_argument(
        "--script",
        default="algorithms/run_experiments.py",
        help="Path to the script to execute for each run.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the script for each scheduled run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute runs.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_schedule(args)


if __name__ == "__main__":
    main()
