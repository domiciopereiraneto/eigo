#!/usr/bin/env python3
"""Optimize GenEval prompts with EIGO, evaluate best images, and export scores."""

import argparse
import copy
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TAGS = {
    "single_object": "Single object", "two_object": "Two object",
    "counting": "Counting", "colors": "Colors", "position": "Position",
    "color_attr": "Color attribution",
}
METHODS = {"cmaes", "snes", "cosyne", "gomea", "ga", "adam", "zero_order", "random_sampler"}
FIELDS = ["Model", "Overall", *TAGS.values()]


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sample_paths(folder, prompts, seeds):
    return {str((folder / "images" / f"{index:05d}" / "samples" / f"{sample:04d}.png").resolve()): metadata
            for index, metadata in prompts for sample in range(len(seeds))}


def summarize(path, name, expected):
    """Require exactly one evaluator result per exported image, then macro-average."""
    scores = {tag: [] for tag in TAGS}
    seen = set()
    for row in read_jsonl(path):
        filename = str(Path(row["filename"]).resolve())
        if filename not in expected or filename in seen:
            raise ValueError(f"Unexpected or duplicate evaluation image: {filename}")
        metadata = expected[filename]
        if row["tag"] != metadata["tag"] or row["prompt"] != metadata["prompt"]:
            raise ValueError(f"Evaluation metadata mismatch: {filename}")
        if type(row["correct"]) is not bool:
            raise ValueError(f"Expected boolean correctness: {filename}")
        seen.add(filename)
        scores[row["tag"]].append(row["correct"])
    if seen != set(expected):
        raise ValueError(f"Evaluation is incomplete: {len(seen)}/{len(expected)} images")
    means = {tag: sum(values) / len(values) if values else None for tag, values in scores.items()}
    # Subset runs leave Overall blank unless all six tasks are represented.
    overall = sum(means.values()) / len(TAGS) if all(v is not None for v in means.values()) else None
    return {"Model": name, "Overall": overall, **{TAGS[tag]: value for tag, value in means.items()}}


def generate(manifest, folder, engine_factory=None):
    """One worker per algorithm releases all GPU memory before evaluation."""
    from PIL import Image

    if engine_factory is None:
        sys.path.insert(0, str(ROOT))
        from eigo import Eigo
        engine_factory = Eigo
    engine = None
    for index, metadata in manifest["prompts"]:
        prompt_folder = folder / "images" / f"{index:05d}"
        samples = prompt_folder / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        write_json(prompt_folder / "metadata.jsonl", metadata)
        for sample, seed in enumerate(manifest["seeds"]):
            destination = samples / f"{sample:04d}.png"
            if destination.exists():
                with Image.open(destination) as image:
                    image.verify()
                continue
            if engine is None:
                engine = engine_factory(manifest["config"])
            print(f"{manifest['name']}: prompt {index}, seed {seed}: {metadata['prompt']}", flush=True)
            method = getattr(engine, f"run_{manifest['config']['optimization_method']}_optimization")
            result = Path(method(seed=seed, seed_number=sample + 1, prompt=metadata["prompt"],
                                 category=metadata["tag"], prompt_number=index + 1))
            best = next((result / f"best_all{ext}" for ext in (".png", ".jpg", ".jpeg")
                         if (result / f"best_all{ext}").is_file()), None)
            if best is None:
                raise FileNotFoundError(f"No best_all image in {result}")
            temporary = destination.with_suffix(".tmp")
            with Image.open(best) as image:
                image.convert("RGB").save(temporary, format="PNG")
            temporary.replace(destination)


def run(args):
    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    # All configured paths are relative to the YAML, independent of working directory.
    def resolve(value):
        return (config_path.parent / Path(value).expanduser()).resolve()

    repo = resolve(settings["geneval_repo"])
    metadata_path = resolve(settings["metadata_file"]) if settings.get("metadata_file") else repo / "prompts/evaluation_metadata.jsonl"
    metadata = read_jsonl(metadata_path)
    if not metadata:
        raise ValueError("GenEval prompt file is empty")
    for item in metadata:
        if item.get("tag") not in TAGS or not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError(f"Invalid GenEval metadata: {item}")
    bounds = settings.get("prompt_index_range") or [0, len(metadata)]
    if (not isinstance(bounds, list) or len(bounds) != 2 or any(type(v) is not int for v in bounds)
            or not 0 <= bounds[0] < bounds[1] <= len(metadata)):
        raise ValueError("prompt_index_range must be a valid zero-based [start, end) range")
    prompts = [[i, metadata[i]] for i in range(*bounds)]
    seeds = settings.get("seeds", [42, 43, 44, 45])
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of unique integers in [0, 2**32)")
    base = {}
    if settings.get("base_config"):
        with resolve(settings["base_config"]).open(encoding="utf-8") as stream:
            base = yaml.safe_load(stream)
    base = merge(base, settings.get("defaults", {}))
    algorithms = settings.get("algorithms")
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("algorithms must be a nonempty list of {name, parameters} mappings")
    output = resolve(settings.get("results_folder", "../../results_geneval"))
    jobs, names = [], set()
    for entry in algorithms:
        name = entry["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("Algorithm names must be unique filesystem-safe names")
        names.add(name)
        config = merge(base, entry.get("parameters", {}))
        if config.get("optimization_method") not in METHODS:
            raise ValueError(f"Unsupported optimization_method for {name}")
        folder = output / name
        config["results_folder"] = str(folder / "optimization")
        manifest = {"name": name, "config": config, "prompts": prompts, "seeds": seeds}
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"Configuration/prompts changed for {name}; choose a new results_folder or name")
        elif folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Refusing to reuse nonempty folder without manifest: {folder}")
        jobs.append((folder, manifest))
    evaluator = repo / "evaluation/evaluate_images.py"
    evaluation = settings.get("evaluation", {})
    if args.stage in ("all", "evaluate"):
        if not evaluator.is_file():
            raise FileNotFoundError(f"GenEval evaluator not found: {evaluator}")
        model_path = resolve(evaluation["model_path"])
        if not model_path.is_dir():
            raise FileNotFoundError(f"Detector directory not found: {model_path}")
    rows = []
    for folder, manifest in jobs:
        print(f"{manifest['name']}: {len(prompts)} prompts x {len(seeds)} seeds -> {folder}", flush=True)
        if args.dry_run:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "manifest.json", manifest)
        if args.stage in ("all", "generate"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(folder / "manifest.json")], check=True)
        expected = sample_paths(folder, prompts, seeds)
        results = folder / "results.jsonl"
        if args.stage in ("all", "evaluate"):
            missing = [path for path in expected if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} generated images, first: {missing[0]}")
            temporary = folder / "results.pending.jsonl"
            command = [str(evaluation.get("python", sys.executable)), str(evaluator), str(folder / "images"),
                       "--outfile", str(temporary), "--model-path", str(model_path)]
            if evaluation.get("model_config"):
                command += ["--model-config", str(resolve(evaluation["model_config"]))]
            if evaluation.get("options"):
                command += ["--options", *[f"{key}={value}" for key, value in evaluation["options"].items()]]
            subprocess.run(command, cwd=repo, check=True)
            summarize(temporary, manifest["name"], expected)
            temporary.replace(results)
        if args.stage != "generate":
            rows.append(summarize(results, manifest["name"], expected))
    if rows:
        temporary = output / "geneval_summary.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "geneval_summary.csv")
        print(f"Saved {output / 'geneval_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "algorithms/config/config_run_geneval.yaml"))
    parser.add_argument("--stage", choices=("all", "generate", "evaluate", "summarize"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and print jobs without loading models")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        path = Path(args.worker).resolve()
        generate(json.loads(path.read_text()), path.parent)
    else:
        run(args)


if __name__ == "__main__":
    main()
