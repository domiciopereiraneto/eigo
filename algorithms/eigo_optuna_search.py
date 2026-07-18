#!/usr/bin/env python3
"""Run Optuna-based EIGO hyperparameter search as an alternative to grid search."""

import argparse
import copy
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add script and parent directories to Python path for module imports
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from eigo_grid_search import (  # noqa: E402
    _coerce_scalar,
    create_run_config,
    extract_final_results,
    get_prompt_list,
    load_yaml,
    save_run_parameters,
)
from src.optimization_targets import resolve_optimization_target  # noqa: E402


DEFAULT_CONFIG = "algorithms/config/config_eigo_optuna_search.yaml"
SUPPORTED_METHODS = ("adam", "ga", "cmaes", "snes", "cosyne", "random_sampler", "zero_order")
AUTO_METRICS = {
    "adam": "combined_score",
    "ga": "max_fitness",
    "cmaes": "max_fitness",
    "snes": "max_fitness",
    "cosyne": "max_fitness",
    "random_sampler": "max_fitness",
    "zero_order": "max_fitness",
}
DEFAULT_OPTUNA_PLOTS = {
    "optimization_history": "plot_optimization_history",
    "param_importances": "plot_param_importances",
    "slice": "plot_slice",
    "contour": "plot_contour",
    "parallel_coordinate": "plot_parallel_coordinate",
    "edf": "plot_edf",
    "timeline": "plot_timeline",
    "intermediate_values": "plot_intermediate_values",
    "rank": "plot_rank",
}


def get_enabled_methods(config):
    methods = []
    if bool(config.get("test_adam", True)):
        methods.append("adam")
    if bool(config.get("test_ga", False)):
        methods.append("ga")
    if bool(config.get("test_cmaes", True)):
        methods.append("cmaes")
    if bool(config.get("test_snes", False)):
        methods.append("snes")
    if bool(config.get("test_cosyne", False)):
        methods.append("cosyne")
    if bool(config.get("test_random_sampler", False)):
        methods.append("random_sampler")
    if bool(config.get("test_zero_order", False)):
        methods.append("zero_order")
    if not methods:
        raise ValueError(
            "At least one of 'test_adam', 'test_ga', 'test_cmaes', 'test_snes', 'test_cosyne', "
            "'test_random_sampler', or 'test_zero_order' must be true."
        )
    return methods


def get_optuna_config(config):
    optuna_config = config.get("optuna", {})
    if optuna_config is None:
        optuna_config = {}
    if not isinstance(optuna_config, dict):
        raise ValueError("'optuna' must be a dictionary.")
    return optuna_config


def get_search_space(config):
    optuna_config = get_optuna_config(config)
    search_space = optuna_config.get("search_space", {})
    if search_space is None:
        search_space = {}
    if not isinstance(search_space, dict):
        raise ValueError("'optuna.search_space' must be a dictionary.")
    return search_space


def get_optuna_prompt_source(config):
    optuna_config = get_optuna_config(config)
    prompt_source = optuna_config.get("prompt_source")
    if prompt_source is None:
        return {"type": "config"}
    if not isinstance(prompt_source, dict):
        raise ValueError("'optuna.prompt_source' must be a dictionary.")
    return prompt_source


def _is_clip_tokenizable(prompt, context_length=77):
    try:
        import clip

        clip.tokenize([prompt], context_length=context_length, truncate=False)
        return True
    except RuntimeError as exc:
        if "too long for context length" in str(exc):
            return False
        raise


def _append_prompt_if_valid(prompts, skipped, prompt, require_clip_tokenizable, clip_context_length):
    if not isinstance(prompt, str) or not prompt.strip():
        return

    prompt = prompt.strip()
    if require_clip_tokenizable and not _is_clip_tokenizable(prompt, clip_context_length):
        skipped["too_long_for_clip"] += 1
        return

    prompts.append(prompt)


def _sample_diffusiondb_prompts(prompt_source):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to sample prompts from DiffusionDB."
        ) from exc

    dataset_name = str(prompt_source.get("dataset", "poloclub/diffusiondb"))
    subset = prompt_source.get("subset", "2m_random_1k")
    split = str(prompt_source.get("split", "train"))
    prompt_column = str(prompt_source.get("prompt_column", "prompt"))
    count = int(prompt_source.get("count", 10))
    seed = int(prompt_source.get("seed", 42))
    streaming = bool(prompt_source.get("streaming", False))
    require_clip_tokenizable = bool(prompt_source.get("require_clip_tokenizable", True))
    clip_context_length = int(prompt_source.get("clip_context_length", 77))

    if count <= 0:
        raise ValueError("'optuna.prompt_source.count' must be greater than zero.")
    if clip_context_length <= 0:
        raise ValueError("'optuna.prompt_source.clip_context_length' must be greater than zero.")

    print(
        "Loading DiffusionDB prompts "
        f"(dataset={dataset_name}, subset={subset}, split={split}, count={count}, seed={seed})"
    )

    load_args = [dataset_name]
    if subset is not None:
        load_args.append(str(subset))
    dataset = load_dataset(*load_args, split=split, streaming=streaming)
    skipped = {"too_long_for_clip": 0}

    if streaming:
        buffer_size = int(prompt_source.get("buffer_size", max(count * 20, 1000)))
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        prompts = []
        for row in dataset:
            _append_prompt_if_valid(
                prompts,
                skipped,
                row.get(prompt_column),
                require_clip_tokenizable,
                clip_context_length,
            )
            if len(prompts) >= count:
                break
    else:
        if prompt_column not in dataset.column_names:
            raise ValueError(
                f"Prompt column '{prompt_column}' not found in DiffusionDB split. "
                f"Available columns: {', '.join(dataset.column_names)}"
            )
        candidates = list(dataset[prompt_column])
        rng = random.Random(seed)
        rng.shuffle(candidates)
        prompts = []
        for prompt in candidates:
            _append_prompt_if_valid(
                prompts,
                skipped,
                prompt,
                require_clip_tokenizable,
                clip_context_length,
            )
            if len(prompts) >= count:
                break

    if len(prompts) < count:
        raise ValueError(
            f"DiffusionDB prompt source yielded {len(prompts)} prompt(s), "
            f"but {count} were requested. Skipped prompts: {skipped}."
        )
    if skipped["too_long_for_clip"]:
        print(
            "Skipped "
            f"{skipped['too_long_for_clip']} DiffusionDB prompt(s) longer than "
            f"CLIP context length {clip_context_length}."
        )
    return prompts


def get_optuna_prompt_list(config):
    prompt_source = get_optuna_prompt_source(config)
    source_type = str(prompt_source.get("type", "config")).lower()

    if source_type == "config":
        return get_prompt_list(config)
    if source_type == "diffusiondb":
        return _sample_diffusiondb_prompts(prompt_source)

    raise ValueError(
        "Unsupported 'optuna.prompt_source.type': "
        f"{source_type}. Supported values: config, diffusiondb."
    )


def validate_prompt_source_config(config):
    prompt_source = get_optuna_prompt_source(config)
    source_type = str(prompt_source.get("type", "config")).lower()

    if source_type == "config":
        get_prompt_list(config)
        return

    if source_type == "diffusiondb":
        count = int(prompt_source.get("count", 10))
        if count <= 0:
            raise ValueError("'optuna.prompt_source.count' must be greater than zero.")
        if "dataset" in prompt_source and not str(prompt_source["dataset"]).strip():
            raise ValueError("'optuna.prompt_source.dataset' must be non-empty.")
        if "subset" in prompt_source and prompt_source["subset"] is not None:
            if not str(prompt_source["subset"]).strip():
                raise ValueError("'optuna.prompt_source.subset' must be non-empty or null.")
        if "split" in prompt_source and not str(prompt_source["split"]).strip():
            raise ValueError("'optuna.prompt_source.split' must be non-empty.")
        if "prompt_column" in prompt_source and not str(prompt_source["prompt_column"]).strip():
            raise ValueError("'optuna.prompt_source.prompt_column' must be non-empty.")
        return

    raise ValueError(
        "Unsupported 'optuna.prompt_source.type': "
        f"{source_type}. Supported values: config, diffusiondb."
    )


def validate_optuna_config(config):
    optuna_config = get_optuna_config(config)
    validate_prompt_source_config(config)
    resolve_optimization_target(config)
    if "results_folder" not in config:
        raise ValueError("Missing required key 'results_folder' in config.")

    n_trials = int(optuna_config.get("n_trials", 20))
    if n_trials <= 0:
        raise ValueError("'optuna.n_trials' must be greater than zero.")

    direction = optuna_config.get("direction", "maximize")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("'optuna.direction' must be either 'maximize' or 'minimize'.")

    enabled_methods = get_enabled_methods(config)
    search_space = get_search_space(config)
    unknown_methods = sorted(set(search_space.keys()) - set(SUPPORTED_METHODS))
    if unknown_methods:
        raise ValueError(
            "'optuna.search_space' contains unsupported method sections: "
            f"{', '.join(unknown_methods)}"
        )

    for method in enabled_methods:
        method_space = search_space.get(method, {})
        if method_space is None:
            continue
        if not isinstance(method_space, dict):
            raise ValueError(f"'optuna.search_space.{method}' must be a dictionary.")

    _get_enabled_plot_names(_get_plot_config(config))


def _suggest_categorical(trial, name, values):
    if not values:
        raise ValueError(f"Categorical parameter '{name}' must have at least one choice.")
    return trial.suggest_categorical(name, [_coerce_scalar(value) for value in values])


def suggest_parameter(trial, name, spec):
    """Suggest one Optuna parameter from a compact YAML spec."""
    if isinstance(spec, list):
        return _suggest_categorical(trial, name, spec)

    if not isinstance(spec, dict):
        return _coerce_scalar(spec)

    param_type = spec.get("type", "categorical" if "choices" in spec else "float")
    if param_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list):
            raise ValueError(f"Categorical parameter '{name}' must define a list of choices.")
        return _suggest_categorical(trial, name, choices)

    if param_type == "float":
        low = _coerce_scalar(spec.get("low"))
        high = _coerce_scalar(spec.get("high"))
        if low is None or high is None:
            raise ValueError(f"Float parameter '{name}' must define low and high.")
        return trial.suggest_float(
            name,
            float(low),
            float(high),
            log=bool(spec.get("log", False)),
            step=_coerce_scalar(spec.get("step")),
        )

    if param_type == "int":
        low = _coerce_scalar(spec.get("low"))
        high = _coerce_scalar(spec.get("high"))
        if low is None or high is None:
            raise ValueError(f"Int parameter '{name}' must define low and high.")
        return trial.suggest_int(
            name,
            int(low),
            int(high),
            log=bool(spec.get("log", False)),
            step=int(_coerce_scalar(spec.get("step", 1))),
        )

    raise ValueError(f"Unsupported Optuna parameter type for '{name}': {param_type}")


def condition_matches(overrides, condition):
    if condition is None:
        return True
    if not isinstance(condition, dict):
        raise ValueError("'depends_on' must be a dictionary.")

    parameter = condition.get("parameter")
    if not parameter:
        raise ValueError("'depends_on.parameter' must be set.")
    if parameter not in overrides:
        return False

    actual = overrides[parameter]
    if "values" in condition:
        values = condition["values"]
        if not isinstance(values, list):
            raise ValueError("'depends_on.values' must be a list.")
        return actual in [_coerce_scalar(value) for value in values]
    if "value" in condition:
        return actual == _coerce_scalar(condition["value"])

    raise ValueError("'depends_on' must define either 'value' or 'values'.")


def parameter_spec_without_condition(spec):
    if not isinstance(spec, dict) or "depends_on" not in spec:
        return spec
    stripped = dict(spec)
    stripped.pop("depends_on")
    return stripped


def suggest_overrides(trial, method, search_space):
    method_space = search_space.get(method, {}) or {}
    overrides = {}
    for key, spec in method_space.items():
        if isinstance(spec, dict) and not condition_matches(overrides, spec.get("depends_on")):
            continue
        # Prefix trial parameter names because the same backend key can have different
        # distributions per optimizer method.
        overrides[key] = suggest_parameter(
            trial,
            f"{method}.{key}",
            parameter_spec_without_condition(spec),
        )
    return overrides


def create_sampler(optuna_config):
    import optuna

    sampler_config = optuna_config.get("sampler", {})
    if sampler_config is None:
        sampler_config = {}
    if not isinstance(sampler_config, dict):
        raise ValueError("'optuna.sampler' must be a dictionary.")

    name = str(sampler_config.get("name", "tpe")).lower()
    seed = sampler_config.get("seed")
    seed = None if seed is None else int(seed)

    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    raise ValueError(f"Unsupported Optuna sampler: {name}")


def ensure_optuna_storage_path(storage):
    """Create parent directories for local SQLite Optuna storage."""
    if not storage or not isinstance(storage, str):
        return

    if storage == "sqlite:///:memory:" or not storage.startswith("sqlite:///"):
        return

    db_path = storage.removeprefix("sqlite:///")
    if not db_path:
        return

    db_path = Path(db_path)
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)


def objective_metric(final_results, method, metric_name):
    if not isinstance(final_results, dict) or final_results.get("status") != "ok":
        raise ValueError(f"Cannot compute objective from final results: {final_results}")

    resolved_metric = AUTO_METRICS[method] if metric_name == "auto" else metric_name
    value = _coerce_scalar(final_results.get(resolved_metric))
    if value is None:
        raise ValueError(f"Metric '{resolved_metric}' not found in final results.")

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Metric '{resolved_metric}' is not finite: {value}")
    return value, resolved_metric


def run_eigo_method(run_config, method):
    from eigo import Eigo

    eigo_engine = Eigo(run_config)
    if method == "adam":
        return eigo_engine.run_adam_optimization()
    if method == "ga":
        return eigo_engine.run_ga_optimization()
    if method == "cmaes":
        return eigo_engine.run_cmaes_optimization()
    if method == "snes":
        return eigo_engine.run_snes_optimization()
    if method == "cosyne":
        return eigo_engine.run_cosyne_optimization()
    if method == "random_sampler":
        return eigo_engine.run_random_sampler_optimization()
    if method == "zero_order":
        return eigo_engine.run_zero_order_optimization()
    raise ValueError(f"Unsupported method: {method}")


def save_optuna_trial_parameters(results_folder, run_id, method, overrides, run_config, trial_number):
    params_path = save_run_parameters(results_folder, run_id, method, overrides, run_config)
    optuna_params_path = Path(results_folder) / "optuna_search_parameters.yaml"
    with optuna_params_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "trial_number": trial_number,
                "run_id": run_id,
                "method": method,
                "selected_prompt": run_config.get("selected_prompt"),
                "overrides": overrides,
                "effective_parameters": {
                    key: run_config.get(key)
                    for key in sorted(set(overrides.keys()) | {"optimization_target"})
                },
            },
            f,
            sort_keys=False,
        )
    return params_path, optuna_params_path


def run_optuna_search(config, dry_run=False):
    validate_optuna_config(config)

    optuna_config = get_optuna_config(config)
    prompt_list = get_optuna_prompt_list(config)
    config["_resolved_optuna_prompts"] = prompt_list
    enabled_methods = get_enabled_methods(config)
    search_space = get_search_space(config)
    n_trials = int(optuna_config.get("n_trials", 20))
    direction = optuna_config.get("direction", "maximize")
    metric_name = optuna_config.get("metric", "auto")
    default_optimization_target = resolve_optimization_target(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_rows = []

    print(f"Prompt count per trial: {len(prompt_list)}")
    print(f"Enabled methods: {', '.join(enabled_methods)}")
    print(f"Optuna trials: {n_trials}")
    print(f"Objective direction: {direction}")
    print(f"Objective metric: {metric_name}")
    print(f"Default optimization target: {default_optimization_target}")

    if dry_run:
        for method in enabled_methods:
            print("-" * 80)
            print(f"Method: {method}")
            print(f"Search space: {json.dumps(search_space.get(method, {}), ensure_ascii=True)}")
        return None, trial_rows

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for this script. Install it with `pip install optuna` "
            "or update the project environment."
        ) from exc

    study_name = optuna_config.get("study_name") or f"eigo_optuna_{timestamp}"
    storage = optuna_config.get("storage")
    ensure_optuna_storage_path(storage)
    load_if_exists = bool(optuna_config.get("load_if_exists", True))
    study = optuna.create_study(
        direction=direction,
        sampler=create_sampler(optuna_config),
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )

    def objective(trial):
        method = (
            enabled_methods[0]
            if len(enabled_methods) == 1
            else trial.suggest_categorical("method", enabled_methods)
        )
        overrides = suggest_overrides(trial, method, search_space)
        values = []
        prompt_results = []

        for prompt_idx, prompt in enumerate(prompt_list, start=1):
            run_id = (
                f"trial_{trial.number:04d}_prompt_{prompt_idx:03d}_"
                f"{method}_{timestamp}"
            )
            run_config = create_run_config(config, method, overrides, run_id, prompt)
            resolved_optimization_target = resolve_optimization_target(run_config)

            print("-" * 80)
            print(f"Trial {trial.number}, prompt {prompt_idx}/{len(prompt_list)}")
            print(f"Prompt: {prompt}")
            print(f"Method: {method}")
            print(f"Optimization target: {resolved_optimization_target}")
            print(f"Overrides: {json.dumps(overrides, ensure_ascii=True)}")
            print(f"Results folder: {run_config['results_folder']}")

            _, optuna_params_path = save_optuna_trial_parameters(
                run_config["results_folder"],
                run_id,
                method,
                overrides,
                run_config,
                trial.number,
            )
            print(f"Trial parameters saved to: {optuna_params_path}")

            produced_folder = run_eigo_method(run_config, method)
            final_results = extract_final_results(produced_folder, method)
            value, resolved_metric = objective_metric(final_results, method, metric_name)
            values.append(value)
            save_optuna_trial_parameters(
                produced_folder,
                run_id,
                method,
                overrides,
                run_config,
                trial.number,
            )
            prompt_results.append(
                {
                    "prompt_index": prompt_idx,
                    "selected_prompt": prompt,
                    "run_id": run_id,
                    "results_folder": run_config["results_folder"],
                    "produced_folder": produced_folder,
                    "metric": resolved_metric,
                    "value": value,
                    "optimization_target": resolved_optimization_target,
                    "final_results": final_results,
                }
            )
            trial.report(float(sum(values) / len(values)), step=prompt_idx)

            if trial.should_prune():
                trial.set_user_attr("method", method)
                trial.set_user_attr("overrides", copy.deepcopy(overrides))
                trial.set_user_attr("prompt_results", prompt_results)
                raise optuna.TrialPruned()

        objective_value = float(sum(values) / len(values))
        row = {
            "trial_number": trial.number,
            "method": method,
            "overrides": copy.deepcopy(overrides),
            "objective_value": objective_value,
            "prompt_results": prompt_results,
        }
        trial_rows.append(row)
        trial.set_user_attr("method", method)
        trial.set_user_attr("overrides", copy.deepcopy(overrides))
        trial.set_user_attr("prompt_results", prompt_results)
        return objective_value

    catch_errors = bool(optuna_config.get("catch_trial_errors", True))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,) if catch_errors else ())
    return study, trial_rows


def _best_trial_payload(study):
    if study is None:
        return None
    try:
        best_trial = study.best_trial
    except ValueError:
        return None
    return {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": dict(best_trial.params),
        "user_attrs": dict(best_trial.user_attrs),
    }


def save_summary(config, study, trial_rows):
    output_dir = Path(str(config.get("results_folder", "results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "optuna_search_summary.yaml"
    optuna_config = get_optuna_config(config)

    payload = {
        "selected_prompts": config.get("_resolved_optuna_prompts") or get_optuna_prompt_list(config),
        "prompt_source": get_optuna_prompt_source(config),
        "enabled_methods": get_enabled_methods(config),
        "default_optimization_target": resolve_optimization_target(config),
        "n_trials": int(optuna_config.get("n_trials", 20)),
        "direction": optuna_config.get("direction", "maximize"),
        "metric": optuna_config.get("metric", "auto"),
        "study_name": None if study is None else study.study_name,
        "best_trial": _best_trial_payload(study),
        "completed_trial_runs": trial_rows,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"Summary saved to: {summary_path}")


def _get_plot_config(config):
    optuna_config = get_optuna_config(config)
    plot_config = optuna_config.get("plots", {})
    if plot_config is None:
        plot_config = {}
    if not isinstance(plot_config, dict):
        raise ValueError("'optuna.plots' must be a dictionary when provided.")
    return plot_config


def _get_enabled_plot_names(plot_config):
    enabled = plot_config.get("enabled", True)
    if isinstance(enabled, bool):
        return list(DEFAULT_OPTUNA_PLOTS.keys()) if enabled else []
    if isinstance(enabled, list):
        unknown = sorted(set(enabled) - set(DEFAULT_OPTUNA_PLOTS.keys()))
        if unknown:
            raise ValueError(
                "'optuna.plots.enabled' contains unsupported plot names: "
                f"{', '.join(unknown)}"
            )
        return list(enabled)
    raise ValueError("'optuna.plots.enabled' must be a boolean or list of plot names.")


def save_optuna_plots(config, study):
    """Save Optuna visualization plots as HTML files.

    Plot generation is intentionally best-effort: a finished study should not fail
    just because one diagnostic plot is unavailable for the completed trials.
    """
    if study is None:
        return

    plot_config = _get_plot_config(config)
    plot_names = _get_enabled_plot_names(plot_config)
    if not plot_names:
        print("Optuna plot generation disabled.")
        return

    try:
        import optuna
        from optuna.trial import TrialState
        import optuna.visualization as vis
    except ImportError as exc:
        print(f"Warning: Optuna visualization unavailable ({exc}). Skipping plots.")
        return

    completed_trials = study.get_trials(
        deepcopy=False,
        states=(TrialState.COMPLETE,),
    )
    if not completed_trials:
        print("No completed Optuna trials available. Skipping plots.")
        return

    output_root = Path(str(config.get("results_folder", "results")))
    output_dir = plot_config.get("output_dir")
    if output_dir is None:
        output_dir = output_root / "optuna_plots"
    else:
        output_dir = Path(str(output_dir))
        if not output_dir.is_absolute():
            output_dir = output_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    include_plotlyjs = plot_config.get("include_plotlyjs", "cdn")
    saved = []
    failed = []

    for plot_name in plot_names:
        function_name = DEFAULT_OPTUNA_PLOTS[plot_name]
        plot_function = getattr(vis, function_name, None)
        if plot_function is None:
            failed.append(
                {
                    "plot": plot_name,
                    "error": f"optuna.visualization.{function_name} is unavailable",
                }
            )
            continue

        try:
            fig = plot_function(study)
            out_path = output_dir / f"{plot_name}.html"
            fig.write_html(str(out_path), include_plotlyjs=include_plotlyjs)
            saved.append(str(out_path))
        except Exception as exc:
            failed.append({"plot": plot_name, "error": str(exc)})
            print(f"Warning: failed to save Optuna plot '{plot_name}': {exc}")

    manifest_path = output_dir / "plot_manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study_name": study.study_name,
                "completed_trials": len(completed_trials),
                "saved": saved,
                "failed": failed,
            },
            f,
            sort_keys=False,
        )

    if saved:
        print(f"Optuna plots saved under: {output_dir}")
    else:
        print(f"No Optuna plots were saved. See: {manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna hyperparameter search for EIGO with one or more prompts."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to the Optuna-search config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print search spaces without executing optimization.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_yaml(Path(args.config).resolve())
    study, trial_rows = run_optuna_search(config, dry_run=args.dry_run)
    save_summary(config, study, trial_rows)
    save_optuna_plots(config, study)


if __name__ == "__main__":
    main()
