"""Per-prompt optimizer checkpoints for extending batch experiments.

Checkpoints retain compact optimizer state plus histories needed to extend reports.
They include Python/NumPy objects and must only be loaded from trusted local
experiment folders, with the same Python/PyTorch/optimizer environment.
"""
from src.optimizer_checkpoint import pack_run_state, unpack_run_state

from pathlib import Path
import os
import random
import shutil


LIMIT_KEYS = {
    "num_generations", "num_iterations", "snes_num_generations",
    "cc_snes_num_generations", "cosyne_num_generations",
    "gomea_num_generations", "zero_order_num_generations", "num_images_to_generate", "num_images", "random_sampler_num_images",
}
# These settings do not change an individual prompt's optimization problem.
RUN_KEYS = {
    "resume_experiment", "results_folder", "seed_path", "selected_prompt",
    "prompt_dataset", "prompt_dataset_path", "prompt_dataset_config",
    "prompt_dataset_split", "prompt_column", "prompt_category_column",
    "prompt_per_categorie", "prompt_sample_seed", "single_prompt_per_seed",
    "use_entire_dataset", "prompt_index_range", "time_limit_seconds", "save_gens",
}


def experiment_limit(config):
    method = config["optimization_method"]
    if method == "adam":
        return int(config["num_iterations"])
    if method == "random_sampler":
        for key in ("num_images_to_generate", "num_images", "random_sampler_num_images"):
            if config.get(key) is not None:
                return int(config[key])
    keys = []
    if method == "snes" and config.get("optimization_target") == "noise_embeddings_cc":
        keys.append("cc_snes_num_generations")
    keys += [f"{method}_num_generations", "num_generations"]
    for key in keys:
        if config.get(key) is not None:
            return int(config[key])
    raise ValueError(f"No generation limit configured for {method}.")


class ExperimentResumeMixin:
    def _experiment_limit(self):
        return experiment_limit(self._resume_parameters)

    def _load_experiment_checkpoint(self, folder, seed, prompt):
        self._checkpoint_ensemble_count = 0
        if not self._resume_parameters.get("resume_experiment", False):
            return None
        folder = Path(folder)
        path = folder / "checkpoint.pt"
        config_path = folder / "config.yaml"
        if not path.exists():
            if not config_path.exists():
                if folder.exists() and any(folder.iterdir()):
                    raise ValueError(f"Cannot resume {folder}: missing config.yaml and checkpoint.pt.")
                return None
            import yaml
            old = yaml.safe_load(config_path.read_text())
            if old.get("seed") != int(seed) or old.get("selected_prompt") != prompt:
                raise ValueError(f"Cannot resume {folder}: saved seed or prompt differs.")
            if self._experiment_limit() <= experiment_limit(old):
                self._experiment_skipped = True
                print(f"Skipping {folder}: requested total has not increased.")
                return {"skip": True}
            raise ValueError(
                f"Cannot extend {folder}: no optimizer checkpoint.pt exists. "
                "Older runs saved only scores/images and cannot be resumed. "
                "Start a new experiment in a different results_folder. "
                "GOMEA does not support optimizer checkpoints."
            )

        import torch
        # v1 contains full optimizer objects; v2 contains compact state snapshots.
        checkpoint = torch.load(path, weights_only=False)
        if checkpoint.get("version") not in (1, 2):
            raise ValueError(f"Unsupported experiment checkpoint version in {path}.")
        if checkpoint["seed"] != int(seed) or checkpoint["prompt"] != prompt:
            raise ValueError(f"Cannot resume {folder}: saved seed or prompt differs.")
        old = checkpoint["config"]
        new = self._resume_parameters
        changed = sorted(key for key in old.keys() | new.keys()
                         if key not in LIMIT_KEYS | RUN_KEYS and old.get(key) != new.get(key))
        if changed:
            raise ValueError(f"Cannot resume {folder}: incompatible settings: {', '.join(changed)}.")
        if self._experiment_limit() <= checkpoint["limit"]:
            self._experiment_skipped = True
            print(f"Skipping {folder}: requested total has not increased.")
            return {"skip": True}
        if checkpoint["version"] == 2:
            checkpoint["state"] = unpack_run_state(checkpoint["state"])
        run = getattr(self, "_ensemble_run", None)
        if run is not None:
            run.records = checkpoint["ensemble_records"]
            run.cohort = checkpoint["ensemble_cohort"]
            self._checkpoint_ensemble_count = len(run.records)
            for record in run.records:
                name = f"{record['sample_id']}.jpg"
                shutil.copyfile(folder / "checkpoint_ensemble" / name, Path(run.directory.name) / name)
        print(f"Resuming {folder} after step {checkpoint['step']} to total {self._experiment_limit()}.")
        return checkpoint

    @staticmethod
    def _restore_experiment_rng(checkpoint):
        import numpy as np
        import torch
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])

    def _extend_population_optimizer(self, optimizer):
        limit = self._experiment_limit()
        if hasattr(optimizer, "max_generations"):
            optimizer.max_generations = limit
        else:
            optimizer.opts.set({"maxiter": limit})
            # Clear pycma's cached stop result from the previous generation limit.
            optimizer.stop(check=False).clear()

    def _save_experiment_checkpoint(self, folder, seed, prompt, step, elapsed, local_state, state_names):
        import numpy as np
        import torch
        checkpoint = {
            "version": 2, "config": self._resume_parameters,
            "seed": int(seed), "prompt": prompt, "step": int(step),
            "limit": self._experiment_limit(), "elapsed": elapsed,
            "state": pack_run_state({key: local_state[key] for key in state_names}),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        run = getattr(self, "_ensemble_run", None)
        if run is not None:
            directory = Path(folder) / "checkpoint_ensemble"
            directory.mkdir(exist_ok=True)
            for record in run.records[self._checkpoint_ensemble_count:]:
                name = f"{record['sample_id']}.jpg"
                destination = directory / name
                shutil.copyfile(Path(run.directory.name) / name, destination)
            checkpoint["ensemble_records"] = run.records
            checkpoint["ensemble_cohort"] = run.cohort
        path = Path(folder) / "checkpoint.pt"
        temporary = path.with_suffix(".pt.tmp")
        try:
            torch.save(checkpoint, temporary, pickle_protocol=5)
            os.replace(temporary, path)
            if run is not None:
                self._checkpoint_ensemble_count = len(run.records)
        finally:
            temporary.unlink(missing_ok=True)
