"""Equal-weight verifier ranks and per-run candidate bookkeeping."""
from functools import wraps
from pathlib import Path
import shutil
import json
import tempfile

import numpy as np
import pandas as pd

METRICS = (
    "aesthetic_score", "clip_score", "image_reward_score", "hpsv2_score",
    "pickscore_score", "jpeg_size_kb",
)
ALIASES = {name.removesuffix("_score"): name for name in METRICS}
ALIASES.update({"jpeg_size": "jpeg_size_kb", "pickscore": "pickscore_score"})


def normalize_ensemble_list(value):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("ensemble_list must be a list of metric names or null.")
    result = []
    for name in value:
        if not isinstance(name, str):
            raise ValueError("ensemble_list entries must be metric names.")
        name = ALIASES.get(name.strip().lower(), name.strip().lower())
        if name not in METRICS:
            raise ValueError(f"Unknown ensemble metric {name!r}; choose from {', '.join(METRICS)}.")
        if name in result:
            raise ValueError(f"Duplicate ensemble metric: {name}.")
        result.append(name)
    return result


def ensemble_scores(samples, metrics):
    """Mean ascending rank (1 = worst), with average ranks for ties.

    JPEG size is minimized; all other verifiers are maximized. No metric scale
    or configured objective weight participates in ranking.
    """
    if not metrics:
        return np.zeros(len(samples), dtype=float)
    if not len(samples):
        return np.empty(0, dtype=float)
    frame = pd.DataFrame(samples)[list(metrics)].astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Ensemble metrics must contain only finite scores.")
    if "jpeg_size_kb" in frame:
        frame["jpeg_size_kb"] *= -1
    return frame.rank(method="average", ascending=True).mean(axis=1).to_numpy()


class EnsembleRun:
    """Keep scalar metrics and temporary images, never full latent populations."""
    def __init__(self, engine):
        self.engine = engine
        self.directory = tempfile.TemporaryDirectory(prefix="eigo-ensemble-")
        self.records = []
        self.cohort = 0

    def add(self, image, result):
        sample_id = len(self.records)
        path = Path(self.directory.name) / f"{sample_id}.jpg"
        self.engine._save_jpeg(image, str(path))
        self.records.append(dict(zip(METRICS, result[1:7]), sample_id=sample_id,
                                 cohort=self.cohort))

    def finish(self, folder):
        engine = self.engine
        frame = pd.DataFrame(self.records)
        if frame.empty:
            return
        frame["ensemble_score"] = ensemble_scores(self.records, engine.ensemble_list)
        frame["selection_ensemble_score"] = 0.0
        for _, indices in frame.groupby("cohort").groups.items():
            frame.loc[indices, "selection_ensemble_score"] = ensemble_scores(
                frame.loc[indices].to_dict("records"), engine.ensemble_list)
        best = int(frame["ensemble_score"].idxmax())
        frame["selected"] = frame.index == best
        frame.to_csv(Path(folder) / "ensemble_samples.csv", index=False)
        shutil.copyfile(Path(self.directory.name) / f"{best}.jpg", Path(folder) / "best_all.jpg")

        selected = frame.loc[best].to_dict()
        selected["ensemble_list"] = engine.ensemble_list
        (Path(folder) / "ensemble_best.json").write_text(json.dumps(selected, indent=2) + "\n")
        sample_path = Path(folder) / "sample_results.csv"
        if sample_path.exists():
            samples = pd.read_csv(sample_path)
            if len(samples) != len(frame):
                raise ValueError("Ensemble sample history does not match sample_results.csv.")
            samples["ensemble_score"] = frame["ensemble_score"].to_numpy()
            samples["fitness"] = samples["ensemble_score"]
            samples.to_csv(sample_path, index=False)

        path = Path(folder) / "fitness_results.csv"
        if path.exists():
            results = pd.read_csv(path)
            # Cooperative methods evaluate two subpopulations per generation.
            cooperative = engine.optimization_target == "noise_embeddings_cc"
            groups = (frame["cohort"] + 1) // 2 if cooperative else frame["cohort"]
            for generation, subset in frame.groupby(groups):
                if generation >= len(results):
                    continue
                # Report all generations on the same final, run-wide rank scale.
                scores = subset["ensemble_score"].to_numpy()
                best_so_far = frame.loc[groups <= generation, "ensemble_score"].max()
                for prefix, value in (("avg", np.mean(scores)), ("std", np.std(scores)),
                                      ("max", best_so_far)):
                    results.loc[generation, f"{prefix}_ensemble_score"] = value
                    results.loc[generation, f"{prefix}_fitness"] = value
            results["ensemble_score"] = results["max_ensemble_score"]
            results.to_csv(path, index=False)
            engine._save_population_plot_results(results, str(folder))


def ensemble_run(method):
    """Scope candidate history to a prompt/seed, including nested dispatch."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "_ensemble_run", None) is not None:
            return method(self, *args, **kwargs)
        run = EnsembleRun(self) if self.ensemble_list else None
        self._ensemble_run = run
        try:
            folder = method(self, *args, **kwargs)
            if run is not None:
                run.finish(folder)
            else:
                for filename in ("fitness_results.csv", "score_results.csv", "runtime_score_results.csv",
                                 "sample_results.csv"):
                    path = Path(folder) / filename
                    if path.exists():
                        frame = pd.read_csv(path)
                        frame["ensemble_score"] = 0.0
                        if "avg_fitness" in frame:
                            for prefix in ("avg", "std", "max"):
                                frame[f"{prefix}_ensemble_score"] = 0.0
                        frame.to_csv(path, index=False)
            return folder
        finally:
            self._ensemble_run = None
            if run is not None:
                run.directory.cleanup()
    return wrapped
