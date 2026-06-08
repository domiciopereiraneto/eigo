"""Shared lightweight typing helpers for EIGO.

Module:
    eigo.utils.typing

Purpose:
    Define small, dependency-light type aliases used across the EIGO package.
    These aliases keep public interfaces readable without forcing central
    modules to import heavy optional libraries such as torch, PIL, NumPy, or
    diffusers. The aliases cover paths, JSON-compatible metadata, configs,
    metric dictionaries, objective functions, optimization candidates, and the
    public string literals used by configuration files.

Design Notes:
    Domain objects such as optimizer results, metric reports, and model adapter
    base classes should live in their respective ``base.py`` modules. This file
    should remain small to avoid circular imports across the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence, TypeAlias


# Accept plain strings in configs and Path objects inside Python code.
PathLike: TypeAlias = str | Path

# JSON-compatible values are useful for configs, manifests, and metadata files.
# They deliberately exclude arbitrary Python objects so saved run records remain
# portable across machines and Python versions.
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
JSONDict: TypeAlias = dict[str, JSONValue]
ConfigMapping: TypeAlias = Mapping[str, JSONValue]
MutableConfigMapping: TypeAlias = MutableMapping[str, JSONValue]

# Public names accepted by the configuration layer. Concrete implementations
# are resolved later by src/eigo/registry.py.
ModelBackend: TypeAlias = Literal[
    "auto",
    "sdxl",
    "flux",
    "pixart",
    "lcm",
    "sana",
    "sana_sprint",
]

# Variable spaces define what the optimizer manipulates. Keeping these names
# explicit makes experiment configs easier to validate and reproduce.
VariableKind: TypeAlias = Literal[
    "prompt_embeddings",
    "tokenized_prompt",
    "latent_noise",
    "hybrid",
]

# Optimizer names are intentionally high level; optimizer-specific parameters
# belong in typed config objects rather than in this shared typing module.
OptimizerKind: TypeAlias = Literal[
    "adam",
    "cmaes",
    "genetic",
    "random_search",
]

# Objective direction is explicit so optimizers can support either convention
# without assuming that larger values are always better.
ObjectiveDirection: TypeAlias = Literal["maximize", "minimize"]

# Device and dtype values are parsed in utils/devices.py. ``DeviceLike`` accepts
# integers for CUDA indices and strings such as "cpu", "cuda", or "cuda:0".
DeviceLike: TypeAlias = str | int
TorchDTypeName: TypeAlias = Literal["auto", "float32", "float16", "bfloat16"]

# Metric dictionaries should use stable metric names because they are written
# into CSV files, manifests, and optimization traces.
MetricName: TypeAlias = str
MetricScores: TypeAlias = Mapping[MetricName, float]
MutableMetricScores: TypeAlias = MutableMapping[MetricName, float]
Metadata: TypeAlias = Mapping[str, JSONValue]
MutableMetadata: TypeAlias = MutableMapping[str, JSONValue]

# Candidate is deliberately broad. A candidate may be a tensor, NumPy array,
# discrete token sequence, dictionary of variable groups, or another structure
# defined by a concrete VariableSpace.
Candidate: TypeAlias = Any
ObjectiveValue: TypeAlias = float
ObjectiveFn: TypeAlias = Callable[[Candidate], ObjectiveValue]

# These broad aliases avoid importing heavy optional dependencies such as
# torch, PIL, or NumPy from shared typing code. Concrete modules can use more
# precise types locally after importing the libraries they need.
TensorLike: TypeAlias = Any
ImageLike: TypeAlias = Any
RandomGeneratorLike: TypeAlias = Any
