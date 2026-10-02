"""Compact optimizer snapshots taken only between completed ask/tell steps.

Never mutate the running optimizer while preparing a snapshot. CMA's sampler,
paths, pending injections and stopping history are deliberately retained: resetting
any of these can change the next population even when the mean is unchanged.
"""
import copy

import numpy as np


SNES_FIELDS = (
    "mean", "sigma", "pop_size", "max_generations", "eta_mu", "eta_sigma",
    "generation", "result",
)
COSYNE_FIELDS = (
    "population", "pop_size", "max_generations", "mutation_probability",
    "mutation_scale", "parent_count", "offspring_count", "generation", "result",
)
# EIGO checkpoints after tell(), and does not use external solution injection,
# surrogate models or logger histories. These are caches, not adaptation state.
CMA_TRANSIENT_FIELDS = {
    "archive", "sent_solutions", "ary", "pop", "pop_sorted", "logger",
    "_stopdict", "inputargs", "mean_old_old",
}


def pack_population_optimizer(optimizer):
    import cma

    if isinstance(optimizer, cma.CMAEvolutionStrategy):
        if not optimizer._flgtelldone or len(optimizer.sent_solutions):
            raise ValueError("CMA checkpoints require a completed ask/tell generation.")
        state = {key: value for key, value in vars(optimizer).items()
                 if key not in CMA_TRANSIENT_FIELDS}
        # Do not pickle a second copy of NumPy's global RandomState through a
        # bound randn method. It must share the restored global RNG on resume.
        global_rng = getattr(optimizer.sm.randn, "__self__", None) is np.random.mtrand._rand
        if global_rng:
            state["sm"] = copy.copy(optimizer.sm)
            state["sm"].randn = None
            state["opts"] = copy.copy(optimizer.opts)
            state["opts"]["randn"] = None
        return {"kind": "cmaes", "library_version": cma.__version__,
                "global_rng": global_rng, "state": state}

    name = type(optimizer).__name__
    if name == "SeparableNaturalEvolutionStrategy":
        if optimizer._noise is not None:
            raise ValueError("SNES checkpoints require a completed ask/tell generation.")
        kind, fields = "snes", SNES_FIELDS
    elif name == "CooperativeSynapseNeuroevolution":
        if optimizer._asked:
            raise ValueError("CoSyNE checkpoints require a completed ask/tell generation.")
        kind, fields = "cosyne", COSYNE_FIELDS
    else:
        raise TypeError(f"Unsupported checkpoint optimizer: {name}")
    return {"kind": kind, "state": {key: getattr(optimizer, key) for key in fields},
            "rng": optimizer.rng.bit_generator.state}


def unpack_population_optimizer(snapshot):
    kind = snapshot["kind"]
    if kind == "cmaes":
        import cma
        from cma.evolution_strategy import _CMASolutionDict, _CMAStopDict

        if snapshot["library_version"] != cma.__version__:
            raise ValueError("Resume CMA checkpoints with the same pycma version used to save them.")
        optimizer = cma.CMAEvolutionStrategy.__new__(cma.CMAEvolutionStrategy)
        optimizer.__dict__.update(snapshot["state"])
        optimizer.archive = _CMASolutionDict()
        optimizer.sent_solutions = _CMASolutionDict()
        optimizer._stopdict = _CMAStopDict()
        optimizer.ary = []
        optimizer.pop = []
        optimizer.pop_sorted = None
        optimizer.inputargs = {"x0": optimizer.x0, "sigma0": optimizer.sigma0,
                               "inopts": optimizer.inopts}
        if snapshot["global_rng"]:
            optimizer.sm.randn = np.random.randn
            optimizer.opts["randn"] = np.random.randn
        optimizer.logger = cma.CMADataLogger(
            optimizer.opts["verb_filenameprefix"], modulo=optimizer.opts["verb_log"],
            expensive_modulo=optimizer.opts["verb_log_expensive"],
        ).register(optimizer)
        return optimizer

    from eigo import SeparableNaturalEvolutionStrategy, CooperativeSynapseNeuroevolution
    classes = {"snes": SeparableNaturalEvolutionStrategy, "cosyne": CooperativeSynapseNeuroevolution}
    cls = classes[kind]
    optimizer = cls.__new__(cls)
    optimizer.__dict__.update(snapshot["state"])
    optimizer.rng = unpack_generator(snapshot["rng"])
    if kind == "snes":
        optimizer._noise = None
    else:
        optimizer._asked = False
    return optimizer


def unpack_generator(state):
    generator = np.random.Generator(getattr(np.random, state["bit_generator"])(0))
    generator.bit_generator.state = state
    return generator


# Scalar sample history is retained once. The reporting lists are derived from it.
SAMPLE_HISTORY_COLUMNS = {
    "fitness_history": "fitness", "aesthetic_history": "aesthetic_score",
    "clip_history": "clip_score", "image_reward_history": "image_reward_score",
    "hpsv2_history": "hpsv2_score", "pickscore_history": "pickscore_score",
    "jpeg_size_history": "jpeg_size_kb", "time_list": "elapsed_time",
    "peak_vram_mb_list": "peak_vram_mb",
}


def pack_run_state(state):
    """Return an independent mapping without redundant initialization/cached data."""
    packed = dict(state)
    if "target_state" in packed:
        target = packed["target_state"]
        keys = {"target_name", "prompt_shape", "pooled_shape", "latent_shape"}
        # Optimized components are reconstructed from the candidate vector.
        # Only fixed components need their actual initial tensors on disk.
        if target["target_name"] == "prompt_embeddings":
            keys.add("latents")
        elif target["target_name"] == "latent_noise":
            keys.update(("prompt_embeds", "pooled_prompt_embeds"))
        packed["target_state"] = {key: target[key] for key in keys}
    for key in ("es", "embedding_es", "noise_es"):
        if key in packed:
            packed[key] = pack_population_optimizer(packed[key])
    if "rng" in packed:
        packed["rng"] = packed["rng"].bit_generator.state
    if "optimizer" in packed:
        # Save AdamW moments, hyperparameters, and step counters, not the optimizer
        # object, hooks, or Parameter references. Rebind on load to the saved values.
        packed["optimizer"] = packed["optimizer"].state_dict()
        packed["trainable_params"] = [parameter.detach() for parameter in packed["trainable_params"]]
        scaler = packed["grad_scaler"]
        packed["grad_scaler"] = {"enabled": scaler.is_enabled(), "state": scaler.state_dict()}
        packed.pop("target_state", None)  # fixed_target_tensors already contains what's needed
        packed["best_target_tensors"] = [
            value if packed["fixed_target_tensors"][key] is None else None
            for key, value in zip(("prompt_embeds", "pooled_prompt_embeds", "latents"),
                                  packed["best_target_tensors"])
        ]
    if "sample_rows" in packed:
        for key in (*SAMPLE_HISTORY_COLUMNS, "sample_times"):
            packed.pop(key, None)
    return packed


def unpack_run_state(state):
    import torch

    restored = dict(state)
    for key in ("es", "embedding_es", "noise_es"):
        if key in restored:
            restored[key] = unpack_population_optimizer(restored[key])
    if "rng" in restored:
        restored["rng"] = unpack_generator(restored["rng"])
    if "optimizer" in restored:
        params = [torch.nn.Parameter(value) for value in restored["trainable_params"]]
        optimizer = torch.optim.AdamW(params)
        optimizer.load_state_dict(restored["optimizer"])
        restored["trainable_params"] = params
        restored["optimizer"] = optimizer
        scaler_state = restored["grad_scaler"]
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_state["enabled"])
        scaler.load_state_dict(scaler_state["state"])
        restored["grad_scaler"] = scaler
        restored["best_target_tensors"] = [
            value if value is not None else restored["fixed_target_tensors"][key]
            for key, value in zip(("prompt_embeds", "pooled_prompt_embeds", "latents"),
                                  restored["best_target_tensors"])
        ]
    if "sample_rows" in restored:
        for key, column in SAMPLE_HISTORY_COLUMNS.items():
            restored[key] = [row[column] for row in restored["sample_rows"]]
        restored["sample_times"] = restored["time_list"][1:]
    return restored
