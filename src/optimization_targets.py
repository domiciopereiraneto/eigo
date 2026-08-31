import numpy as np
import torch


PROMPT_EMBEDDINGS = "prompt_embeddings"
LATENT_NOISE = "latent_noise"
NOISE_EMBEDDINGS_FLAT = "noise_embeddings_flat"
NOISE_EMBEDDINGS_CC = "noise_embeddings_cc"


def resolve_optimization_target(parameters):
    target = str(parameters.get("optimization_target", PROMPT_EMBEDDINGS)).lower().replace("-", "_")
    aliases = {
        "prompt": PROMPT_EMBEDDINGS,
        "prompts": PROMPT_EMBEDDINGS,
        "embedding": PROMPT_EMBEDDINGS,
        "embeddings": PROMPT_EMBEDDINGS,
        "text_embeddings": PROMPT_EMBEDDINGS,
        "prompt_embeddings": PROMPT_EMBEDDINGS,
        "latent": LATENT_NOISE,
        "latents": LATENT_NOISE,
        "noise": LATENT_NOISE,
        "latent_noise": LATENT_NOISE,
        "noise_latent": LATENT_NOISE,
        "noise_embeddings_flat": NOISE_EMBEDDINGS_FLAT,
        "noise_embeddings": NOISE_EMBEDDINGS_FLAT,
        "latent_embeddings_flat": NOISE_EMBEDDINGS_FLAT,
        "latent_embeddings": NOISE_EMBEDDINGS_FLAT,
        "embeddings_noise_flat": NOISE_EMBEDDINGS_FLAT,
        "embeddings_noise": NOISE_EMBEDDINGS_FLAT,
        "joint": NOISE_EMBEDDINGS_FLAT,
        "joint_flat": NOISE_EMBEDDINGS_FLAT,
        "noise_embeddings_cc": NOISE_EMBEDDINGS_CC,
        "latent_embeddings_cc": NOISE_EMBEDDINGS_CC,
        "embeddings_noise_cc": NOISE_EMBEDDINGS_CC,
        "joint_cc": NOISE_EMBEDDINGS_CC,
    }
    if target not in aliases:
        valid = ", ".join(sorted({
            PROMPT_EMBEDDINGS,
            LATENT_NOISE,
            NOISE_EMBEDDINGS_FLAT,
            NOISE_EMBEDDINGS_CC,
        }))
        raise ValueError(f"Invalid optimization_target '{target}'. Expected one of: {valid}.")
    return aliases[target]


def build_target_state(target_name, prompt_embeds, pooled_prompt_embeds, latents=None):
    state = {
        "target_name": target_name,
        "prompt_shape": tuple(prompt_embeds.shape),
        "pooled_shape": tuple(pooled_prompt_embeds.shape),
        "latent_shape": None if latents is None else tuple(latents.shape),
        "prompt_embeds": prompt_embeds.detach().clone().to(torch.float32),
        "pooled_prompt_embeds": pooled_prompt_embeds.detach().clone().to(torch.float32),
        "latents": None if latents is None else latents.detach().clone().to(torch.float32),
    }

    if target_name == PROMPT_EMBEDDINGS:
        state["initial_vector"] = torch.cat([
            state["prompt_embeds"].flatten(),
            state["pooled_prompt_embeds"].flatten(),
        ]).cpu().numpy()
    elif target_name == LATENT_NOISE:
        if state["latents"] is None:
            raise ValueError("Latent noise optimization requires initial latents.")
        state["initial_vector"] = state["latents"].flatten().cpu().numpy()
    elif target_name == NOISE_EMBEDDINGS_FLAT:
        if state["latents"] is None:
            raise ValueError("Joint noise/embedding optimization requires initial latents.")
        state["initial_vector"] = torch.cat([
            state["prompt_embeds"].flatten(),
            state["pooled_prompt_embeds"].flatten(),
            state["latents"].flatten(),
        ]).cpu().numpy()
    elif target_name == NOISE_EMBEDDINGS_CC:
        if state["latents"] is None:
            raise ValueError("CC noise/embedding optimization requires initial latents.")
        state["initial_embedding_vector"] = torch.cat([
            state["prompt_embeds"].flatten(),
            state["pooled_prompt_embeds"].flatten(),
        ]).cpu().numpy()
        state["initial_noise_vector"] = state["latents"].flatten().cpu().numpy()
        state["initial_vector"] = cc_components_to_vector(
            state["initial_embedding_vector"],
            state["initial_noise_vector"],
        )
    else:
        raise ValueError(f"Unsupported optimization target: {target_name}")

    return state


def tensors_from_vector(vector, state, device):
    target_name = state["target_name"]
    if target_name == PROMPT_EMBEDDINGS:
        prompt_size = int(np.prod(state["prompt_shape"]))
        pooled_size = int(np.prod(state["pooled_shape"]))
        prompt_embeds = torch.tensor(vector[:prompt_size], dtype=torch.float32, device=device).view(state["prompt_shape"])
        pooled_prompt_embeds = torch.tensor(vector[prompt_size:prompt_size + pooled_size], dtype=torch.float32, device=device).view(state["pooled_shape"])
        latents = state["latents"]
        if latents is not None:
            latents = latents.to(device=device, dtype=torch.float32)
    elif target_name == LATENT_NOISE:
        prompt_embeds = state["prompt_embeds"].to(device=device, dtype=torch.float32)
        pooled_prompt_embeds = state["pooled_prompt_embeds"].to(device=device, dtype=torch.float32)
        latents = torch.tensor(vector, dtype=torch.float32, device=device).view(state["latent_shape"])
    elif target_name in {NOISE_EMBEDDINGS_FLAT, NOISE_EMBEDDINGS_CC}:
        prompt_size = int(np.prod(state["prompt_shape"]))
        pooled_size = int(np.prod(state["pooled_shape"]))
        latent_size = int(np.prod(state["latent_shape"]))
        vector = cc_components_to_vector(*vector) if is_cc_component_vector(vector) else vector
        prompt_end = prompt_size
        pooled_end = prompt_end + pooled_size
        latent_end = pooled_end + latent_size
        if len(vector) != latent_end:
            raise ValueError(
                f"Expected vector length {latent_end} for {target_name}, got {len(vector)}."
            )
        prompt_embeds = torch.tensor(vector[:prompt_end], dtype=torch.float32, device=device).view(state["prompt_shape"])
        pooled_prompt_embeds = torch.tensor(vector[prompt_end:pooled_end], dtype=torch.float32, device=device).view(state["pooled_shape"])
        latents = torch.tensor(vector[pooled_end:latent_end], dtype=torch.float32, device=device).view(state["latent_shape"])
    else:
        raise ValueError(f"Unsupported optimization target: {target_name}")
    return prompt_embeds, pooled_prompt_embeds, latents


def adam_parameters_from_state(state):
    if state["target_name"] == PROMPT_EMBEDDINGS:
        trainable = [
            torch.nn.Parameter(state["prompt_embeds"].clone()),
            torch.nn.Parameter(state["pooled_prompt_embeds"].clone()),
        ]
        fixed = {
            "prompt_embeds": None,
            "pooled_prompt_embeds": None,
            "latents": state["latents"],
        }
    elif state["target_name"] == LATENT_NOISE:
        trainable = [torch.nn.Parameter(state["latents"].clone())]
        fixed = {
            "prompt_embeds": state["prompt_embeds"],
            "pooled_prompt_embeds": state["pooled_prompt_embeds"],
            "latents": None,
        }
    elif state["target_name"] in {NOISE_EMBEDDINGS_FLAT, NOISE_EMBEDDINGS_CC}:
        trainable = [
            torch.nn.Parameter(state["prompt_embeds"].clone()),
            torch.nn.Parameter(state["pooled_prompt_embeds"].clone()),
            torch.nn.Parameter(state["latents"].clone()),
        ]
        fixed = {
            "prompt_embeds": None,
            "pooled_prompt_embeds": None,
            "latents": None,
        }
    else:
        raise ValueError(f"Unsupported optimization target: {state['target_name']}")
    return trainable, fixed


def adam_tensors(trainable, fixed, target_name):
    if target_name == PROMPT_EMBEDDINGS:
        return trainable[0], trainable[1], fixed["latents"]
    if target_name == LATENT_NOISE:
        return fixed["prompt_embeds"], fixed["pooled_prompt_embeds"], trainable[0]
    if target_name in {NOISE_EMBEDDINGS_FLAT, NOISE_EMBEDDINGS_CC}:
        return trainable[0], trainable[1], trainable[2]
    raise ValueError(f"Unsupported optimization target: {target_name}")


def clone_best_adam_tensors(trainable, fixed, target_name):
    prompt_embeds, pooled_prompt_embeds, latents = adam_tensors(trainable, fixed, target_name)
    return [
        prompt_embeds.detach().clone(),
        pooled_prompt_embeds.detach().clone(),
        None if latents is None else latents.detach().clone(),
    ]


def is_cc_component_vector(vector):
    return (
        isinstance(vector, (tuple, list))
        and len(vector) == 2
        and not torch.is_tensor(vector[0])
        and not torch.is_tensor(vector[1])
    )


def cc_components_to_vector(embedding_vector, noise_vector):
    return np.concatenate([
        np.asarray(embedding_vector, dtype=np.float64).reshape(-1),
        np.asarray(noise_vector, dtype=np.float64).reshape(-1),
    ])
