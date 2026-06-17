import numpy as np
import torch


PROMPT_EMBEDDINGS = "prompt_embeddings"
LATENT_NOISE = "latent_noise"


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
    }
    if target not in aliases:
        valid = ", ".join(sorted({PROMPT_EMBEDDINGS, LATENT_NOISE}))
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
    else:
        raise ValueError(f"Unsupported optimization target: {target_name}")

    return state


def tensors_from_vector(vector, state, device):
    target_name = state["target_name"]
    if target_name == PROMPT_EMBEDDINGS:
        split = int(np.prod(state["prompt_shape"]))
        prompt_embeds = torch.tensor(vector[:split], dtype=torch.float32, device=device).view(state["prompt_shape"])
        pooled_prompt_embeds = torch.tensor(vector[split:], dtype=torch.float32, device=device).view(state["pooled_shape"])
        latents = state["latents"]
        if latents is not None:
            latents = latents.to(device=device, dtype=torch.float32)
    elif target_name == LATENT_NOISE:
        prompt_embeds = state["prompt_embeds"].to(device=device, dtype=torch.float32)
        pooled_prompt_embeds = state["pooled_prompt_embeds"].to(device=device, dtype=torch.float32)
        latents = torch.tensor(vector, dtype=torch.float32, device=device).view(state["latent_shape"])
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
    else:
        raise ValueError(f"Unsupported optimization target: {state['target_name']}")
    return trainable, fixed


def adam_tensors(trainable, fixed, target_name):
    if target_name == PROMPT_EMBEDDINGS:
        return trainable[0], trainable[1], fixed["latents"]
    if target_name == LATENT_NOISE:
        return fixed["prompt_embeds"], fixed["pooled_prompt_embeds"], trainable[0]
    raise ValueError(f"Unsupported optimization target: {target_name}")


def clone_best_adam_tensors(trainable, fixed, target_name):
    prompt_embeds, pooled_prompt_embeds, latents = adam_tensors(trainable, fixed, target_name)
    return [
        prompt_embeds.detach().clone(),
        pooled_prompt_embeds.detach().clone(),
        None if latents is None else latents.detach().clone(),
    ]
