"""
This script contains the Eigo class, an optimization engine for diffusion-based image
generation guided by aesthetic, prompt-alignment, and ImageReward scores. The engine
supports CMA-ES (standard, sep-CMA-ES and VD-CMA), GA, Adam, and random sampling.
"""

import sys
import os
import shutil
import gc
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    PixArtAlphaPipeline,
    StableDiffusionXLPipeline,
)
try:
    from diffusers import SanaPipeline, SanaSprintPipeline
except ImportError:
    SanaPipeline = None
    SanaSprintPipeline = None
import random
from PIL import Image
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import csv
from pptx import Presentation
from pptx.util import Inches
import cma
from cma.restricted_gaussian_sampler import GaussVDSampler 
from datasets import load_dataset
import clip
import argparse
import re
import ast
import inspect
from contextlib import nullcontext
from pathlib import Path
from transformers import AutoModel, AutoProcessor

class Eigo:
    _MODEL_CACHE = {}
    _ACTIVE_CACHE_KEY = None

    @classmethod
    def clear_model_cache(cls):
        for cache_entry in cls._MODEL_CACHE.values():
            cache_entry["pipe"] = None
            cache_entry["clip_model"] = None
            cache_entry["clip_preprocess"] = None
            cache_entry["aesthetic_model"] = None
            cache_entry["image_reward_model"] = None
            cache_entry["hpsv2_model"] = None
            cache_entry["hpsv2_preprocess"] = None
            cache_entry["hpsv2_tokenizer"] = None
            cache_entry["pickscore_model"] = None
            cache_entry["pickscore_processor"] = None
        cls._MODEL_CACHE.clear()
        cls._ACTIVE_CACHE_KEY = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def __init__(self, config_parameters):
        cls = type(self)
        self.parameters = config_parameters
        self.model_backend = self._resolve_model_backend(config_parameters)
        self.guidance_scale = float(config_parameters.get("guidance_scale", 0.0))
        self.lcm_origin_steps = int(config_parameters.get("lcm_origin_steps", 50))
        if self.lcm_origin_steps <= 0:
            raise ValueError("lcm_origin_steps must be a positive integer.")
        self.max_sequence_length = int(config_parameters.get("max_sequence_length", 512))
        self.model_dtype = self._resolve_model_dtype(config_parameters)
        self.use_multi_gpu = bool(config_parameters.get("use_multi_gpu", False))
        self.pipeline_device_map = config_parameters.get("pipeline_device_map", "balanced")
        self.max_memory = self._resolve_max_memory(config_parameters.get("max_memory", None))
        self.enable_attention_slicing = bool(config_parameters.get("enable_attention_slicing", False))
        self.enable_vae_slicing = bool(config_parameters.get("enable_vae_slicing", False))
        self.enable_vae_tiling = bool(config_parameters.get("enable_vae_tiling", False))
        self.enable_gradient_checkpointing = bool(config_parameters.get("enable_gradient_checkpointing", False))
        self._active_prompt_attention_mask = None
        self._active_negative_prompt_embeds = None
        self._active_negative_prompt_attention_mask = None
        self.aesthetic_score_weight = float(
            config_parameters.get("aesthetic_score_weight", config_parameters.get("alpha", 0.0))
        )
        self.clip_score_weight = float(
            config_parameters.get("clip_score_weight", config_parameters.get("beta", 0.0))
        )
        self.image_reward_score_weight = float(
            config_parameters.get("image_reward_score_weight", config_parameters.get("gamma", 0.0))
        )
        self.hpsv2_score_weight = float(config_parameters.get("hpsv2_score_weight", 0.0))
        self.pickscore_score_weight = float(config_parameters.get("pickscore_score_weight", 0.0))
        self.evaluate_zero_weight_metrics = bool(config_parameters.get("evaluate_zero_weight_metrics", False))
        self._metric_scales = {
            "aesthetic_score": float(config_parameters.get("max_aesthetic_score", 1.0)),
            "clip_score": float(config_parameters.get("max_clip_score", 1.0)),
            "image_reward_score": float(config_parameters.get("max_image_reward_score", 1.0)),
            "hpsv2_score": float(config_parameters.get("max_hpsv2_score", 1.0)),
            "pickscore_score": float(config_parameters.get("max_pickscore_score", 1.0)),
        }
        image_reward_model_name = config_parameters.get(
            "image_reward_model_name",
            config_parameters.get("image_reward_model", "ImageReward-v1.0"),
        )
        self.image_reward_model_name = (
            None if image_reward_model_name is None else str(image_reward_model_name)
        )
        self.use_image_reward = (
            self.image_reward_model_name is not None
            and self._should_evaluate_metric("image_reward_score")
        )
        hpsv2_version = config_parameters.get("hpsv2_version", "v2.1")
        self.hpsv2_version = None if hpsv2_version is None else str(hpsv2_version)
        self.use_hpsv2 = (
            self.hpsv2_version is not None
            and self._should_evaluate_metric("hpsv2_score")
        )
        pickscore_processor_name = config_parameters.get(
            "pickscore_processor_name",
            config_parameters.get("pickscore_processor", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
        )
        self.pickscore_processor_name = None if pickscore_processor_name is None else str(pickscore_processor_name)
        pickscore_model_name = config_parameters.get(
            "pickscore_model_name",
            config_parameters.get("pickscore_model", "yuvalkirstain/PickScore_v1"),
        )
        self.pickscore_model_name = None if pickscore_model_name is None else str(pickscore_model_name)
        self.use_pickscore = (
            self.pickscore_model_name is not None
            and self.pickscore_processor_name is not None
            and self._should_evaluate_metric("pickscore_score")
        )

        if config_parameters["predictor"] == 0:
            predictor_name = 'simulacra'
        elif config_parameters["predictor"] == 1:
            predictor_name = 'laionv1'
        elif config_parameters["predictor"] == 2:
            predictor_name = 'laionv2'
        else:
            raise ValueError("Invalid predictor option.")

        if config_parameters["optimization_method"] == "adam":
            method_save_name = "adam"
        elif config_parameters["optimization_method"] == "cmaes":
            if config_parameters["cmaes_variant"] == "cmaes":
                method_save_name = "cmaes"
            elif config_parameters["cmaes_variant"] == "sep":
                method_save_name = "sepcmaes"
            elif config_parameters["cmaes_variant"] == "vd":
                method_save_name = "vdcmae"
            else:
                raise ValueError(f"Unknown CMA-ES variant: {config_parameters['cmaes_variant']}")
        elif config_parameters["optimization_method"] == "ga":
            method_save_name = "ga"
        elif config_parameters["optimization_method"] == "random_sampler":
            method_save_name = "randomsampler"
        else:
            raise ValueError(f"Unknown optimization method: {config_parameters['optimization_method']}")

        model_tag = self._model_id_tag(config_parameters["model_id"])
        self.OUTPUT_FOLDER = (
            f"{config_parameters['results_folder']}/"
            f"{method_save_name}_clip_{predictor_name}_{self.model_backend}_{model_tag}_"
            f"{config_parameters['seed']}_"
            f"aesw{int(self.aesthetic_score_weight*100)}_"
            f"clipw{int(self.clip_score_weight*100)}_"
            f"irw{int(self.image_reward_score_weight*100)}_"
            f"hpsw{int(self.hpsv2_score_weight*100)}_"
            f"psw{int(self.pickscore_score_weight*100)}"
        )

        # Save the selected prompts and their categories to a text file in the results folder
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)

        # Check if a GPU is available and if not, use the CPU
        if config_parameters["cuda"] == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda:" + str(config_parameters["cuda"]) if torch.cuda.is_available() else "cpu"

        cache_key = (
            config_parameters["model_id"],
            self.model_backend,
            str(self.model_dtype),
            self.device,
            self.use_multi_gpu,
            str(self.pipeline_device_map),
            str(self.max_memory),
            config_parameters["predictor"],
            self.use_image_reward,
            self.image_reward_model_name if self.use_image_reward else None,
            self.use_hpsv2,
            self.hpsv2_version if self.use_hpsv2 else None,
            self.use_pickscore,
            self.pickscore_processor_name if self.use_pickscore else None,
            self.pickscore_model_name if self.use_pickscore else None,
            self._should_evaluate_metric("aesthetic_score"),
            self._should_evaluate_metric("clip_score"),
            self.evaluate_zero_weight_metrics,
        )

        if cls._ACTIVE_CACHE_KEY is not None and cls._ACTIVE_CACHE_KEY != cache_key:
            self.clear_model_cache()

        if cache_key not in cls._MODEL_CACHE:
            pipe, is_sharded = self._load_pipeline(
                model_id=config_parameters["model_id"],
                use_safetensors=config_parameters.get("use_safetensors", True),
            )
            self._configure_pipeline_output_options(pipe)
            if not is_sharded:
                pipe = pipe.to(self.device)
            pipe.set_progress_bar_config(disable=True)
            self._freeze_module_params(pipe)
            self._configure_pipeline_memory_options(pipe)

            clip_model_name = "ViT-L/14"
            clip_model = None
            clip_preprocess = None
            if self._should_evaluate_metric("clip_score"):
                clip_model, clip_preprocess = clip.load(clip_model_name, device=self.device)
                self._freeze_module_params(clip_model)

            aesthetic_model = None
            if config_parameters["predictor"] == 0:
                model_name = "SAM"
                if self._should_evaluate_metric("aesthetic_score"):
                    from aesthetic_evaluation.src import simulacra_rank_image
                    aesthetic_model = simulacra_rank_image.SimulacraAesthetic(self.device)
            elif config_parameters["predictor"] == 1:
                model_name = "LAIONV1"
                if self._should_evaluate_metric("aesthetic_score"):
                    from aesthetic_evaluation.src import laion_rank_image
                    aesthetic_model = laion_rank_image.LAIONAesthetic(self.device, clip_model=clip_model_name)
            elif config_parameters["predictor"] == 2:
                model_name = "LAIONV2"
                if self._should_evaluate_metric("aesthetic_score"):
                    from aesthetic_evaluation.src import laion_v2_rank_image
                    aesthetic_model = laion_v2_rank_image.LAIONV2Aesthetic(self.device, clip_model=clip_model_name)
            else:
                raise ValueError("Invalid predictor option.")

            image_reward_model = None
            if self.use_image_reward:
                try:
                    import ImageReward as RM
                except ImportError as exc:
                    raise ImportError(
                        "ImageReward scoring requires the 'image-reward' package when an ImageReward model is configured."
                    ) from exc
                image_reward_model = RM.load(self.image_reward_model_name, device=self.device)
                self._freeze_module_params(image_reward_model)

            hpsv2_model = None
            hpsv2_preprocess = None
            hpsv2_tokenizer = None
            if self.use_hpsv2:
                self._ensure_hpsv2_tokenizer_assets()
                try:
                    import hpsv2
                    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
                    from hpsv2.utils import hps_version_map
                    import huggingface_hub
                except ImportError as exc:
                    raise ImportError(
                        "HPSv2 scoring requires the 'hpsv2' package when hpsv2_score_weight != 0."
                    ) from exc

                checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[self.hpsv2_version])
                hpsv2_model, _, hpsv2_preprocess = create_model_and_transforms(
                    'ViT-H-14',
                    'laion2B-s32B-b79K',
                    precision='amp',
                    device=self.device,
                    jit=False,
                    force_quick_gelu=False,
                    force_custom_text=False,
                    force_patch_dropout=False,
                    force_image_size=None,
                    pretrained_image=False,
                    image_mean=None,
                    image_std=None,
                    light_augmentation=True,
                    aug_cfg={},
                    output_dict=True,
                    with_score_predictor=False,
                    with_region_predictor=False,
                )
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                hpsv2_model.load_state_dict(checkpoint['state_dict'])
                hpsv2_model = hpsv2_model.to(self.device).eval()
                self._freeze_module_params(hpsv2_model)
                hpsv2_tokenizer = get_tokenizer('ViT-H-14')

            pickscore_model = None
            pickscore_processor = None
            if self.use_pickscore:
                pickscore_processor = AutoProcessor.from_pretrained(self.pickscore_processor_name)
                pickscore_model = AutoModel.from_pretrained(self.pickscore_model_name).eval().to(self.device)
                self._freeze_module_params(pickscore_model)

            cls._MODEL_CACHE[cache_key] = {
                "pipe": pipe,
                "is_sharded": is_sharded,
                "clip_model": clip_model,
                "clip_preprocess": clip_preprocess,
                "aesthetic_model": aesthetic_model,
                "image_reward_model": image_reward_model,
                "hpsv2_model": hpsv2_model,
                "hpsv2_preprocess": hpsv2_preprocess,
                "hpsv2_tokenizer": hpsv2_tokenizer,
                "pickscore_model": pickscore_model,
                "pickscore_processor": pickscore_processor,
                "model_name": model_name,
            }
            cls._ACTIVE_CACHE_KEY = cache_key

        cached_models = cls._MODEL_CACHE[cache_key]
        cls._ACTIVE_CACHE_KEY = cache_key
        self.pipe = cached_models["pipe"]
        self.pipe_is_sharded = cached_models.get("is_sharded", False)
        self.clip_model = cached_models["clip_model"]
        self.clip_preprocess = cached_models["clip_preprocess"]
        self.aesthetic_model = cached_models["aesthetic_model"]
        self.image_reward_model = cached_models.get("image_reward_model")
        self.hpsv2_model = cached_models.get("hpsv2_model")
        self.hpsv2_preprocess = cached_models.get("hpsv2_preprocess")
        self.hpsv2_tokenizer = cached_models.get("hpsv2_tokenizer")
        self.pickscore_model = cached_models.get("pickscore_model")
        self.pickscore_processor = cached_models.get("pickscore_processor")
        self.model_name = cached_models["model_name"]
        wrapped_call = getattr(self.pipe.__class__.__call__, "__wrapped__", None)
        self.call_with_grad = wrapped_call.__get__(self.pipe, self.pipe.__class__) if wrapped_call is not None else self.pipe.__call__

        # Differentiable CLIP score evaluation for Adam
        self._CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
        self._CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)

    @staticmethod
    def _model_id_tag(model_id):
        return re.sub(r"[^a-z0-9]+", "", model_id.lower().split("/")[-1])

    def _should_evaluate_metric(self, metric_name):
        if self.evaluate_zero_weight_metrics:
            return True
        metric_weights = {
            "aesthetic_score": self.aesthetic_score_weight,
            "clip_score": self.clip_score_weight,
            "image_reward_score": self.image_reward_score_weight,
            "hpsv2_score": self.hpsv2_score_weight,
            "pickscore_score": self.pickscore_score_weight,
        }
        return metric_weights.get(metric_name, 0.0) != 0.0

    @staticmethod
    def _resolve_model_backend(config_parameters):
        requested = str(config_parameters.get("model_backend", "auto")).lower().replace("-", "_")
        if requested in ("sdxl", "flux", "pixart", "lcm", "sana", "sana_sprint"):
            return requested
        if requested != "auto":
            raise ValueError(
                f"Invalid model_backend '{requested}'. Expected one of: auto, sdxl, flux, pixart, lcm, sana, sana_sprint."
            )
        model_id = config_parameters["model_id"].lower()
        if "lcm" in model_id or "latent-consistency" in model_id:
            return "lcm"
        if "pixart" in model_id:
            return "pixart"
        if "sana_sprint" in model_id or "sana-sprint" in model_id or "sanasprint" in model_id:
            return "sana_sprint"
        if "sana" in model_id:
            return "sana"
        return "flux" if "flux" in model_id else "sdxl"

    def _resolve_model_dtype(self, config_parameters):
        dtype_name = str(config_parameters.get("torch_dtype", "auto")).lower()
        if dtype_name == "auto":
            if self.model_backend in ("flux", "sana", "sana_sprint") and torch.cuda.is_available():
                return torch.bfloat16
            return torch.float32

        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(
                f"Invalid torch_dtype '{dtype_name}'. Expected one of: auto, float32, float16, bfloat16."
            )
        return dtype_map[dtype_name]

    @staticmethod
    def _freeze_module_params(module):
        if module is None:
            return
        parameters = getattr(module, "parameters", None)
        if callable(parameters):
            for p in parameters():
                p.requires_grad_(False)
            return

        components = getattr(module, "components", None)
        if isinstance(components, dict):
            for component in components.values():
                component_parameters = getattr(component, "parameters", None)
                if component is not None and callable(component_parameters):
                    for p in component_parameters():
                        p.requires_grad_(False)

    @staticmethod
    def _ensure_hpsv2_tokenizer_assets():
        site_packages = (
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        target_dir = site_packages / "hpsv2" / "src" / "open_clip"
        target_file = target_dir / "bpe_simple_vocab_16e6.txt.gz"
        if target_file.exists():
            return

        candidate_paths = [
            site_packages / "open_clip" / "bpe_simple_vocab_16e6.txt.gz",
            site_packages / "clip" / "bpe_simple_vocab_16e6.txt.gz",
            Path(__file__).resolve().parent / "aesthetic_evaluation" / "CLIP" / "clip" / "bpe_simple_vocab_16e6.txt.gz",
        ]
        source_file = next((path for path in candidate_paths if path.exists()), None)
        if source_file is None:
            raise FileNotFoundError(
                "HPSv2 tokenizer vocabulary file is missing and no fallback copy was found."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target_file)

    @staticmethod
    def _resolve_max_memory(max_memory_cfg):
        if max_memory_cfg is None:
            return None
        if isinstance(max_memory_cfg, dict):
            return max_memory_cfg
        if isinstance(max_memory_cfg, str):
            cfg = max_memory_cfg.strip()
            if not cfg:
                return None
            try:
                parsed = ast.literal_eval(cfg)
                if isinstance(parsed, dict):
                    return parsed
            except (SyntaxError, ValueError):
                pass
            raise ValueError(
                "max_memory must be a dict or a dict-like string, e.g. "
                "{\"cuda:0\": \"22GiB\", \"cuda:1\": \"22GiB\", \"cpu\": \"64GiB\"}."
            )
        raise ValueError("max_memory must be either dict, string, or null.")

    def _configure_pipeline_memory_options(self, pipe):
        if self.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if self.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if self.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        if self.enable_gradient_checkpointing:
            for module_name in ("transformer", "unet", "vae"):
                module = getattr(pipe, module_name, None)
                if module is not None and hasattr(module, "enable_gradient_checkpointing"):
                    module.enable_gradient_checkpointing()

    def _configure_pipeline_output_options(self, pipe):
        if self.model_backend == "lcm" and hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
            if hasattr(pipe, "requires_safety_checker"):
                pipe.requires_safety_checker = False

    def _load_pipeline(self, model_id, use_safetensors):
        common_kwargs = {
            "torch_dtype": self.model_dtype,
        }
        if use_safetensors is not None:
            common_kwargs["use_safetensors"] = bool(use_safetensors)
        is_sharded = False
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            common_kwargs["device_map"] = self.pipeline_device_map
            if self.max_memory is not None:
                common_kwargs["max_memory"] = self.max_memory
            is_sharded = True

        if self.model_backend == "flux":
            return FluxPipeline.from_pretrained(model_id, **common_kwargs), is_sharded
        if self.model_backend == "pixart":
            return PixArtAlphaPipeline.from_pretrained(model_id, **common_kwargs), is_sharded
        if self.model_backend == "lcm":
            return DiffusionPipeline.from_pretrained(model_id, **common_kwargs), is_sharded
        if self.model_backend == "sana":
            if SanaPipeline is None:
                raise ImportError("Sana backend requires a diffusers version that provides SanaPipeline.")
            return SanaPipeline.from_pretrained(model_id, **common_kwargs), is_sharded
        if self.model_backend == "sana_sprint":
            if SanaSprintPipeline is None:
                raise ImportError("Sana Sprint backend requires a diffusers version that provides SanaSprintPipeline.")
            return SanaSprintPipeline.from_pretrained(model_id, **common_kwargs), is_sharded
        if self.model_backend == "sdxl":
            return StableDiffusionXLPipeline.from_pretrained(model_id, **common_kwargs), is_sharded
        raise ValueError(f"Unsupported model backend: {self.model_backend}")

    def _pipeline_input_device(self):
        execution_device = getattr(self.pipe, "_execution_device", None)
        if execution_device is not None:
            return str(execution_device)
        return self.device

    def _encode_prompt_embeddings(self, prompt):
        encode_kwargs = {
            "prompt": prompt,
            "device": self._pipeline_input_device(),
            "num_images_per_prompt": 1,
        }

        if self.model_backend == "sdxl":
            encode_kwargs["negative_prompt"] = ""
            encode_kwargs["do_classifier_free_guidance"] = self.guidance_scale > 1.0
        elif self.model_backend == "lcm":
            encode_kwargs["negative_prompt"] = ""
            encode_kwargs["do_classifier_free_guidance"] = False
        elif self.model_backend == "flux":
            encode_kwargs["max_sequence_length"] = self.max_sequence_length
        elif self.model_backend == "pixart":
            encode_kwargs["negative_prompt"] = ""
            encode_kwargs["do_classifier_free_guidance"] = self.guidance_scale > 1.0
        elif self.model_backend == "sana":
            encode_kwargs["negative_prompt"] = ""
            encode_kwargs["do_classifier_free_guidance"] = self.guidance_scale > 1.0
            encode_kwargs["max_sequence_length"] = self.max_sequence_length
        elif self.model_backend == "sana_sprint":
            encode_kwargs["max_sequence_length"] = self.max_sequence_length

        encoded = self.pipe.encode_prompt(**encode_kwargs)
        if not isinstance(encoded, tuple):
            raise RuntimeError("Unexpected encode_prompt output type. Expected tuple.")

        if self.model_backend in ("pixart", "sana", "sana_sprint"):
            if len(encoded) < 2:
                raise RuntimeError(f"Unexpected {self.model_backend} encode_prompt output length.")
            self._active_prompt_attention_mask = encoded[1]
            self._active_negative_prompt_embeds = encoded[2] if self.model_backend == "sana" and len(encoded) >= 3 else None
            self._active_negative_prompt_attention_mask = encoded[3] if self.model_backend == "sana" and len(encoded) >= 4 else None
            # Keep the optimization loop shape contract unchanged with a small dummy tensor.
            aux = torch.zeros((1, 1), dtype=encoded[0].dtype, device=self.device)
            return encoded[0], aux

        if self.model_backend == "lcm":
            if len(encoded) < 1:
                raise RuntimeError("Unexpected LCM encode_prompt output length.")
            # LCM pipelines use a fixed empty unconditional prompt internally for CFG.
            # Keep the optimization loop shape contract unchanged with a small dummy tensor.
            aux = torch.zeros((1, 1), dtype=encoded[0].dtype, device=self.device)
            return encoded[0], aux

        if len(encoded) >= 4:
            # SDXL: prompt, negative_prompt, pooled_prompt, negative_pooled_prompt
            return encoded[0], encoded[2]
        if len(encoded) >= 2:
            # FLUX: prompt, pooled_prompt, text_ids
            return encoded[0], encoded[1]

        raise RuntimeError("Unexpected encode_prompt output length.")

    def _lcm_origin_steps_call_key(self):
        try:
            call_parameters = inspect.signature(self.pipe.__class__.__call__).parameters
        except (TypeError, ValueError):
            return "lcm_origin_steps"
        if "original_inference_steps" in call_parameters:
            return "original_inference_steps"
        return "lcm_origin_steps"

    def _build_generation_kwargs(self, prompt_embeds, pooled_prompt_embeds, generator):
        prompt_device = self._pipeline_input_device()
        prompt_embeds = prompt_embeds.to(device=prompt_device, dtype=self.model_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=prompt_device, dtype=self.model_dtype)
        effective_steps = int(self.parameters["num_inference_steps"])
        if self.model_backend == "pixart" and effective_steps == 1:
            # Some PixArt scheduler paths in diffusers expect >=2 steps.
            effective_steps = 2

        kwargs = {
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": effective_steps,
            "generator": generator,
            "height": self.parameters["height"],
            "width": self.parameters["width"],
            "output_type": "pt",
        }

        if self.model_backend in ("pixart", "sana", "sana_sprint"):
            if self._active_prompt_attention_mask is None:
                raise RuntimeError(
                    f"{self.model_backend} prompt attention mask is not initialized. Call _encode_prompt_embeddings first."
                )
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs["prompt_attention_mask"] = self._active_prompt_attention_mask.to(prompt_device)
            if self.model_backend == "sana":
                if self.guidance_scale > 1.0 and self._active_negative_prompt_embeds is None:
                    raise RuntimeError("Sana negative prompt embeddings are required when guidance_scale > 1.")
                if self._active_negative_prompt_embeds is not None:
                    kwargs["negative_prompt_embeds"] = self._active_negative_prompt_embeds.to(
                        device=prompt_device,
                        dtype=self.model_dtype,
                    )
                if self._active_negative_prompt_attention_mask is not None:
                    kwargs["negative_prompt_attention_mask"] = self._active_negative_prompt_attention_mask.to(prompt_device)
        elif self.model_backend == "lcm":
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs[self._lcm_origin_steps_call_key()] = self.lcm_origin_steps
        else:
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds

        if self.model_backend in ("flux", "sana", "sana_sprint"):
            kwargs["max_sequence_length"] = self.max_sequence_length

        return kwargs

    def _call_generation_pipeline(self, call_fn, generation_kwargs):
        try:
            return call_fn(**generation_kwargs)["images"]
        except UnboundLocalError as exc:
            if (
                self.model_backend in ("sana", "sana_sprint")
                and "local variable 'image' referenced before assignment" in str(exc)
            ):
                raise RuntimeError(
                    "Sana/Sana Sprint generation failed while decoding latents to an image. "
                    "Diffusers raised a misleading UnboundLocalError after catching a VAE "
                    "out-of-memory error internally. For Adam with Sana Sprint, reduce memory "
                    "pressure by using torch_dtype: auto or bfloat16, setting "
                    "evaluate_zero_weight_metrics: false, enabling gradient checkpointing, "
                    "lowering height/width, or enabling VAE tiling/slicing."
                ) from exc
            raise
        
    def generate_image_from_embeddings_cmaes(self, prompt_embeds, pooled_prompt_embeds, seed):
        generator = torch.Generator(device=self._pipeline_input_device()).manual_seed(seed)

        out = self._call_generation_pipeline(
            self.pipe,
            self._build_generation_kwargs(prompt_embeds, pooled_prompt_embeds, generator),
        )

        image = out.clamp(0, 1).squeeze(0).permute(1, 2, 0)      # HWC
        return image.to(self.device)

    def generate_image_from_embeddings_adam(self, text_embeddings, seed):
        generator = torch.Generator(device=self._pipeline_input_device()).manual_seed(seed)

        prompt_embeds = text_embeddings[0]
        pooled_prompt_embeds = text_embeddings[1]

        out = self._call_generation_pipeline(
            self.call_with_grad,
            self._build_generation_kwargs(prompt_embeds, pooled_prompt_embeds, generator),
        )

        image = out.clamp(0, 1).squeeze(0).permute(1, 2, 0)      # HWC
        return image.to(self.device)

    def _adam_autocast_context(self):
        autocast_dtype = self.model_dtype
        if autocast_dtype not in (torch.float16, torch.bfloat16):
            return nullcontext()

        pipeline_device = str(self._pipeline_input_device())
        if pipeline_device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if pipeline_device.startswith("cpu") and autocast_dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=autocast_dtype)
        return nullcontext()

    @staticmethod
    def _tensor_to_uint8_image(image_tensor):
        image_np = image_tensor.detach().to(torch.float32).cpu().numpy()
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
        image_np = np.clip(image_np, 0.0, 1.0)
        return (image_np * 255).astype(np.uint8)

    def _uint8_image_to_tensor(self, image_np):
        image_np = np.asarray(image_np, dtype=np.float32) / 255.0
        return torch.from_numpy(image_np).to(device=self.device, dtype=torch.float32)

    def _load_saved_image_tensor(self, image_path):
        with Image.open(image_path) as pil_image:
            return self._uint8_image_to_tensor(pil_image.convert("RGB"))

    def aesthetic_evaluation(self, image):
        if not self._should_evaluate_metric("aesthetic_score") or self.aesthetic_model is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # image is a tensor of shape [H, W, C]
        # Convert to [N, C, H, W] and ensure it's in float32
        image_input = image.permute(2, 0, 1).to(torch.float32)  # [1, C, H, W]

        if self.parameters["predictor"] == 0:
            # Simulacra Aesthetic Model
            score = self.aesthetic_model.predict_from_tensor(image_input)
        elif self.parameters["predictor"] == 1 or self.parameters["predictor"] == 2:
            # LAION Aesthetic Predictor V1 and V2
            score = self.aesthetic_model.predict_from_tensor(image_input)
        else:
            return torch.tensor(0.0, device=self.device)

        return score

    def evaluate_clip_score_cmaes(self, image_tensor, prompt):
        if self.clip_model is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Convert the image tensor to a PIL image
        image = (image_tensor * 255).clamp(0, 255).byte()
        image = Image.fromarray(image.cpu().numpy())

        # Preprocess the image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize the prompt
        text_input = clip.tokenize([prompt]).to(self.device)

        # Compute the CLIP embeddings
        image_features = self.clip_model.encode_image(image_input)
        text_features = self.clip_model.encode_text(text_input)

        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute the cosine similarity (CLIP score)
        clip_score = (image_features @ text_features.T)

        return clip_score

    def evaluate_clip_score_adam(self, image_tensor, text_features):
        if self.clip_model is None or text_features is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        self.clip_model.eval()

        # --- differentiable preprocess (no PIL, no .byte) ---
        img = image_tensor.permute(2,0,1).unsqueeze(0)            # [1,C,H,W]
        img = img.to(device=self.device, dtype=torch.float32)
        img = F.interpolate(img, size=(224,224), mode="bicubic", align_corners=False)
        mean = self._CLIP_MEAN.to(img.device, img.dtype)
        std  = self._CLIP_STD.to(img.device, img.dtype)
        img = (img - mean) / std

        # Encode image WITH grad (through CLIP image tower)
        image_features = self.clip_model.encode_image(img).float()
        image_features = F.normalize(image_features, dim=-1, eps=1e-6)

        sim = (image_features @ text_features.T).squeeze()  # scalar
        return sim

    def evaluate_image_reward_cmaes(self, image_tensor, prompt):
        if not self.use_image_reward or self.image_reward_model is None:
            return 0.0

        pil_image = Image.fromarray(self._tensor_to_uint8_image(image_tensor))
        return float(self.image_reward_model.score(prompt, pil_image))

    def evaluate_image_reward_adam(self, image_tensor, prompt_ids, prompt_attention_mask):
        if not self.use_image_reward or self.image_reward_model is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        image = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
        mean = self._CLIP_MEAN.to(image.device, image.dtype)
        std = self._CLIP_STD.to(image.device, image.dtype)
        image = (image - mean) / std

        reward = self.image_reward_model.score_gard(prompt_ids, prompt_attention_mask, image).squeeze()
        return reward.to(torch.float32)

    def evaluate_hpsv2_cmaes(self, image_tensor, prompt):
        if not self.use_hpsv2 or self.hpsv2_model is None or self.hpsv2_preprocess is None or self.hpsv2_tokenizer is None:
            return 0.0

        image = Image.fromarray(self._tensor_to_uint8_image(image_tensor))
        image = self.hpsv2_preprocess(image).unsqueeze(0).to(device=self.device, non_blocking=True)
        text = self.hpsv2_tokenizer([prompt]).to(device=self.device, non_blocking=True)

        autocast_context = (
            torch.cuda.amp.autocast()
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        with autocast_context:
            outputs = self.hpsv2_model(image, text)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits_per_image = image_features @ text_features.T

        return float(torch.diagonal(logits_per_image).detach().to(torch.float32).cpu().item())

    def evaluate_hpsv2_adam(self, image_tensor, text_tokens):
        if not self.use_hpsv2 or self.hpsv2_model is None or text_tokens is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        image = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
        mean = self._CLIP_MEAN.to(image.device, image.dtype)
        std = self._CLIP_STD.to(image.device, image.dtype)
        image = (image - mean) / std

        autocast_context = (
            torch.cuda.amp.autocast()
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        with autocast_context:
            outputs = self.hpsv2_model(image, text_tokens)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits_per_image = image_features @ text_features.T

        return torch.diagonal(logits_per_image).squeeze().to(torch.float32)

    def _pickscore_image_size(self):
        if self.pickscore_processor is None:
            return 224
        image_processor = self.pickscore_processor.image_processor
        crop_size = getattr(image_processor, "crop_size", None)
        if isinstance(crop_size, dict):
            return int(crop_size.get("height", crop_size.get("width", 224)))
        if isinstance(crop_size, int):
            return int(crop_size)
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            return int(size.get("shortest_edge", size.get("height", size.get("width", 224))))
        if isinstance(size, int):
            return int(size)
        return 224

    def _pickscore_mean_std(self, dtype):
        image_processor = self.pickscore_processor.image_processor
        mean = torch.tensor(image_processor.image_mean, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(image_processor.image_std, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        return mean, std

    def evaluate_pickscore_cmaes(self, image_tensor, prompt):
        if not self.use_pickscore or self.pickscore_model is None or self.pickscore_processor is None:
            return 0.0

        pil_image = Image.fromarray(self._tensor_to_uint8_image(image_tensor))
        image_inputs = self.pickscore_processor(
            images=pil_image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        text_inputs = self.pickscore_processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_embeds = self.pickscore_model.get_image_features(**image_inputs)
            image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
            text_embeds = self.pickscore_model.get_text_features(**text_inputs)
            text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
            scores = self.pickscore_model.logit_scale.exp() * (text_embeds @ image_embeds.T)
        return float(scores.squeeze().detach().to(torch.float32).cpu().item())

    def evaluate_pickscore_adam(self, image_tensor, text_inputs):
        if not self.use_pickscore or self.pickscore_model is None or self.pickscore_processor is None or text_inputs is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        image = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        image = F.interpolate(image, size=(self._pickscore_image_size(), self._pickscore_image_size()), mode="bicubic", align_corners=False)
        mean, std = self._pickscore_mean_std(image.dtype)
        image = (image - mean) / std

        image_embeds = self.pickscore_model.get_image_features(pixel_values=image)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        text_embeds = self.pickscore_model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        scores = self.pickscore_model.logit_scale.exp() * (text_embeds @ image_embeds.T)
        return scores.squeeze().to(torch.float32)

    def _combine_metric_components(self, metric_values):
        metric_weights = {
            "aesthetic_score": self.aesthetic_score_weight,
            "clip_score": self.clip_score_weight,
            "image_reward_score": self.image_reward_score_weight,
            "hpsv2_score": self.hpsv2_score_weight,
            "pickscore_score": self.pickscore_score_weight,
        }
        components = {}
        total = None
        for metric_name, raw_value in metric_values.items():
            weight = metric_weights.get(metric_name, 0.0)
            scale = self._metric_scales.get(metric_name, 1.0)
            component = weight * raw_value / scale
            components[metric_name] = component
            total = component if total is None else total + component
        return total, components

    def _evaluate_canonical_image_scores(self, image, selected_prompt):
        with torch.no_grad():
            aesthetic_score = self.aesthetic_evaluation(image).item()
            clip_score = self.evaluate_clip_score_cmaes(image, selected_prompt).item()
            image_reward_score = self.evaluate_image_reward_cmaes(image, selected_prompt)
            hpsv2_score = self.evaluate_hpsv2_cmaes(image, selected_prompt)
            pickscore_score = self.evaluate_pickscore_cmaes(image, selected_prompt)

        combined_score, components = self._combine_metric_components({
            "aesthetic_score": aesthetic_score,
            "clip_score": clip_score,
            "image_reward_score": image_reward_score,
            "hpsv2_score": hpsv2_score,
            "pickscore_score": pickscore_score,
        })

        return (
            float(combined_score),
            aesthetic_score,
            clip_score,
            image_reward_score,
            hpsv2_score,
            pickscore_score,
            components,
        )

    def _evaluate_canonical_image_path_scores(self, image_path, selected_prompt):
        image = self._load_saved_image_tensor(image_path)
        return self._evaluate_canonical_image_scores(image, selected_prompt)

    @staticmethod
    def _population_metric_columns():
        return {
            "fitness": ("avg_fitness", "std_fitness", "max_fitness"),
            "aesthetic_score": ("avg_aesthetic_score", "std_aesthetic_score", "max_aesthetic_score"),
            "clip_score": ("avg_clip_score", "std_clip_score", "max_clip_score"),
            "image_reward_score": ("avg_image_reward_score", "std_image_reward_score", "max_image_reward_score"),
            "hpsv2_score": ("avg_hpsv2_score", "std_hpsv2_score", "max_hpsv2_score"),
            "pickscore_score": ("avg_pickscore_score", "std_pickscore_score", "max_pickscore_score"),
        }

    def _postprocess_population_results_from_saved_images(self, results, results_folder, selected_prompt):
        results = results.copy()
        metric_columns = self._population_metric_columns()
        for row_idx in range(len(results)):
            if row_idx == 0:
                image_path = os.path.join(results_folder, "it_0.png")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(
                        f"Cannot rebuild fitness_results.csv because {image_path} is missing."
                    )
                (
                    canonical_score,
                    canonical_aesthetic_score,
                    canonical_clip_score,
                    canonical_image_reward_score,
                    canonical_hpsv2_score,
                    canonical_pickscore_score,
                    _,
                ) = self._evaluate_canonical_image_path_scores(image_path, selected_prompt)
                metric_values = {
                    "fitness": [canonical_score],
                    "aesthetic_score": [canonical_aesthetic_score],
                    "clip_score": [canonical_clip_score],
                    "image_reward_score": [canonical_image_reward_score],
                    "hpsv2_score": [canonical_hpsv2_score],
                    "pickscore_score": [canonical_pickscore_score],
                }
            else:
                metric_values = None
                gen_folder = os.path.join(results_folder, f"gen_{row_idx}")
                if self.parameters.get("save_gens", False) and os.path.isdir(gen_folder):
                    gen_image_paths = sorted(
                        os.path.join(gen_folder, name)
                        for name in os.listdir(gen_folder)
                        if name.lower().endswith(".png")
                    )
                    if gen_image_paths:
                        metric_values = {key: [] for key in metric_columns}
                        for image_path in gen_image_paths:
                            (
                                canonical_score,
                                canonical_aesthetic_score,
                                canonical_clip_score,
                                canonical_image_reward_score,
                                canonical_hpsv2_score,
                                canonical_pickscore_score,
                                _,
                            ) = self._evaluate_canonical_image_path_scores(image_path, selected_prompt)
                            metric_values["fitness"].append(canonical_score)
                            metric_values["aesthetic_score"].append(canonical_aesthetic_score)
                            metric_values["clip_score"].append(canonical_clip_score)
                            metric_values["image_reward_score"].append(canonical_image_reward_score)
                            metric_values["hpsv2_score"].append(canonical_hpsv2_score)
                            metric_values["pickscore_score"].append(canonical_pickscore_score)

                if metric_values is None:
                    image_path = os.path.join(results_folder, f"best_{row_idx}.png")
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(
                            f"Cannot rebuild fitness_results.csv because {image_path} is missing."
                        )
                    (
                        canonical_score,
                        canonical_aesthetic_score,
                        canonical_clip_score,
                        canonical_image_reward_score,
                        canonical_hpsv2_score,
                        canonical_pickscore_score,
                        _,
                    ) = self._evaluate_canonical_image_path_scores(image_path, selected_prompt)
                    results.at[row_idx, "max_fitness"] = canonical_score
                    results.at[row_idx, "max_aesthetic_score"] = canonical_aesthetic_score
                    results.at[row_idx, "max_clip_score"] = canonical_clip_score
                    results.at[row_idx, "max_image_reward_score"] = canonical_image_reward_score
                    results.at[row_idx, "max_hpsv2_score"] = canonical_hpsv2_score
                    results.at[row_idx, "max_pickscore_score"] = canonical_pickscore_score
                    continue

            for metric_name, values in metric_values.items():
                avg_col, std_col, max_col = metric_columns[metric_name]
                values_arr = np.asarray(values, dtype=float)
                results.at[row_idx, avg_col] = float(np.mean(values_arr))
                results.at[row_idx, std_col] = float(np.std(values_arr))
                results.at[row_idx, max_col] = float(np.max(values_arr))

        return results

    def format_time(self, seconds):
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _save_run_config(self, results_folder, seed, selected_prompt, category=None, prompt_number=None):
        run_config = dict(self.parameters)
        run_config["seed"] = int(seed)
        run_config["selected_prompt"] = selected_prompt
        run_config["results_folder"] = self.parameters["results_folder"]
        run_config["experiment_output_folder"] = self.OUTPUT_FOLDER
        run_config["run_results_folder"] = results_folder
        run_config["resolved_model_backend"] = self.model_backend
        run_config["resolved_torch_dtype"] = str(self.model_dtype).replace("torch.", "")
        run_config["resolved_lcm_origin_steps"] = self.lcm_origin_steps

        if category is not None:
            run_config["category"] = category
        if prompt_number is not None:
            run_config["prompt_number"] = prompt_number

        config_save_path = os.path.join(results_folder, "config.yaml")
        with open(config_save_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(run_config, file, sort_keys=False, allow_unicode=True)

    def evaluate(self, input_embedding, seed, embedding_shape, selected_prompt, save_path=None):
        # x is a NumPy array representing the embedding vector
        # Convert it to a torch tensor

        # Reshape the embedding to the original shape
        split = np.prod(embedding_shape[0])
        pe  = torch.tensor(input_embedding[:split],  dtype=torch.float32, device=self.device).view(embedding_shape[0])
        ppe = torch.tensor(input_embedding[split:], dtype=torch.float32, device=self.device).view(embedding_shape[1])

        with torch.no_grad():
            image = self.generate_image_from_embeddings_cmaes(pe, ppe, seed)

            fitness, aesthetic_score, clip_score, image_reward_score, hpsv2_score, pickscore_score, components = (
                self._evaluate_canonical_image_scores(image, selected_prompt)
            )

        if save_path is not None:
            # Save the generated image
            image_np = image.detach().clone().to(torch.float32).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(save_path)

        # CMA-ES minimizes the function, so we need to invert the score if higher is better
        return -fitness, aesthetic_score, clip_score, image_reward_score, hpsv2_score, pickscore_score, components

    def _save_population_plot_results(self, results, results_folder):
        def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
            lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
            upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]

            plt.plot(x_axis, m_vec, '--', label=description + " Avg.")
            plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3, label=description + " Avg. ± SD")
            if title is not None:
                plt.title(title)
            if y_label is not None:
                plt.ylabel(y_label)
            if x_label is not None:
                plt.xlabel(x_label)

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_fitness'], results['std_fitness'], "Fitness")
        plt.plot(results['generation'], results['max_fitness'], 'r-', label="Best Fitness")
        plt.ylim(0, 1.1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/fitness_evolution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_aesthetic_score'], results['std_aesthetic_score'], "Population")
        plt.plot(results['generation'], results['max_aesthetic_score'], 'r-', label="Best")
        plt.ylim(0, 10)
        plt.xlabel('Generation')
        plt.ylabel('Aesthetic Score')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/aesthetic_score_evolution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_clip_score'], results['std_clip_score'], "Population")
        plt.plot(results['generation'], results['max_clip_score'], 'r-', label="Best")
        plt.ylim(0, 0.6)
        plt.xlabel('Generation')
        plt.ylabel('CLIP Score')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/clip_score_evolution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_image_reward_score'], results['std_image_reward_score'], "Population")
        plt.plot(results['generation'], results['max_image_reward_score'], 'r-', label="Best")
        plt.xlabel('Generation')
        plt.ylabel('ImageReward Score')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/image_reward_evolution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_hpsv2_score'], results['std_hpsv2_score'], "Population")
        plt.plot(results['generation'], results['max_hpsv2_score'], 'r-', label="Best")
        plt.xlabel('Generation')
        plt.ylabel('HPSv2 Score')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/hpsv2_evolution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_pickscore_score'], results['std_pickscore_score'], "Population")
        plt.plot(results['generation'], results['max_pickscore_score'], 'r-', label="Best")
        plt.xlabel('Generation')
        plt.ylabel('PickScore')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/pickscore_evolution.png")
        plt.close()

    def run_cmaes_optimization(self, seed = None, seed_number = None, prompt = None, category = None, prompt_number = None):

        if seed is None:
            seed = self.parameters["seed"]
        if prompt is None:
            selected_prompt = self.parameters["selected_prompt"]
        else:
            selected_prompt = prompt

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if category is not None:
            print(f"Selected prompt: {selected_prompt} (Category: {category})")
        else:
            print(f"Selected prompt: {selected_prompt}")

        results_folder = f"{self.OUTPUT_FOLDER}/results_{self.model_name}_{seed}"

        if prompt_number is not None:
            results_folder += f"_{prompt_number}"

        os.makedirs(results_folder, exist_ok=True)
        self._save_run_config(results_folder, seed, selected_prompt, category=category, prompt_number=prompt_number)

        save_path = None

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_embeddings(selected_prompt)

        # Set CMA-ES options
        es_options = {
            'seed': seed,
            'popsize': self.parameters["pop_size"],
            'maxiter': self.parameters["num_generations"],
            'verb_filenameprefix': results_folder + '/outcmaes',  # Save logs
            'verb_log': 0,  # Disable log output
            'verbose': -9,  # Suppress console output
        }

        if self.parameters["cmaes_variant"] == "cmaes":
            print("Using standard CMA-ES")
        elif self.parameters["cmaes_variant"] == "sep":
            print("Using sep-CMA-ES")
            es_options['CMA_diagonal'] = True
        elif self.parameters["cmaes_variant"] == "vd":
            print("Using VD-CMA-ES")
            es_options = GaussVDSampler.extend_cma_options(es_options)
        else:
            raise ValueError(f"Unknown CMA-ES variant: {self.parameters['cmaes_variant']}")

        trainable_params_init = torch.cat([
            prompt_embeds.flatten(),
            pooled_prompt_embeds.flatten()
            ]).to(torch.float32).cpu().numpy()

        sh_prompt  = prompt_embeds.shape         
        sh_pooled  = pooled_prompt_embeds.shape  

        text_embeddings_init_shape = [sh_prompt, sh_pooled]

        es = cma.CMAEvolutionStrategy(trainable_params_init, self.parameters["sigma"], es_options)

        with torch.no_grad():
            initial_image = self.generate_image_from_embeddings_cmaes(prompt_embeds.clone(), pooled_prompt_embeds.clone(), seed)
            image_np = initial_image.detach().clone().to(torch.float32).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(f"{results_folder}/it_0.png")

            initial_fitness, initial_aesthetic_score, initial_clip_score, initial_image_reward_score, initial_hpsv2_score, initial_pickscore_score, initial_components = self.evaluate(trainable_params_init, seed, text_embeddings_init_shape, selected_prompt)

        time_list = [0]
        best_aesthetic_score_overall = initial_aesthetic_score
        best_clip_score_overall = initial_clip_score
        best_fitness_overall = initial_fitness
        best_text_embeddings_overall = trainable_params_init

        start_time = time.time()
        generation = 0

        max_fit_list = [-initial_fitness]
        avg_fit_list = [-initial_fitness]
        std_fit_list = [0]

        max_aesthetic_score_list = [initial_aesthetic_score]
        avg_aesthetic_score_list = [initial_aesthetic_score]
        std_aesthetic_score_list = [0]

        max_clip_score_list = [initial_clip_score]
        avg_clip_score_list = [initial_clip_score]
        std_clip_score_list = [0]

        max_image_reward_score_list = [initial_image_reward_score]
        avg_image_reward_score_list = [initial_image_reward_score]
        std_image_reward_score_list = [0]
        max_hpsv2_score_list = [initial_hpsv2_score]
        avg_hpsv2_score_list = [initial_hpsv2_score]
        std_hpsv2_score_list = [0]
        max_pickscore_score_list = [initial_pickscore_score]
        avg_pickscore_score_list = [initial_pickscore_score]
        std_pickscore_score_list = [0]

        while not es.stop():
            elapsed_time = time.time() - start_time
            if self.parameters['time_limit_seconds'] is not None and elapsed_time >= self.parameters['time_limit_seconds']:
                print(
                    "Time limit reached before starting generation "
                    f"{generation + 1}/{self.parameters['num_generations']} (elapsed: {self.format_time(elapsed_time)})."
                )
                break

            print(f"Generation {generation+1}/{self.parameters['num_generations']}")

            if self.parameters["save_gens"]:
                os.makedirs(results_folder+"/gen_%d" % (generation+1), exist_ok=True)

            # Ask for new candidate solutions
            solutions = es.ask()
            # Evaluate candidate solutions
            tmp_fitnesses = []
            aesthetic_scores = []
            clip_scores = []
            image_reward_scores = []
            hpsv2_scores = []
            pickscore_scores = []

            ind_id = 1
            for x in solutions:
                if self.parameters["save_gens"]:
                    save_path = results_folder + "/gen_%d/id_%d.png" % (generation+1, ind_id)
                fitness, aesthetic_score, clip_score, image_reward_score, hpsv2_score, pickscore_score, _ = self.evaluate(x, seed, text_embeddings_init_shape, selected_prompt, save_path)
                tmp_fitnesses.append(fitness)
                aesthetic_scores.append(aesthetic_score)
                clip_scores.append(clip_score)
                image_reward_scores.append(image_reward_score)
                hpsv2_scores.append(hpsv2_score)
                pickscore_scores.append(pickscore_score)
                ind_id += 1
            # Tell CMA-ES the fitnesses
            es.tell(solutions, tmp_fitnesses)

            # Record statistics
            fitnesses = [-f for f in tmp_fitnesses]  # Convert back to positive scores

            max_fit = max(fitnesses)
            avg_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)

            max_fit_list.append(max_fit)
            avg_fit_list.append(avg_fit)
            std_fit_list.append(std_fit)

            max_aesthetic_score = max(aesthetic_scores)
            avg_aesthetic_score = np.mean(aesthetic_scores)
            std_aesthetic_score = np.std(aesthetic_scores)

            max_aesthetic_score_list.append(max_aesthetic_score)
            avg_aesthetic_score_list.append(avg_aesthetic_score)
            std_aesthetic_score_list.append(std_aesthetic_score)

            max_clip_score = max(clip_scores)
            avg_clip_score = np.mean(clip_scores)
            std_clip_score = np.std(clip_scores)

            max_clip_score_list.append(max_clip_score)
            avg_clip_score_list.append(avg_clip_score)
            std_clip_score_list.append(std_clip_score)

            max_image_reward_score = max(image_reward_scores)
            avg_image_reward_score = np.mean(image_reward_scores)
            std_image_reward_score = np.std(image_reward_scores)

            max_image_reward_score_list.append(max_image_reward_score)
            avg_image_reward_score_list.append(avg_image_reward_score)
            std_image_reward_score_list.append(std_image_reward_score)
            max_hpsv2_score = max(hpsv2_scores)
            avg_hpsv2_score = np.mean(hpsv2_scores)
            std_hpsv2_score = np.std(hpsv2_scores)

            max_hpsv2_score_list.append(max_hpsv2_score)
            avg_hpsv2_score_list.append(avg_hpsv2_score)
            std_hpsv2_score_list.append(std_hpsv2_score)
            max_pickscore_score = max(pickscore_scores)
            avg_pickscore_score = np.mean(pickscore_scores)
            std_pickscore_score = np.std(pickscore_scores)
            max_pickscore_score_list.append(max_pickscore_score)
            avg_pickscore_score_list.append(avg_pickscore_score)
            std_pickscore_score_list.append(std_pickscore_score)

            current_best_idx = int(np.argmin(tmp_fitnesses))
            current_best_x = solutions[current_best_idx]
            best_x = es.result.xbest
            best_fitness = -es.result.fbest  # Convert back to positive score

            with torch.no_grad():
                # Save the best image from the current generation.
                split = np.prod(text_embeddings_init_shape[0])
                best_pe  = torch.tensor(current_best_x[:split],  dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[0])
                best_ppe = torch.tensor(current_best_x[split:], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[1])
                best_image = self.generate_image_from_embeddings_cmaes(best_pe, best_ppe, seed)
                image_np = best_image.detach().clone().to(torch.float32).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                pil_image.save(results_folder + "/best_%d.png" % (generation+1))

            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_text_embeddings_overall = best_x

            generation += 1

            elapsed_time = time.time() - start_time
            generations_done = generation
            generations_left = self.parameters["num_generations"] - generations_done
            average_time_per_generation = elapsed_time / generations_done
            estimated_time_remaining = average_time_per_generation * generations_left

            formatted_time_remaining = self.format_time(estimated_time_remaining)

            time_list.append(elapsed_time)

            # Save the metrics
            results = pd.DataFrame({
                "generation": list(range(0, generation + 1)),
                "prompt": [selected_prompt] + [''] * generation,
                "avg_fitness": avg_fit_list,
                "std_fitness": std_fit_list,
                "max_fitness": max_fit_list,
                "avg_aesthetic_score": avg_aesthetic_score_list,
                "std_aesthetic_score": std_aesthetic_score_list,
                "max_aesthetic_score": max_aesthetic_score_list,
                "avg_clip_score": avg_clip_score_list,
                "std_clip_score": std_clip_score_list,
                "max_clip_score": max_clip_score_list,
                "avg_image_reward_score": avg_image_reward_score_list,
                "std_image_reward_score": std_image_reward_score_list,
                "max_image_reward_score": max_image_reward_score_list,
                "avg_hpsv2_score": avg_hpsv2_score_list,
                "std_hpsv2_score": std_hpsv2_score_list,
                "max_hpsv2_score": max_hpsv2_score_list,
                "avg_pickscore_score": avg_pickscore_score_list,
                "std_pickscore_score": std_pickscore_score_list,
                "max_pickscore_score": max_pickscore_score_list,
                "elapsed_time": time_list
            })

            if category is not None:
                results["category"] = [category] + [''] * generation

            results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')

            self._save_population_plot_results(results, results_folder)

            # Print stats
            print(f"Generation {generation}/{self.parameters['num_generations']}: Max fitness: {max_fit}, Avg fitness: {avg_fit}, Max aesthetic score: {max_aesthetic_score}, Avg aesthetic score: {avg_aesthetic_score}, Max clip score: {max_clip_score}, Avg clip score: {avg_clip_score}, Max ImageReward score: {max_image_reward_score}, Avg ImageReward score: {avg_image_reward_score}, Max HPSv2 score: {max_hpsv2_score}, Avg HPSv2 score: {avg_hpsv2_score}, Max PickScore: {max_pickscore_score}, Avg PickScore: {avg_pickscore_score}, Estimated time remaining: {formatted_time_remaining}")

        # Save the overall best image
        with torch.no_grad():
            split = np.prod(text_embeddings_init_shape[0])
            best_overall_pe  = torch.tensor(best_text_embeddings_overall[:split],  dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[0])
            best_overall_ppe = torch.tensor(best_text_embeddings_overall[split:], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[1])
            best_image = self.generate_image_from_embeddings_cmaes(best_overall_pe, best_overall_ppe, seed)
        best_image_np = best_image.detach().to(torch.float32).cpu().numpy()
        best_image_np = (best_image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(best_image_np)
        pil_image.save(f"{results_folder}/best_all.png")

        results = pd.DataFrame({
            "generation": list(range(0, generation + 1)),
            "prompt": [selected_prompt] + [''] * generation,
            "avg_fitness": avg_fit_list,
            "std_fitness": std_fit_list,
            "max_fitness": max_fit_list,
            "avg_aesthetic_score": avg_aesthetic_score_list,
            "std_aesthetic_score": std_aesthetic_score_list,
            "max_aesthetic_score": max_aesthetic_score_list,
            "avg_clip_score": avg_clip_score_list,
            "std_clip_score": std_clip_score_list,
            "max_clip_score": max_clip_score_list,
            "avg_image_reward_score": avg_image_reward_score_list,
            "std_image_reward_score": std_image_reward_score_list,
            "max_image_reward_score": max_image_reward_score_list,
            "avg_hpsv2_score": avg_hpsv2_score_list,
            "std_hpsv2_score": std_hpsv2_score_list,
            "max_hpsv2_score": max_hpsv2_score_list,
            "avg_pickscore_score": avg_pickscore_score_list,
            "std_pickscore_score": std_pickscore_score_list,
            "max_pickscore_score": max_pickscore_score_list,
            "elapsed_time": time_list
        })
        if category is not None:
            results["category"] = [category] + [''] * generation
        results = self._postprocess_population_results_from_saved_images(results, results_folder, selected_prompt)
        results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')
        self._save_population_plot_results(results, results_folder)

        return results_folder

    def run_ga_optimization(self, seed = None, seed_number = None, prompt = None, category = None, prompt_number = None):
        if seed is None:
            seed = self.parameters["seed"]
        if prompt is None:
            selected_prompt = self.parameters["selected_prompt"]
        else:
            selected_prompt = prompt

        seed = int(seed)
        num_generations = int(self.parameters["num_generations"])
        pop_size = int(self.parameters["pop_size"])
        mutation_std = float(self.parameters.get("ga_mutation_std", self.parameters.get("sigma", 0.1)))
        elite_count = int(self.parameters.get("ga_elite_count", 1))
        crossover_rate = float(self.parameters.get("ga_crossover_rate", 0.5))
        mutation_rate = float(self.parameters.get("ga_mutation_rate", 0.1))

        if pop_size < 2:
            raise ValueError("GA requires pop_size >= 2.")
        if mutation_std <= 0:
            raise ValueError("GA requires ga_mutation_std > 0.")
        if not 1 <= elite_count <= pop_size:
            raise ValueError("GA requires 1 <= ga_elite_count <= pop_size.")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("GA requires 0 <= ga_crossover_rate <= 1.")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("GA requires 0 <= ga_mutation_rate <= 1.")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)

        if category is not None:
            print(f"Selected prompt: {selected_prompt} (Category: {category})")
        else:
            print(f"Selected prompt: {selected_prompt}")

        results_folder = f"{self.OUTPUT_FOLDER}/results_{self.model_name}_{seed}"
        if prompt_number is not None:
            results_folder += f"_{prompt_number}"
        os.makedirs(results_folder, exist_ok=True)
        self._save_run_config(results_folder, seed, selected_prompt, category=category, prompt_number=prompt_number)

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_embeddings(selected_prompt)
            initial_image = self.generate_image_from_embeddings_cmaes(prompt_embeds.clone(), pooled_prompt_embeds.clone(), seed)
            image_np = initial_image.detach().clone().to(torch.float32).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(f"{results_folder}/it_0.png")

        trainable_params_init = torch.cat([
            prompt_embeds.flatten(),
            pooled_prompt_embeds.flatten()
        ]).to(torch.float32).cpu().numpy()

        sh_prompt = prompt_embeds.shape
        sh_pooled = pooled_prompt_embeds.shape
        text_embeddings_init_shape = [sh_prompt, sh_pooled]

        initial_fitness, initial_aesthetic_score, initial_clip_score, initial_image_reward_score, initial_hpsv2_score, initial_pickscore_score, initial_components = self.evaluate(
            trainable_params_init, seed, text_embeddings_init_shape, selected_prompt
        )

        population = np.repeat(trainable_params_init[None, :], pop_size, axis=0)
        population += rng.normal(0.0, mutation_std, size=population.shape)
        population[0] = trainable_params_init.copy()

        best_fitness_overall = -initial_fitness
        best_text_embeddings_overall = trainable_params_init.copy()
        time_list = [0]
        start_time = time.time()

        max_fit_list = [best_fitness_overall]
        avg_fit_list = [best_fitness_overall]
        std_fit_list = [0]
        max_aesthetic_score_list = [initial_aesthetic_score]
        avg_aesthetic_score_list = [initial_aesthetic_score]
        std_aesthetic_score_list = [0]
        max_clip_score_list = [initial_clip_score]
        avg_clip_score_list = [initial_clip_score]
        std_clip_score_list = [0]
        max_image_reward_score_list = [initial_image_reward_score]
        avg_image_reward_score_list = [initial_image_reward_score]
        std_image_reward_score_list = [0]
        max_hpsv2_score_list = [initial_hpsv2_score]
        avg_hpsv2_score_list = [initial_hpsv2_score]
        std_hpsv2_score_list = [0]
        max_pickscore_score_list = [initial_pickscore_score]
        avg_pickscore_score_list = [initial_pickscore_score]
        std_pickscore_score_list = [0]

        for generation in range(1, num_generations + 1):
            elapsed_time = time.time() - start_time
            if self.parameters['time_limit_seconds'] is not None and elapsed_time >= self.parameters['time_limit_seconds']:
                print(
                    "Time limit reached before starting generation "
                    f"{generation}/{num_generations} (elapsed: {self.format_time(elapsed_time)})."
                )
                break

            print(f"Generation {generation}/{num_generations}")

            if self.parameters["save_gens"]:
                os.makedirs(results_folder + "/gen_%d" % generation, exist_ok=True)

            tmp_fitnesses = []
            aesthetic_scores = []
            clip_scores = []
            image_reward_scores = []
            hpsv2_scores = []
            pickscore_scores = []

            for ind_id, x in enumerate(population, start=1):
                save_path = None
                if self.parameters["save_gens"]:
                    save_path = results_folder + "/gen_%d/id_%d.png" % (generation, ind_id)
                fitness, aesthetic_score, clip_score, image_reward_score, hpsv2_score, pickscore_score, _ = self.evaluate(
                    x, seed, text_embeddings_init_shape, selected_prompt, save_path
                )
                tmp_fitnesses.append(fitness)
                aesthetic_scores.append(aesthetic_score)
                clip_scores.append(clip_score)
                image_reward_scores.append(image_reward_score)
                hpsv2_scores.append(hpsv2_score)
                pickscore_scores.append(pickscore_score)

            fitnesses = np.array([-f for f in tmp_fitnesses], dtype=float)
            order = np.argsort(fitnesses)[::-1]
            population = population[order]
            fitnesses = fitnesses[order]
            aesthetic_scores = np.array(aesthetic_scores, dtype=float)[order]
            clip_scores = np.array(clip_scores, dtype=float)[order]
            image_reward_scores = np.array(image_reward_scores, dtype=float)[order]
            hpsv2_scores = np.array(hpsv2_scores, dtype=float)[order]
            pickscore_scores = np.array(pickscore_scores, dtype=float)[order]

            max_fit = float(np.max(fitnesses))
            avg_fit = float(np.mean(fitnesses))
            std_fit = float(np.std(fitnesses))
            max_aesthetic_score = float(np.max(aesthetic_scores))
            avg_aesthetic_score = float(np.mean(aesthetic_scores))
            std_aesthetic_score = float(np.std(aesthetic_scores))
            max_clip_score = float(np.max(clip_scores))
            avg_clip_score = float(np.mean(clip_scores))
            std_clip_score = float(np.std(clip_scores))
            max_image_reward_score = float(np.max(image_reward_scores))
            avg_image_reward_score = float(np.mean(image_reward_scores))
            std_image_reward_score = float(np.std(image_reward_scores))
            max_hpsv2_score = float(np.max(hpsv2_scores))
            avg_hpsv2_score = float(np.mean(hpsv2_scores))
            std_hpsv2_score = float(np.std(hpsv2_scores))
            max_pickscore_score = float(np.max(pickscore_scores))
            avg_pickscore_score = float(np.mean(pickscore_scores))
            std_pickscore_score = float(np.std(pickscore_scores))

            max_fit_list.append(max_fit)
            avg_fit_list.append(avg_fit)
            std_fit_list.append(std_fit)
            max_aesthetic_score_list.append(max_aesthetic_score)
            avg_aesthetic_score_list.append(avg_aesthetic_score)
            std_aesthetic_score_list.append(std_aesthetic_score)
            max_clip_score_list.append(max_clip_score)
            avg_clip_score_list.append(avg_clip_score)
            std_clip_score_list.append(std_clip_score)
            max_image_reward_score_list.append(max_image_reward_score)
            avg_image_reward_score_list.append(avg_image_reward_score)
            std_image_reward_score_list.append(std_image_reward_score)
            max_hpsv2_score_list.append(max_hpsv2_score)
            avg_hpsv2_score_list.append(avg_hpsv2_score)
            std_hpsv2_score_list.append(std_hpsv2_score)
            max_pickscore_score_list.append(max_pickscore_score)
            avg_pickscore_score_list.append(avg_pickscore_score)
            std_pickscore_score_list.append(std_pickscore_score)

            best_x = population[0].copy()
            if max_fit > best_fitness_overall:
                best_fitness_overall = max_fit
                best_text_embeddings_overall = best_x.copy()

            with torch.no_grad():
                split = np.prod(text_embeddings_init_shape[0])
                best_pe = torch.tensor(best_x[:split], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[0])
                best_ppe = torch.tensor(best_x[split:], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[1])
                best_image = self.generate_image_from_embeddings_cmaes(best_pe, best_ppe, seed)
                image_np = best_image.detach().clone().to(torch.float32).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                pil_image.save(results_folder + "/best_%d.png" % generation)

            elapsed_time = time.time() - start_time
            generations_done = generation
            generations_left = num_generations - generations_done
            average_time_per_generation = elapsed_time / generations_done
            estimated_time_remaining = average_time_per_generation * generations_left
            formatted_time_remaining = self.format_time(estimated_time_remaining)
            time_list.append(elapsed_time)

            results = pd.DataFrame({
                "generation": list(range(0, generation + 1)),
                "prompt": [selected_prompt] + [''] * generation,
                "avg_fitness": avg_fit_list,
                "std_fitness": std_fit_list,
                "max_fitness": max_fit_list,
                "avg_aesthetic_score": avg_aesthetic_score_list,
                "std_aesthetic_score": std_aesthetic_score_list,
                "max_aesthetic_score": max_aesthetic_score_list,
                "avg_clip_score": avg_clip_score_list,
                "std_clip_score": std_clip_score_list,
                "max_clip_score": max_clip_score_list,
                "avg_image_reward_score": avg_image_reward_score_list,
                "std_image_reward_score": std_image_reward_score_list,
                "max_image_reward_score": max_image_reward_score_list,
                "avg_hpsv2_score": avg_hpsv2_score_list,
                "std_hpsv2_score": std_hpsv2_score_list,
                "max_hpsv2_score": max_hpsv2_score_list,
                "avg_pickscore_score": avg_pickscore_score_list,
                "std_pickscore_score": std_pickscore_score_list,
                "max_pickscore_score": max_pickscore_score_list,
                "elapsed_time": time_list
            })

            if category is not None:
                results["category"] = [category] + [''] * generation

            results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')
            self._save_population_plot_results(results, results_folder)

            print(f"Generation {generation}/{num_generations}: Max fitness: {max_fit}, Avg fitness: {avg_fit}, Max aesthetic score: {max_aesthetic_score}, Avg aesthetic score: {avg_aesthetic_score}, Max clip score: {max_clip_score}, Avg clip score: {avg_clip_score}, Max ImageReward score: {max_image_reward_score}, Avg ImageReward score: {avg_image_reward_score}, Max HPSv2 score: {max_hpsv2_score}, Avg HPSv2 score: {avg_hpsv2_score}, Max PickScore: {max_pickscore_score}, Avg PickScore: {avg_pickscore_score}, Estimated time remaining: {formatted_time_remaining}")

            elites = population[:elite_count].copy()
            next_population = [elites[i].copy() for i in range(elite_count)]

            while len(next_population) < pop_size:
                parent_indices = rng.integers(0, elite_count, size=2)
                parent_a = elites[parent_indices[0]]
                parent_b = elites[parent_indices[1]]
                crossover_mask = rng.random(parent_a.shape[0]) < crossover_rate
                child = np.where(crossover_mask, parent_a, parent_b)
                mutation_mask = rng.random(child.shape[0]) < mutation_rate
                if np.any(mutation_mask):
                    child = child.copy()
                    child[mutation_mask] += rng.normal(0.0, mutation_std, size=int(np.sum(mutation_mask)))
                next_population.append(child)

            population = np.array(next_population[:pop_size], dtype=np.float32)
            population[0] = best_text_embeddings_overall.copy()

        with torch.no_grad():
            split = np.prod(text_embeddings_init_shape[0])
            best_overall_pe = torch.tensor(best_text_embeddings_overall[:split], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[0])
            best_overall_ppe = torch.tensor(best_text_embeddings_overall[split:], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[1])
            best_image = self.generate_image_from_embeddings_cmaes(best_overall_pe, best_overall_ppe, seed)
        best_image_np = best_image.detach().to(torch.float32).cpu().numpy()
        best_image_np = (best_image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(best_image_np)
        pil_image.save(f"{results_folder}/best_all.png")

        results = pd.DataFrame({
            "generation": list(range(0, generation + 1)),
            "prompt": [selected_prompt] + [''] * generation,
            "avg_fitness": avg_fit_list,
            "std_fitness": std_fit_list,
            "max_fitness": max_fit_list,
            "avg_aesthetic_score": avg_aesthetic_score_list,
            "std_aesthetic_score": std_aesthetic_score_list,
            "max_aesthetic_score": max_aesthetic_score_list,
            "avg_clip_score": avg_clip_score_list,
            "std_clip_score": std_clip_score_list,
            "max_clip_score": max_clip_score_list,
            "avg_image_reward_score": avg_image_reward_score_list,
            "std_image_reward_score": std_image_reward_score_list,
            "max_image_reward_score": max_image_reward_score_list,
            "avg_hpsv2_score": avg_hpsv2_score_list,
            "std_hpsv2_score": std_hpsv2_score_list,
            "max_hpsv2_score": max_hpsv2_score_list,
            "avg_pickscore_score": avg_pickscore_score_list,
            "std_pickscore_score": std_pickscore_score_list,
            "max_pickscore_score": max_pickscore_score_list,
            "elapsed_time": time_list
        })
        if category is not None:
            results["category"] = [category] + [''] * generation
        results = self._postprocess_population_results_from_saved_images(results, results_folder, selected_prompt)
        results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')
        self._save_population_plot_results(results, results_folder)

        return results_folder

    def _random_sampler_num_images(self):
        for key in ("num_images_to_generate", "num_images", "random_sampler_num_images"):
            if key in self.parameters and self.parameters[key] is not None:
                num_images = int(self.parameters[key])
                if num_images <= 0:
                    raise ValueError(f"Random sampler requires {key} > 0.")
                return num_images
        raise ValueError(
            "Random sampler requires one of: num_images_to_generate, num_images, "
            "or random_sampler_num_images."
        )

    @staticmethod
    def _generate_sample_seeds(seed, num_images, excluded_seeds=None):
        rng = np.random.default_rng(int(seed))
        sample_seeds = []
        seen = set(int(s) for s in (excluded_seeds or []))
        while len(sample_seeds) < num_images:
            candidate = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
            if candidate in seen:
                continue
            seen.add(candidate)
            sample_seeds.append(candidate)
        return sample_seeds

    def run_random_sampler_optimization(self, seed=None, seed_number=None, prompt=None, category=None, prompt_number=None):
        if seed is None:
            seed = self.parameters["seed"]
        if prompt is None:
            selected_prompt = self.parameters["selected_prompt"]
        else:
            selected_prompt = prompt

        seed = int(seed)
        num_images = self._random_sampler_num_images()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if category is not None:
            print(f"Selected prompt: {selected_prompt} (Category: {category})")
        else:
            print(f"Selected prompt: {selected_prompt}")

        results_folder = f"{self.OUTPUT_FOLDER}/results_{self.model_name}_{seed}"
        if prompt_number is not None:
            results_folder += f"_{prompt_number}"
        os.makedirs(results_folder, exist_ok=True)
        self._save_run_config(results_folder, seed, selected_prompt, category=category, prompt_number=prompt_number)

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_embeddings(selected_prompt)

        trainable_params_init = torch.cat([
            prompt_embeds.flatten(),
            pooled_prompt_embeds.flatten()
        ]).to(torch.float32).cpu().numpy()

        text_embeddings_init_shape = [prompt_embeds.shape, pooled_prompt_embeds.shape]
        sample_seeds = self._generate_sample_seeds(seed, num_images, excluded_seeds={seed})

        sample_rows = []
        sample_paths = []
        sample_times = []
        time_list = [0.0]
        start_time = time.time()

        fitness_history = []
        aesthetic_history = []
        clip_history = []
        image_reward_history = []
        hpsv2_history = []
        pickscore_history = []

        baseline_path = os.path.join(results_folder, "it_0.png")
        initial_fitness, initial_aesthetic_score, initial_clip_score, initial_image_reward_score, initial_hpsv2_score, initial_pickscore_score, _ = self.evaluate(
            trainable_params_init,
            seed,
            text_embeddings_init_shape,
            selected_prompt,
            baseline_path,
        )
        initial_positive_fitness = float(-initial_fitness)
        initial_aesthetic_score = float(initial_aesthetic_score)
        initial_clip_score = float(initial_clip_score)
        initial_image_reward_score = float(initial_image_reward_score)
        initial_hpsv2_score = float(initial_hpsv2_score)
        initial_pickscore_score = float(initial_pickscore_score)

        best_fitness_overall = initial_positive_fitness
        best_sample_path = baseline_path

        fitness_history.append(initial_positive_fitness)
        aesthetic_history.append(initial_aesthetic_score)
        clip_history.append(initial_clip_score)
        image_reward_history.append(initial_image_reward_score)
        hpsv2_history.append(initial_hpsv2_score)
        pickscore_history.append(initial_pickscore_score)

        sample_rows.append({
            "sample": 0,
            "generation": 0,
            "seed": seed,
            "prompt": selected_prompt,
            "fitness": initial_positive_fitness,
            "aesthetic_score": initial_aesthetic_score,
            "clip_score": initial_clip_score,
            "image_reward_score": initial_image_reward_score,
            "hpsv2_score": initial_hpsv2_score,
            "pickscore_score": initial_pickscore_score,
            "elapsed_time": 0.0,
        })
        if category is not None:
            sample_rows[-1]["category"] = category

        def build_random_sampler_results(histories, elapsed_times, sampled_seeds):
            row_count = len(histories["fitness"])
            return pd.DataFrame({
                "generation": list(range(row_count)),
                "prompt": [selected_prompt] + [''] * (row_count - 1),
                "sampled_seed": sampled_seeds[:row_count],
                "avg_fitness": [float(np.mean(histories["fitness"][:i])) for i in range(1, row_count + 1)],
                "std_fitness": [float(np.std(histories["fitness"][:i])) for i in range(1, row_count + 1)],
                "max_fitness": [float(np.max(histories["fitness"][:i])) for i in range(1, row_count + 1)],
                "avg_aesthetic_score": [float(np.mean(histories["aesthetic"][:i])) for i in range(1, row_count + 1)],
                "std_aesthetic_score": [float(np.std(histories["aesthetic"][:i])) for i in range(1, row_count + 1)],
                "max_aesthetic_score": [float(np.max(histories["aesthetic"][:i])) for i in range(1, row_count + 1)],
                "avg_clip_score": [float(np.mean(histories["clip"][:i])) for i in range(1, row_count + 1)],
                "std_clip_score": [float(np.std(histories["clip"][:i])) for i in range(1, row_count + 1)],
                "max_clip_score": [float(np.max(histories["clip"][:i])) for i in range(1, row_count + 1)],
                "avg_image_reward_score": [float(np.mean(histories["image_reward"][:i])) for i in range(1, row_count + 1)],
                "std_image_reward_score": [float(np.std(histories["image_reward"][:i])) for i in range(1, row_count + 1)],
                "max_image_reward_score": [float(np.max(histories["image_reward"][:i])) for i in range(1, row_count + 1)],
                "avg_hpsv2_score": [float(np.mean(histories["hpsv2"][:i])) for i in range(1, row_count + 1)],
                "std_hpsv2_score": [float(np.std(histories["hpsv2"][:i])) for i in range(1, row_count + 1)],
                "max_hpsv2_score": [float(np.max(histories["hpsv2"][:i])) for i in range(1, row_count + 1)],
                "avg_pickscore_score": [float(np.mean(histories["pickscore"][:i])) for i in range(1, row_count + 1)],
                "std_pickscore_score": [float(np.std(histories["pickscore"][:i])) for i in range(1, row_count + 1)],
                "max_pickscore_score": [float(np.max(histories["pickscore"][:i])) for i in range(1, row_count + 1)],
                "elapsed_time": elapsed_times[:row_count],
            })

        for sample_index, sample_seed in enumerate(sample_seeds, start=1):
            elapsed_time = time.time() - start_time
            if self.parameters['time_limit_seconds'] is not None and elapsed_time >= self.parameters['time_limit_seconds']:
                print(
                    "Time limit reached before starting sample "
                    f"{sample_index}/{num_images} (elapsed: {self.format_time(elapsed_time)})."
                )
                break

            print(f"Random sample {sample_index}/{num_images} with generation seed {sample_seed}")

            sample_path = os.path.join(results_folder, f"sample_{sample_index}_seed_{sample_seed}.png")
            fitness, aesthetic_score, clip_score, image_reward_score, hpsv2_score, pickscore_score, _ = self.evaluate(
                trainable_params_init,
                sample_seed,
                text_embeddings_init_shape,
                selected_prompt,
                sample_path,
            )
            positive_fitness = float(-fitness)
            aesthetic_score = float(aesthetic_score)
            clip_score = float(clip_score)
            image_reward_score = float(image_reward_score)
            hpsv2_score = float(hpsv2_score)
            pickscore_score = float(pickscore_score)

            if positive_fitness > best_fitness_overall:
                best_fitness_overall = positive_fitness
                best_sample_path = sample_path
                shutil.copyfile(sample_path, os.path.join(results_folder, f"best_{sample_index}.png"))

            fitness_history.append(positive_fitness)
            aesthetic_history.append(aesthetic_score)
            clip_history.append(clip_score)
            image_reward_history.append(image_reward_score)
            hpsv2_history.append(hpsv2_score)
            pickscore_history.append(pickscore_score)
            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            sample_times.append(elapsed_time)

            sample_rows.append({
                "sample": sample_index,
                "generation": sample_index,
                "seed": sample_seed,
                "prompt": "",
                "fitness": positive_fitness,
                "aesthetic_score": aesthetic_score,
                "clip_score": clip_score,
                "image_reward_score": image_reward_score,
                "hpsv2_score": hpsv2_score,
                "pickscore_score": pickscore_score,
                "elapsed_time": elapsed_time,
            })
            sample_paths.append(sample_path)

            results = build_random_sampler_results({
                "fitness": fitness_history,
                "aesthetic": aesthetic_history,
                "clip": clip_history,
                "image_reward": image_reward_history,
                "hpsv2": hpsv2_history,
                "pickscore": pickscore_history,
            }, time_list, [seed] + sample_seeds[:sample_index])

            if category is not None:
                results["category"] = [category] + [''] * (len(results) - 1)
                sample_rows[-1]["category"] = ""

            results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')
            pd.DataFrame(sample_rows).to_csv(f"{results_folder}/sample_results.csv", index=False, na_rep='nan')
            self._save_population_plot_results(results, results_folder)

            print(
                f"Sample {sample_index}/{num_images}: Fitness: {positive_fitness}, "
                f"Aesthetic score: {aesthetic_score}, CLIP score: {clip_score}, "
                f"ImageReward score: {image_reward_score}, HPSv2 score: {hpsv2_score}, "
                f"PickScore: {pickscore_score}, Best fitness: {best_fitness_overall}"
            )

        shutil.copyfile(best_sample_path, f"{results_folder}/best_all.png")

        canonical_sample_rows = []
        canonical_fitness_history = []
        canonical_aesthetic_history = []
        canonical_clip_history = []
        canonical_image_reward_history = []
        canonical_hpsv2_history = []
        canonical_pickscore_history = []

        canonical_entries = [(0, seed, baseline_path, 0.0)] + [
            (idx, sample_seed, sample_path, elapsed_time)
            for idx, (sample_seed, sample_path, elapsed_time) in enumerate(
                zip(sample_seeds[:len(sample_paths)], sample_paths, sample_times),
                start=1,
            )
        ]

        best_canonical_fitness = -np.inf
        best_canonical_path = baseline_path
        for idx, sample_seed, sample_path, elapsed_time in canonical_entries:
            (
                canonical_fitness,
                canonical_aesthetic_score,
                canonical_clip_score,
                canonical_image_reward_score,
                canonical_hpsv2_score,
                canonical_pickscore_score,
                _,
            ) = self._evaluate_canonical_image_path_scores(sample_path, selected_prompt)

            if float(canonical_fitness) > best_canonical_fitness:
                best_canonical_fitness = float(canonical_fitness)
                best_canonical_path = sample_path

            canonical_fitness_history.append(float(canonical_fitness))
            canonical_aesthetic_history.append(float(canonical_aesthetic_score))
            canonical_clip_history.append(float(canonical_clip_score))
            canonical_image_reward_history.append(float(canonical_image_reward_score))
            canonical_hpsv2_history.append(float(canonical_hpsv2_score))
            canonical_pickscore_history.append(float(canonical_pickscore_score))

            canonical_sample_rows.append({
                "sample": idx,
                "generation": idx,
                "seed": sample_seed,
                "prompt": selected_prompt if idx == 0 else "",
                "fitness": float(canonical_fitness),
                "aesthetic_score": float(canonical_aesthetic_score),
                "clip_score": float(canonical_clip_score),
                "image_reward_score": float(canonical_image_reward_score),
                "hpsv2_score": float(canonical_hpsv2_score),
                "pickscore_score": float(canonical_pickscore_score),
                "elapsed_time": elapsed_time,
            })

        shutil.copyfile(best_canonical_path, f"{results_folder}/best_all.png")

        results = build_random_sampler_results({
            "fitness": canonical_fitness_history,
            "aesthetic": canonical_aesthetic_history,
            "clip": canonical_clip_history,
            "image_reward": canonical_image_reward_history,
            "hpsv2": canonical_hpsv2_history,
            "pickscore": canonical_pickscore_history,
        }, time_list, [seed] + sample_seeds[:len(sample_paths)])

        if category is not None:
            results["category"] = [category] + [''] * (len(canonical_sample_rows) - 1)
            if canonical_sample_rows:
                canonical_sample_rows[0]["category"] = category

        results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')
        pd.DataFrame(canonical_sample_rows).to_csv(f"{results_folder}/sample_results.csv", index=False, na_rep='nan')
        self._save_population_plot_results(results, results_folder)

        return results_folder

    def run_adam_optimization(self, seed = None, seed_number = None, prompt = None, category = None, prompt_number = None):

        def plot_results(results, results_folder):
            plt.figure(figsize=(10, 6))  # Increase figure size
            plt.plot(results['iteration'], results['aesthetic_score'], label="Aesthetic Score")
            plt.xlabel('Iteration')
            plt.ylabel('Aesthetic Score')
            plt.title('Aesthetic Score Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/aesthetic_evolution.png")
            plt.close()

            plt.figure(figsize=(10, 6))  # Increase figure size
            plt.plot(results['iteration'], results['clip_score'], label="CLIP Score")
            plt.xlabel('Iteration')
            plt.ylabel('CLIP Score')
            plt.title('CLIP Score Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/clip_evolution.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(results['iteration'], results['image_reward_score'], label="ImageReward Score")
            plt.xlabel('Iteration')
            plt.ylabel('ImageReward Score')
            plt.title('ImageReward Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(results_folder + "/image_reward_evolution.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(results['iteration'], results['hpsv2_score'], label="HPSv2 Score")
            plt.xlabel('Iteration')
            plt.ylabel('HPSv2 Score')
            plt.title('HPSv2 Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(results_folder + "/hpsv2_evolution.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(results['iteration'], results['pickscore_score'], label="PickScore")
            plt.xlabel('Iteration')
            plt.ylabel('PickScore')
            plt.title('PickScore Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(results_folder + "/pickscore_evolution.png")
            plt.close()

            # Plot all losses in one plot
            plt.figure(figsize=(10, 6))  # Increase figure size
            plt.plot(results['iteration'], results['combined_loss'], label="Combined Loss")
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Loss Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/loss_evolution.png")
            plt.close()

        def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
            lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
            upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]

            plt.plot(x_axis, m_vec, '--', label=description + " Avg.")
            plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3, label=description + " Avg. ± SD")
            if title is not None:
                plt.title(title)
            if y_label is not None:
                plt.ylabel(y_label)
            if x_label is not None:
                plt.xlabel(x_label)
    
        if seed is None:
            seed = self.parameters["seed"]
        if prompt is None:
            selected_prompt = self.parameters["selected_prompt"]
        else:
            selected_prompt = prompt

        seed = int(seed)
        num_iterations = int(self.parameters["num_iterations"])
        adam_lr = float(self.parameters["adam_lr"])
        adam_weight_decay = float(self.parameters["adam_weight_decay"])
        adam_eps = float(self.parameters["adam_eps"])
        adam_beta1 = float(self.parameters["adam_beta1"])
        adam_beta2 = float(self.parameters["adam_beta2"])
        adam_max_grad_norm = self.parameters.get("adam_max_grad_norm", None)
        if adam_max_grad_norm is not None:
            adam_max_grad_norm = float(adam_max_grad_norm)
            if adam_max_grad_norm <= 0:
                adam_max_grad_norm = None

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if category is not None:
            print(f"Selected prompt: {selected_prompt} (Category: {category})")
        else:
            print(f"Selected prompt: {selected_prompt}")

        results_folder = f"{self.OUTPUT_FOLDER}/results_{self.model_name}_{seed}"
        if prompt_number is not None:
            results_folder += f"_{prompt_number}"
        os.makedirs(results_folder, exist_ok=True)
        self._save_run_config(results_folder, seed, selected_prompt, category=category, prompt_number=prompt_number)

        # Text features don't depend on your params; compute w/o grad
        with torch.no_grad():
            if self.clip_model is not None:
                text_tokens = clip.tokenize([selected_prompt]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens).float()
                text_features = F.normalize(text_features, dim=-1, eps=1e-6)
            else:
                text_features = None
            if self.use_image_reward and self.image_reward_model is not None:
                image_reward_text = self.image_reward_model.blip.tokenizer(
                    selected_prompt,
                    padding='max_length',
                    truncation=True,
                    max_length=35,
                    return_tensors="pt",
                ).to(self.device)
                image_reward_prompt_ids = image_reward_text.input_ids
                image_reward_attention_mask = image_reward_text.attention_mask
            else:
                image_reward_prompt_ids = None
                image_reward_attention_mask = None
            if self.use_hpsv2 and self.hpsv2_tokenizer is not None:
                hpsv2_text_tokens = self.hpsv2_tokenizer([selected_prompt]).to(
                    device=self.device,
                    non_blocking=True,
                )
            else:
                hpsv2_text_tokens = None
            if self.use_pickscore and self.pickscore_processor is not None:
                pickscore_text_inputs = self.pickscore_processor(
                    text=selected_prompt,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
            else:
                pickscore_text_inputs = None

        prompt_embeds, pooled_prompt_embeds = self._encode_prompt_embeddings(selected_prompt)
        text_embeddings_init = [
            prompt_embeds.detach().clone().to(torch.float32),
            pooled_prompt_embeds.detach().clone().to(torch.float32),
        ]
        text_embeddings = [
            torch.nn.Parameter(text_embeddings_init[0].clone()),
            torch.nn.Parameter(text_embeddings_init[1].clone()),
        ]

        with torch.no_grad():
            with self._adam_autocast_context():
                initial_image = self.generate_image_from_embeddings_adam(text_embeddings_init, seed)
            image_np = self._tensor_to_uint8_image(initial_image)
            pil_image = Image.fromarray(image_np)
            pil_image.save(f"{results_folder}/it_0.png")

        aesthetic_score = self.aesthetic_evaluation(initial_image)

        clip_score = self.evaluate_clip_score_adam(initial_image, text_features)
        image_reward_score = self.evaluate_image_reward_adam(
            initial_image,
            image_reward_prompt_ids,
            image_reward_attention_mask,
        )
        hpsv2_score = self.evaluate_hpsv2_adam(initial_image, hpsv2_text_tokens)
        pickscore_score = self.evaluate_pickscore_adam(initial_image, pickscore_text_inputs)

        initial_combined_score, _ = self._combine_metric_components({
            "aesthetic_score": aesthetic_score,
            "clip_score": clip_score,
            "image_reward_score": image_reward_score,
            "hpsv2_score": hpsv2_score,
            "pickscore_score": pickscore_score,
        })
        initial_combined_loss = 1 - initial_combined_score
        if not torch.isfinite(initial_combined_loss):
            raise RuntimeError(
                "Initial ADAM objective is non-finite. Try lowering adam_lr or using a more stable torch_dtype."
            )

        combined_score_list = [initial_combined_score.item()]
        combined_loss_list = [initial_combined_loss.item()]
        time_list = [0]
        best_score = initial_combined_score.item()
        best_text_embeddings = [t.detach().clone() for t in text_embeddings_init]

        optimizer = torch.optim.AdamW(
            text_embeddings,
            lr=adam_lr,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_eps,
        )

        start_time = time.time()
        elapsed_time = 0.0

        # Add lists to store the metrics
        aesthetic_score_list = [aesthetic_score.item()]
        clip_score_list = [clip_score.item()]
        image_reward_score_list = [image_reward_score.item()]
        hpsv2_score_list = [hpsv2_score.item()]
        pickscore_score_list = [pickscore_score.item()]

        runtime_results = pd.DataFrame({
            "iteration": [0],
            "prompt": [selected_prompt],
            "combined_score": combined_score_list,
            "combined_loss": combined_loss_list,
            "aesthetic_score": aesthetic_score_list,
            "clip_score": clip_score_list,
            "image_reward_score": image_reward_score_list,
            "hpsv2_score": hpsv2_score_list,
            "pickscore_score": pickscore_score_list,
            "elapsed_time": time_list
        })
        if category is not None:
            runtime_results["category"] = [category]
        runtime_results.to_csv(f"{results_folder}/runtime_score_results.csv", index=False, na_rep='nan')

        for iteration in range(1, num_iterations + 1):
            if self.parameters['time_limit_seconds'] is not None and elapsed_time >= self.parameters['time_limit_seconds']:
                print(
                    "Time limit reached before starting iteration "
                    f"{iteration}/{num_iterations} (elapsed: {self.format_time(elapsed_time)})."
                )
                break
            print(f"Iteration {iteration}/{num_iterations}")

            optimizer.zero_grad(set_to_none=True)

            with self._adam_autocast_context():
                image = self.generate_image_from_embeddings_adam(text_embeddings, seed)
            aesthetic_score = self.aesthetic_evaluation(image)
            clip_score = self.evaluate_clip_score_adam(image, text_features)
            image_reward_score = self.evaluate_image_reward_adam(
                image,
                image_reward_prompt_ids,
                image_reward_attention_mask,
            )
            hpsv2_score = self.evaluate_hpsv2_adam(image, hpsv2_text_tokens)
            pickscore_score = self.evaluate_pickscore_adam(image, pickscore_text_inputs)
            combined_score, _ = self._combine_metric_components({
                "aesthetic_score": aesthetic_score,
                "clip_score": clip_score,
                "image_reward_score": image_reward_score,
                "hpsv2_score": hpsv2_score,
                "pickscore_score": pickscore_score,
            })
            combined_loss = 1 - combined_score
            if not torch.isfinite(combined_loss):
                print(
                    f"Non-finite objective at iteration {iteration}. "
                    "Stopping early and keeping best finite embedding found so far."
                )
                break

            # Calculate gradients
            combined_loss.backward()
            if adam_max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(text_embeddings, max_norm=adam_max_grad_norm)
            # Update parameters
            optimizer.step()

            # Append metrics to their respective lists
            aesthetic_score_list.append(aesthetic_score.item())
            clip_score_list.append(clip_score.item())
            image_reward_score_list.append(image_reward_score.item())
            hpsv2_score_list.append(hpsv2_score.item())
            pickscore_score_list.append(pickscore_score.item())

            if combined_score.item() > best_score:
                best_score = combined_score.item()
                best_text_embeddings = [t.detach().clone() for t in text_embeddings]

            combined_score_list.append(combined_score.item())
            combined_loss_list.append(combined_loss.item())

            image_np = self._tensor_to_uint8_image(image)
            pil_image = Image.fromarray(image_np)
            pil_image.save(f"{results_folder}/it_{iteration}.png")

            elapsed_time = time.time() - start_time
            iterations_done = iteration
            iterations_left = num_iterations - iteration
            average_time_per_iteration = elapsed_time / iterations_done
            estimated_time_remaining = average_time_per_iteration * iterations_left

            formatted_time_remaining = self.format_time(estimated_time_remaining)

            time_list.append(elapsed_time)

            # Save the differentiable in-optimization metrics separately from canonical scores.
            runtime_results = pd.DataFrame({
                "iteration": list(range(0, iteration + 1)),
                "prompt": [selected_prompt] + [''] * iteration,
                "combined_score": combined_score_list,
                "combined_loss": combined_loss_list,
                "aesthetic_score": aesthetic_score_list,
                "clip_score": clip_score_list,
                "image_reward_score": image_reward_score_list,
                "hpsv2_score": hpsv2_score_list,
                "pickscore_score": pickscore_score_list,
                "elapsed_time": time_list
            })

            if category is not None:
                runtime_results["category"] = [category] + [''] * iteration

            runtime_results.to_csv(f"{results_folder}/runtime_score_results.csv", index=False, na_rep='nan')

            # Plot and save the fitness evolution
            plot_results(runtime_results, results_folder)

            # Print stats
            print(f"Iteration {iteration}/{num_iterations}: Combined Score: {combined_score.item()}, Aesthetic Score: {aesthetic_score.item()}, CLIP Score: {clip_score.item()}, ImageReward Score: {image_reward_score.item()}, HPSv2 Score: {hpsv2_score.item()}, PickScore: {pickscore_score.item()}, Estimated time remaining: {formatted_time_remaining}")

        # Save the overall best image
        with torch.no_grad():
            with self._adam_autocast_context():
                best_image = self.generate_image_from_embeddings_adam(best_text_embeddings, seed)
        best_image_np = self._tensor_to_uint8_image(best_image)
        pil_image = Image.fromarray(best_image_np)
        pil_image.save(f"{results_folder}/best_all.png")

        canonical_rows = []
        for row_idx in range(len(combined_score_list)):
            image_path = f"{results_folder}/it_{row_idx}.png"
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Cannot build canonical score_results.csv because {image_path} is missing."
                )
            pil_image = Image.open(image_path).convert("RGB")
            image = self._uint8_image_to_tensor(pil_image)
            (
                canonical_score,
                canonical_aesthetic_score,
                canonical_clip_score,
                canonical_image_reward_score,
                canonical_hpsv2_score,
                canonical_pickscore_score,
                _,
            ) = self._evaluate_canonical_image_scores(image, selected_prompt)
            canonical_rows.append({
                "iteration": row_idx,
                "prompt": selected_prompt if row_idx == 0 else "",
                "combined_score": canonical_score,
                "combined_loss": 1 - canonical_score,
                "aesthetic_score": canonical_aesthetic_score,
                "clip_score": canonical_clip_score,
                "image_reward_score": canonical_image_reward_score,
                "hpsv2_score": canonical_hpsv2_score,
                "pickscore_score": canonical_pickscore_score,
                "elapsed_time": time_list[row_idx] if row_idx < len(time_list) else np.nan,
            })

        results = pd.DataFrame(canonical_rows)
        if category is not None and not results.empty:
            results["category"] = [category] + [''] * (len(results) - 1)

        results.to_csv(f"{results_folder}/score_results.csv", index=False, na_rep='nan')
        plot_results(results, results_folder)

        return results_folder
