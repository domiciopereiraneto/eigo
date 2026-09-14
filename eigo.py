"""Discrete text encoder token optimization for diffusion image generation."""

import sys
import os
import shutil
import gc
import yaml
import torch
import numpy as np
import pandas as pd
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    PixArtAlphaPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
try:
    from diffusers import SanaPipeline, SanaSprintPipeline
except ImportError:
    SanaPipeline = None
    SanaSprintPipeline = None
from PIL import Image
import matplotlib.pyplot as plt
import time
import clip
import re
import ast
import inspect
from io import BytesIO
from contextlib import nullcontext
from pathlib import Path
from transformers import AutoModel, AutoProcessor
from src.optimization_targets import TokenSpace, resolve_optimization_target
from src.token_optimizers import optimize
import json

from src.aesthetic_evaluation import (
    LAIONAesthetic,
    LAIONV2Aesthetic,
    SimulacraAesthetic,
)


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
        if "aesthetic_predictor" not in config_parameters and "predictor" in config_parameters:
            config_parameters["aesthetic_predictor"] = config_parameters["predictor"]
        elif (
            "aesthetic_predictor" in config_parameters
            and "predictor" in config_parameters
            and config_parameters["aesthetic_predictor"] != config_parameters["predictor"]
        ):
            raise ValueError("Use either aesthetic_predictor or predictor, not conflicting values.")
        elif "aesthetic_predictor" not in config_parameters:
            config_parameters["aesthetic_predictor"] = 2
        self.model_backend = self._resolve_model_backend(config_parameters)
        self.optimization_target = resolve_optimization_target(config_parameters)
        if config_parameters.get("optimization_method") not in {"ga", "gomea", "random_sampler"}:
            raise ValueError("optimization_method must be ga, gomea, or random_sampler")
        self.guidance_scale = float(config_parameters.get("guidance_scale", 0.0))
        self.lcm_origin_steps = int(config_parameters.get("lcm_origin_steps", 50))
        if self.lcm_origin_steps <= 0:
            raise ValueError("lcm_origin_steps must be a positive integer.")
        self.max_sequence_length = int(config_parameters.get("max_sequence_length", 512))
        self.model_dtype = self._resolve_model_dtype(config_parameters)
        self.use_multi_gpu = bool(config_parameters.get("use_multi_gpu", False))
        self.pipeline_device_map = config_parameters.get("pipeline_device_map", "balanced")
        self.max_memory = self._resolve_max_memory(config_parameters.get("max_memory", None))
        self.base_model_id = self._resolve_base_model_id(config_parameters)
        self.unet_model_id = self._resolve_unet_model_id(config_parameters)
        self.unet_subfolder = str(config_parameters.get("unet_subfolder", "unet"))
        self.enable_attention_slicing = bool(config_parameters.get("enable_attention_slicing", False))
        self.enable_vae_slicing = bool(config_parameters.get("enable_vae_slicing", False))
        self.enable_vae_tiling = bool(config_parameters.get("enable_vae_tiling", False))
        self._active_prompt_attention_mask = None
        self._active_negative_prompt_embeds = None
        self._active_negative_prompt_attention_mask = None
        self._warned_clip_truncation_prompts = set()
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
        self.jpeg_size_weight = float(config_parameters.get("jpeg_size_weight", 0.0))
        self.jpeg_quality = int(config_parameters.get("jpeg_quality", 95))
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100.")
        self.evaluate_zero_weight_metrics = bool(config_parameters.get("evaluate_zero_weight_metrics", False))
        self._metric_scales = {
            "aesthetic_score": float(config_parameters.get("max_aesthetic_score", 1.0)),
            "clip_score": float(config_parameters.get("max_clip_score", 1.0)),
            "image_reward_score": float(config_parameters.get("max_image_reward_score", 1.0)),
            "hpsv2_score": float(config_parameters.get("max_hpsv2_score", 1.0)),
            "pickscore_score": float(config_parameters.get("max_pickscore_score", 1.0)),
            "jpeg_size_kb": float(config_parameters.get("max_jpeg_size_kb", 1024.0)),
        }
        if self._metric_scales["jpeg_size_kb"] <= 0:
            raise ValueError("max_jpeg_size_kb must be greater than 0.")
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
        clip_model_name = config_parameters.get("clip_model_name", "ViT-L/14")
        self.clip_model_name = None if clip_model_name is None else str(clip_model_name)
        if self.clip_model_name is None and (
            self._should_evaluate_metric("clip_score")
            or (
                config_parameters["aesthetic_predictor"] in (1, 2)
                and self._should_evaluate_metric("aesthetic_score")
            )
        ):
            raise ValueError(
                "clip_model_name must be configured when CLIP score or LAION aesthetic score is evaluated."
            )

        if config_parameters["aesthetic_predictor"] == 0:
            predictor_name = 'simulacra'
        elif config_parameters["aesthetic_predictor"] == 1:
            predictor_name = 'laionv1'
        elif config_parameters["aesthetic_predictor"] == 2:
            predictor_name = 'laionv2'
        else:
            raise ValueError("Invalid aesthetic_predictor option.")

        method_save_name = config_parameters["optimization_method"]

        model_tag = self._model_id_tag(config_parameters["model_id"])
        self.OUTPUT_FOLDER = (
            f"{config_parameters['results_folder']}/"
            f"{method_save_name}_{self.optimization_target}_clip_{predictor_name}_{self.model_backend}_{model_tag}_"
            f"{config_parameters['seed']}_"
            f"aesw{int(self.aesthetic_score_weight*100)}_"
            f"clipw{int(self.clip_score_weight*100)}_"
            f"irw{int(self.image_reward_score_weight*100)}_"
            f"hpsw{int(self.hpsv2_score_weight*100)}_"
            f"psw{int(self.pickscore_score_weight*100)}_"
            f"jpgw{int(self.jpeg_size_weight*100)}"
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
            self.base_model_id,
            self.unet_model_id,
            self.unet_subfolder,
            self.model_backend,
            str(self.model_dtype),
            self.device,
            self.use_multi_gpu,
            str(self.pipeline_device_map),
            str(self.max_memory),
            config_parameters["aesthetic_predictor"],
            self.use_image_reward,
            self.image_reward_model_name if self.use_image_reward else None,
            self.use_hpsv2,
            self.hpsv2_version if self.use_hpsv2 else None,
            self.use_pickscore,
            self.pickscore_processor_name if self.use_pickscore else None,
            self.pickscore_model_name if self.use_pickscore else None,
            self.clip_model_name,
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

            clip_model = None
            clip_preprocess = None
            if self.clip_model_name is not None and self._should_evaluate_metric("clip_score"):
                clip_model, clip_preprocess = clip.load(self.clip_model_name, device=self.device)
                self._freeze_module_params(clip_model)

            aesthetic_model = None
            if config_parameters["aesthetic_predictor"] == 0:
                model_name = "SAM"
                if self._should_evaluate_metric("aesthetic_score"):
                    aesthetic_model = SimulacraAesthetic(self.device)
            elif config_parameters["aesthetic_predictor"] == 1:
                model_name = "LAIONV1"
                if self._should_evaluate_metric("aesthetic_score"):
                    aesthetic_model = LAIONAesthetic(self.device, clip_model=self.clip_model_name)
            elif config_parameters["aesthetic_predictor"] == 2:
                model_name = "LAIONV2"
                if self._should_evaluate_metric("aesthetic_score"):
                    aesthetic_model = LAIONV2Aesthetic(self.device, clip_model=self.clip_model_name)
            else:
                raise ValueError("Invalid aesthetic_predictor option.")

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
            "jpeg_size_kb": self.jpeg_size_weight,
        }
        return metric_weights.get(metric_name, 0.0) != 0.0

    @staticmethod
    def _resolve_model_backend(config_parameters):
        requested = str(config_parameters.get("model_backend", "auto")).lower().replace("-", "_")
        if requested in ("sd", "stable_diffusion", "sdxl", "flux", "pixart", "lcm", "sana", "sana_sprint"):
            if requested == "stable_diffusion":
                return "sd"
            return requested
        if requested != "auto":
            raise ValueError(
                f"Invalid model_backend '{requested}'. Expected one of: auto, sd, sdxl, flux, pixart, lcm, sana, sana_sprint."
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
        if (
            "stable-diffusion-xl" in model_id
            or "stable_diffusion_xl" in model_id
            or "sdxl" in model_id
        ):
            return "sdxl"
        if (
            "stable-diffusion-v1" in model_id
            or "stable-diffusion-1" in model_id
            or "stable-diffusion-v2" in model_id
            or "stable-diffusion-2" in model_id
            or "stable-diffusion" in model_id
            or "sd-v1" in model_id
            or "sd-v2" in model_id
            or "sd-turbo" in model_id
            or "dpo-sd1.5" in model_id
        ):
            return "sd"
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
    def _is_dpo_sd15_model(model_id):
        return str(model_id).lower().strip().rstrip("/") == "mhdang/dpo-sd1.5-text2image-v1"

    def _resolve_base_model_id(self, config_parameters):
        base_model_id = config_parameters.get("base_model_id", None)
        if base_model_id is not None:
            return str(base_model_id)
        if self.model_backend == "sd" and self._is_dpo_sd15_model(config_parameters["model_id"]):
            return "runwayml/stable-diffusion-v1-5"
        return None

    def _resolve_unet_model_id(self, config_parameters):
        unet_model_id = config_parameters.get("unet_model_id", None)
        if unet_model_id is not None:
            return str(unet_model_id)
        if self.model_backend == "sd" and self._is_dpo_sd15_model(config_parameters["model_id"]):
            return str(config_parameters["model_id"])
        return None

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


    def _configure_pipeline_output_options(self, pipe):
        if self.model_backend in ("sd", "lcm") and hasattr(pipe, "safety_checker"):
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
        if self.model_backend == "sd":
            pipeline_model_id = self.base_model_id or model_id
            pipe = StableDiffusionPipeline.from_pretrained(pipeline_model_id, **common_kwargs)
            if self.unet_model_id is not None:
                unet_kwargs = dict(common_kwargs)
                unet = UNet2DConditionModel.from_pretrained(
                    self.unet_model_id,
                    subfolder=self.unet_subfolder,
                    **unet_kwargs,
                )
                pipe.unet = unet
            return pipe, is_sharded
        if self.model_backend == "sdxl":
            return StableDiffusionXLPipeline.from_pretrained(model_id, **common_kwargs), is_sharded
        raise ValueError(f"Unsupported model backend: {self.model_backend}")

    def _pipeline_input_device(self):
        execution_device = getattr(self.pipe, "_execution_device", None)
        if execution_device is not None:
            return str(execution_device)
        return self.device

    def _encode_prompt_embeddings(self, prompt):
        self._active_prompt_attention_mask = None
        self._active_negative_prompt_embeds = None
        self._active_negative_prompt_attention_mask = None

        encode_kwargs = {
            "prompt": prompt,
            "device": self._pipeline_input_device(),
            "num_images_per_prompt": 1,
        }

        if self.model_backend in ("sd", "sdxl"):
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

        if self.model_backend == "sd":
            if len(encoded) < 1:
                raise RuntimeError("Unexpected Stable Diffusion encode_prompt output length.")
            if self.guidance_scale > 1.0:
                if len(encoded) < 2 or encoded[1] is None:
                    raise RuntimeError(
                        "Stable Diffusion negative prompt embeddings are required when guidance_scale > 1."
                    )
                return encoded[0], encoded[1]
            # Keep the optimization loop shape contract unchanged when CFG is disabled.
            aux = torch.zeros((1, 1), dtype=encoded[0].dtype, device=self.device)
            return encoded[0], aux

        if len(encoded) >= 4:
            # SDXL: prompt, negative_prompt, pooled_prompt, negative_pooled_prompt
            return encoded[0], encoded[2]
        if len(encoded) >= 2:
            # FLUX: prompt, pooled_prompt, text_ids
            return encoded[0], encoded[1]

        raise RuntimeError("Unexpected encode_prompt output length.")

    def _tokenize_clip_text(self, prompt):
        if self.clip_model is None:
            return None

        context_length = int(getattr(self.clip_model, "context_length", 77))
        try:
            return clip.tokenize([prompt], context_length=context_length).to(self.device)
        except RuntimeError as exc:
            if "is too long for context length" not in str(exc):
                raise
            if prompt not in self._warned_clip_truncation_prompts:
                print(
                    f"Warning: truncating CLIP scoring prompt to {context_length} tokens."
                )
                self._warned_clip_truncation_prompts.add(prompt)
            return clip.tokenize(
                [prompt],
                context_length=context_length,
                truncate=True,
            ).to(self.device)

    def _lcm_origin_steps_call_key(self):
        try:
            call_parameters = inspect.signature(self.pipe.__class__.__call__).parameters
        except (TypeError, ValueError):
            return "lcm_origin_steps"
        if "original_inference_steps" in call_parameters:
            return "original_inference_steps"
        return "lcm_origin_steps"


    @staticmethod
    def _repeat_to_batch(tensor, batch_size):
        if tensor is None or batch_size == 1:
            return tensor
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Cannot expand tensor with batch dimension {tensor.shape[0]} to batch_size {batch_size}."
            )
        repeat_shape = [batch_size] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeat_shape)

    def _generators_for_batch(self, seed, batch_size):
        generator_device = self._pipeline_input_device()
        if isinstance(seed, (list, tuple, np.ndarray)):
            if len(seed) != batch_size:
                raise ValueError("Seed lists must match the generation batch size.")
            generators = [
                torch.Generator(device=generator_device).manual_seed(int(seed_value))
                for seed_value in seed
            ]
            return generators[0] if batch_size == 1 else generators
        if batch_size == 1:
            return torch.Generator(device=generator_device).manual_seed(int(seed))
        return [
            torch.Generator(device=generator_device).manual_seed(int(seed))
            for _ in range(batch_size)
        ]

    def _build_generation_kwargs(self, prompt_embeds, pooled_prompt_embeds, generator, latents=None):
        prompt_device = self._pipeline_input_device()
        batch_size = int(prompt_embeds.shape[0])
        prompt_embeds = prompt_embeds.to(device=prompt_device, dtype=self.model_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=prompt_device, dtype=self.model_dtype)
        if latents is not None:
            latents = latents.to(device=prompt_device, dtype=self.model_dtype)
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
            kwargs["prompt_attention_mask"] = self._repeat_to_batch(
                self._active_prompt_attention_mask,
                batch_size,
            ).to(prompt_device)
            if self.model_backend == "sana":
                if self.guidance_scale > 1.0 and self._active_negative_prompt_embeds is None:
                    raise RuntimeError("Sana negative prompt embeddings are required when guidance_scale > 1.")
                if self._active_negative_prompt_embeds is not None:
                    kwargs["negative_prompt_embeds"] = self._repeat_to_batch(
                        self._active_negative_prompt_embeds,
                        batch_size,
                    ).to(
                        device=prompt_device,
                        dtype=self.model_dtype,
                    )
                if self._active_negative_prompt_attention_mask is not None:
                    kwargs["negative_prompt_attention_mask"] = self._repeat_to_batch(
                        self._active_negative_prompt_attention_mask,
                        batch_size,
                    ).to(prompt_device)
        elif self.model_backend == "lcm":
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs[self._lcm_origin_steps_call_key()] = self.lcm_origin_steps
        elif self.model_backend == "sd":
            kwargs["prompt_embeds"] = prompt_embeds
            if self.guidance_scale > 1.0:
                kwargs["negative_prompt_embeds"] = pooled_prompt_embeds
        else:
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds

        if self.model_backend in ("flux", "sana", "sana_sprint"):
            kwargs["max_sequence_length"] = self.max_sequence_length
        if latents is not None:
            kwargs["latents"] = latents

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
                    "out-of-memory error internally. For Sana Sprint, reduce memory "
                    "pressure by using torch_dtype: auto or bfloat16, setting "
                    "evaluate_zero_weight_metrics: false, "
                    "lowering height/width, or enabling VAE tiling/slicing."
                ) from exc
            raise
        
    def generate_image_from_tensors(self, prompt_embeds, pooled_prompt_embeds, seed, latents=None):
        generator = self._generators_for_batch(seed, int(prompt_embeds.shape[0]))

        out = self._call_generation_pipeline(
            self.pipe,
            self._build_generation_kwargs(prompt_embeds, pooled_prompt_embeds, generator, latents=latents),
        )

        image = out.clamp(0, 1).squeeze(0).permute(1, 2, 0)      # HWC
        return image.to(self.device)

    def generate_images_from_tensors(self, prompt_embeds, pooled_prompt_embeds, seed, latents=None):
        batch_size = int(prompt_embeds.shape[0])
        generator = self._generators_for_batch(seed, batch_size)

        out = self._call_generation_pipeline(
            self.pipe,
            self._build_generation_kwargs(prompt_embeds, pooled_prompt_embeds, generator, latents=latents),
        )

        if out.ndim == 3:
            out = out.unsqueeze(0)
        return out.clamp(0, 1).permute(0, 2, 3, 1).to(self.device)


    @staticmethod
    def _tensor_to_uint8_image(image_tensor):
        image_np = image_tensor.detach().to(torch.float32).cpu().numpy()
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
        image_np = np.clip(image_np, 0.0, 1.0)
        return (image_np * 255).astype(np.uint8)

    def _jpeg_bytes(self, image):
        if torch.is_tensor(image):
            image = Image.fromarray(self._tensor_to_uint8_image(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8, copy=False))
        image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        return buffer.getvalue()

    def _jpeg_size_kb(self, image):
        return float(len(self._jpeg_bytes(image)) / 1024.0)

    def _save_jpeg(self, image, path):
        if not str(path).lower().endswith((".jpg", ".jpeg")):
            raise ValueError(f"Generated images must use a .jpg or .jpeg extension: {path}")
        jpeg_bytes = self._jpeg_bytes(image)
        with open(path, "wb") as file:
            file.write(jpeg_bytes)
        return float(len(jpeg_bytes) / 1024.0)

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

        if self.parameters["aesthetic_predictor"] == 0:
            # Simulacra Aesthetic Model
            score = self.aesthetic_model.predict_from_tensor(image_input)
        elif self.parameters["aesthetic_predictor"] == 1 or self.parameters["aesthetic_predictor"] == 2:
            # LAION Aesthetic Predictor V1 and V2
            score = self.aesthetic_model.predict_from_tensor(image_input)
        else:
            return torch.tensor(0.0, device=self.device)

        return score

    def evaluate_clip_score(self, image_tensor, prompt):
        if self.clip_model is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Convert the image tensor to a PIL image
        image = (image_tensor * 255).clamp(0, 255).byte()
        image = Image.fromarray(image.cpu().numpy())

        # Preprocess the image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize the prompt
        text_input = self._tokenize_clip_text(prompt)

        # Compute the CLIP embeddings
        image_features = self.clip_model.encode_image(image_input)
        text_features = self.clip_model.encode_text(text_input)

        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute the cosine similarity (CLIP score)
        clip_score = (image_features @ text_features.T)

        return clip_score


    def evaluate_image_reward(self, image_tensor, prompt):
        if not self.use_image_reward or self.image_reward_model is None:
            return 0.0

        pil_image = Image.fromarray(self._tensor_to_uint8_image(image_tensor))
        return float(self.image_reward_model.score(prompt, pil_image))


    def evaluate_hpsv2(self, image_tensor, prompt):
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

    def evaluate_pickscore(self, image_tensor, prompt):
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


    def _combine_metric_components(self, metric_values):
        metric_weights = {
            "aesthetic_score": self.aesthetic_score_weight,
            "clip_score": self.clip_score_weight,
            "image_reward_score": self.image_reward_score_weight,
            "hpsv2_score": self.hpsv2_score_weight,
            "pickscore_score": self.pickscore_score_weight,
            "jpeg_size_kb": self.jpeg_size_weight,
        }
        components = {}
        total = None
        for metric_name, raw_value in metric_values.items():
            weight = metric_weights.get(metric_name, 0.0)
            scale = self._metric_scales.get(metric_name, 1.0)
            direction = -1.0 if metric_name == "jpeg_size_kb" else 1.0
            component = direction * weight * raw_value / scale
            components[metric_name] = component
            total = component if total is None else total + component
        return total, components

    def _evaluate_canonical_image_scores(self, image, selected_prompt, jpeg_size_kb=None):
        with torch.no_grad():
            aesthetic_score = self.aesthetic_evaluation(image).item()
            clip_score = self.evaluate_clip_score(image, selected_prompt).item()
            image_reward_score = self.evaluate_image_reward(image, selected_prompt)
            hpsv2_score = self.evaluate_hpsv2(image, selected_prompt)
            pickscore_score = self.evaluate_pickscore(image, selected_prompt)
            if jpeg_size_kb is None:
                jpeg_size_kb = self._jpeg_size_kb(image)

        combined_score, components = self._combine_metric_components({
            "aesthetic_score": aesthetic_score,
            "clip_score": clip_score,
            "image_reward_score": image_reward_score,
            "hpsv2_score": hpsv2_score,
            "pickscore_score": pickscore_score,
            "jpeg_size_kb": jpeg_size_kb,
        })

        return (
            float(combined_score),
            aesthetic_score,
            clip_score,
            image_reward_score,
            hpsv2_score,
            pickscore_score,
            jpeg_size_kb,
            components,
        )

    def _evaluate_canonical_image_path_scores(self, image_path, selected_prompt):
        image = self._load_saved_image_tensor(image_path)
        jpeg_size_kb = float(os.path.getsize(image_path) / 1024.0)
        return self._evaluate_canonical_image_scores(image, selected_prompt, jpeg_size_kb=jpeg_size_kb)


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

    def _reset_peak_vram(self):
        if not torch.cuda.is_available() or torch.device(self.device).type != "cuda":
            return
        torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_vram_mb(self):
        if not torch.cuda.is_available() or torch.device(self.device).type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

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
        run_config["resolved_optimization_target"] = self.optimization_target

        if category is not None:
            run_config["category"] = category
        if prompt_number is not None:
            run_config["prompt_number"] = prompt_number

        config_save_path = os.path.join(results_folder, "config.yaml")
        with open(config_save_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(run_config, file, sort_keys=False, allow_unicode=True)


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
        plt.savefig(results_folder + "/fitness_evolution.jpg")
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
        plt.savefig(results_folder + "/aesthetic_score_evolution.jpg")
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
        plt.savefig(results_folder + "/clip_score_evolution.jpg")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_image_reward_score'], results['std_image_reward_score'], "Population")
        plt.plot(results['generation'], results['max_image_reward_score'], 'r-', label="Best")
        plt.xlabel('Generation')
        plt.ylabel('ImageReward Score')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/image_reward_evolution.jpg")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_hpsv2_score'], results['std_hpsv2_score'], "Population")
        plt.plot(results['generation'], results['max_hpsv2_score'], 'r-', label="Best")
        plt.xlabel('Generation')
        plt.ylabel('HPSv2 Score')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/hpsv2_evolution.jpg")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_pickscore_score'], results['std_pickscore_score'], "Population")
        plt.plot(results['generation'], results['max_pickscore_score'], 'r-', label="Best")
        plt.xlabel('Generation')
        plt.ylabel('PickScore')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/pickscore_evolution.jpg")
        plt.close()

        plt.figure(figsize=(10, 6))
        plot_mean_std(results['generation'], results['avg_jpeg_size_kb'], results['std_jpeg_size_kb'], "Population")
        plt.plot(results['generation'], results['min_jpeg_size_kb'], 'r-', label="Smallest")
        plt.xlabel('Generation')
        plt.ylabel('JPEG Size (KB)')
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(results_folder + "/jpeg_size_evolution.jpg")
        plt.close()

    def run_ga_optimization(self, **kwargs):
        return self._run_token_optimization('ga', **kwargs)

    def run_gomea_optimization(self, **kwargs):
        return self._run_token_optimization('gomea', **kwargs)

    def run_random_sampler_optimization(self, **kwargs):
        return self._run_token_optimization('random_sampler', **kwargs)

    def _run_token_optimization(self, method, seed=None, seed_number=None, prompt=None,
                                category=None, prompt_number=None):
        seed = int(self.parameters['seed'] if seed is None else seed)
        prompt = self.parameters['selected_prompt'] if prompt is None else prompt
        folder = Path(self.OUTPUT_FOLDER) / f'results_{self.model_name}_{seed}'
        if prompt_number is not None:
            folder = folder.with_name(folder.name + f'_{prompt_number}')
        folder.mkdir(parents=True, exist_ok=True)
        self._save_run_config(str(folder), seed, prompt, category, prompt_number)
        torch.manual_seed(seed)
        space = TokenSpace(self, prompt)
        (folder / 'initial_tokens.json').write_text(json.dumps(space.artifact(space.initial), indent=2))
        rows, metrics, candidate_rows = [], [], []
        start = time.monotonic()
        self._reset_peak_vram()
        best_image, best_value = None, float('inf')

        def evaluate(vector, generation):
            nonlocal best_image, best_value
            with torch.no_grad():
                pe, ppe = space.encode(vector)
                image = self.generate_image_from_tensors(pe, ppe, seed)
            path = folder / 'it_0.jpg'
            if generation:
                path = folder / f'gen_{generation}' / f'id_{len(metrics) + 1}.jpg'
            if generation == 0 or self.parameters.get('save_gens', True):
                path.parent.mkdir(parents=True, exist_ok=True)
                self._save_jpeg(image, str(path))
            # Score the exact JPEG representation used in reports, even when
            # individual images are not retained.
            with Image.open(BytesIO(self._jpeg_bytes(image))) as jpeg:
                canonical = self._uint8_image_to_tensor(jpeg.convert('RGB'))
            values = self._evaluate_canonical_image_scores(canonical, prompt, self._jpeg_size_kb(image))
            metrics.append(values[:7])
            value = -values[0]
            if not np.isfinite(value):
                raise ValueError('Candidate fitness is not finite.')
            if value < best_value:
                best_image, best_value = image.detach().clone(), value
            candidate_rows.append({'generation': generation, 'fitness': values[0],
                                   'seed': seed, 'tokens': vector.tolist()})
            return value

        def record(generation, best, best_score, evaluations):
            row = {'generation': generation, 'prompt': prompt if generation == 0 else '',
                   'elapsed_time': time.monotonic() - start, 'peak_vram_mb': self._peak_vram_mb(),
                   'evaluations': evaluations, 'best_fitness': -best_score}
            names = ['fitness', 'aesthetic_score', 'clip_score', 'image_reward_score',
                     'hpsv2_score', 'pickscore_score', 'jpeg_size_kb']
            for i, name in enumerate(names):
                values = [m[i] for m in metrics]
                row['avg_' + name] = float(np.mean(values)) if values else np.nan
                row['std_' + name] = float(np.std(values)) if values else np.nan
                prefix = 'min_' if name == 'jpeg_size_kb' else 'max_'
                row[prefix + name] = float((min if name == 'jpeg_size_kb' else max)(values)) if values else np.nan
            if category is not None: row['category'] = category
            rows.append(row)
            self._save_jpeg(best_image, str(folder / f'best_{generation}.jpg'))
            self._save_jpeg(best_image, str(folder / 'best_all.jpg'))
            (folder / 'best_tokens.json').write_text(json.dumps(space.artifact(best), indent=2))
            pd.DataFrame(rows).to_csv(folder / 'fitness_results.csv', index=False)
            with (folder / 'candidates.jsonl').open('a' if generation else 'w') as stream:
                for candidate in candidate_rows:
                    stream.write(json.dumps(candidate) + '\n')
            metrics.clear(); candidate_rows.clear()
            print(f'Generation {generation}: best fitness {-best_score:.6f}, evaluations {evaluations}')

        optimize(space, method, self.parameters, seed, evaluate, record)
        self._save_population_plot_results(pd.DataFrame(rows), str(folder))
        return str(folder)
