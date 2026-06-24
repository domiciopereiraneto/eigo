"""Improved (LAION v2) aesthetic predictor."""

import contextlib
from pathlib import Path
from urllib.request import urlretrieve

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(values, axis=-1):
    norm = values.norm(p=2, dim=axis, keepdim=True).clamp(min=1e-8)
    return values / norm


class _ImprovedMLP(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class LAIONV2Aesthetic:
    def __init__(
        self,
        device=None,
        *,
        clip_model="ViT-L/14",
        weight_variant="sac+logos+ava1-l14-linearMSE.pth",
        cache_dir=None,
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.clip_model_name = clip_model
        self.clip_model, self._preprocess_pil = clip.load(clip_model, device=self.device)
        self.clip_model.eval().requires_grad_(False)

        cache_path = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "emb_reader"
        weights_path = cache_path / weight_variant
        if not weights_path.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://huggingface.co/camenduru/improved-aesthetic-predictor/resolve/main/"
                + weight_variant
            )
            urlretrieve(url, weights_path)

        self.mlp = _ImprovedMLP(embedding_dim=768).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.mlp.load_state_dict(state_dict, strict=True)
        self.mlp.eval().requires_grad_(False)

        self._mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=self.device
        ).view(-1, 1, 1)
        self._std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=self.device
        ).view(-1, 1, 1)

    @staticmethod
    def _resize_shorter_side(image, target=224):
        _, height, width = image.shape
        scale = target / min(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))
        return F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def _preprocess_tensor(self, image):
        image = self._resize_shorter_side(image.to(self.device))
        _, height, width = image.shape
        top = (height - 224) // 2
        left = (width - 224) // 2
        image = image[:, top : top + 224, left : left + 224]
        return (image - self._mean) / self._std

    def _embed(self, image, *, no_grad=True):
        context = torch.no_grad() if no_grad else contextlib.nullcontext()
        with context:
            embedding = self.clip_model.encode_image(image).float()
        return _l2_normalize(embedding)

    def predict_from_pil(self, image):
        image_tensor = self._preprocess_pil(image).unsqueeze(0).to(self.device)
        embedding = self._embed(image_tensor, no_grad=True)
        with torch.no_grad():
            return self.mlp(embedding).squeeze()

    def predict_from_tensor(self, image_tensor, dtype=torch.float32):
        image_tensor = self._preprocess_tensor(image_tensor).unsqueeze(0)
        embedding = self._embed(image_tensor, no_grad=False)
        return self.mlp(embedding).to(dtype).squeeze()
