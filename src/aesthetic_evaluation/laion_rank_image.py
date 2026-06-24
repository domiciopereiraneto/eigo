"""LAION v1 aesthetic predictor."""

from pathlib import Path
from urllib.request import urlretrieve

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class LAIONAesthetic:
    def __init__(self, device, clip_model="ViT-L/14"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if clip_model not in {"ViT-L/14", "ViT-B/32"}:
            raise ValueError(f"Unsupported clip model: {clip_model}")

        self.clip_model_name = clip_model
        self.clip_model = clip.load(clip_model, jit=False, device=self.device)[0]
        self.clip_model.eval().requires_grad_(False)

        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.preprocess_pil = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        model_variant = "vit_l_14" if clip_model == "ViT-L/14" else "vit_b_32"
        embedding_dim = 768 if model_variant == "vit_l_14" else 512
        weights_path = Path.home() / ".cache" / "emb_reader" / f"sa_0_4_{model_variant}_linear.pth"
        if not weights_path.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://github.com/LAION-AI/aesthetic-predictor/raw/main/"
                f"sa_0_4_{model_variant}_linear.pth"
            )
            urlretrieve(url, weights_path)

        self.model = nn.Linear(embedding_dim, 1)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval().requires_grad_(False).to(self.device)

    def preprocess_tensor_diff(self, img_tensor):
        _, height, width = img_tensor.shape
        scale = 224 / min(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))
        resized = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        top = (new_height - 224) // 2
        left = (new_width - 224) // 2
        cropped = resized[:, :, top : top + 224, left : left + 224].squeeze(0)
        mean = torch.tensor(self.mean, device=cropped.device, dtype=cropped.dtype).view(-1, 1, 1)
        std = torch.tensor(self.std, device=cropped.device, dtype=cropped.dtype).view(-1, 1, 1)
        return (cropped - mean) / std

    def preprocess_image(self, image):
        if isinstance(image, torch.Tensor):
            return self.preprocess_tensor_diff(image)
        return self.preprocess_pil(image).to(self.device)

    def predict_from_pil(self, image):
        image_tensor = self.preprocess_image(image)
        embedding = F.normalize(self.clip_model.encode_image(image_tensor[None, ...]).float(), dim=-1)
        return self.model(embedding)

    def predict_from_tensor(self, image_tensor, data_type=torch.float32):
        image_tensor = self.preprocess_image(image_tensor)
        embedding = F.normalize(self.clip_model.encode_image(image_tensor[None, ...]).float(), dim=-1)
        return self.model(embedding).to(data_type)
