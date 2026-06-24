"""Simulacra Aesthetic Captions predictor."""

from pathlib import Path

import clip
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, features_in):
        super().__init__()
        self.linear = nn.Linear(features_in, 1)

    def forward(self, inputs):
        normalized = F.normalize(inputs, dim=-1) * inputs.shape[-1] ** 0.5
        return self.linear(normalized)


class SimulacraAesthetic:
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_model_name = "ViT-B/16"
        self.clip_model = clip.load(self.clip_model_name, jit=False, device=self.device)[0]
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

        weights_path = Path(__file__).with_name("models") / "sac_public_2022_06_29_vit_b_16_linear.pth"
        self.model = AestheticMeanPredictionLinearModel(512)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval().requires_grad_(False).to(self.device)

    def preprocess_tensor_diff(self, image_tensor):
        _, height, width = image_tensor.shape
        scale = 224 / min(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))
        resized = F.interpolate(
            image_tensor.unsqueeze(0),
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
        return self.preprocess_pil(image).to(self.device)

    def predict_from_pil(self, image):
        image_tensor = self.preprocess_image(image)
        embedding = F.normalize(self.clip_model.encode_image(image_tensor[None, ...]).float(), dim=-1)
        return self.model(embedding)

    def predict_from_tensor(self, image_tensor, data_type=torch.float32):
        image_tensor = self.preprocess_tensor_diff(image_tensor)
        embedding = F.normalize(self.clip_model.encode_image(image_tensor[None, ...]).float(), dim=-1)
        return self.model(embedding).to(data_type)
