"""Shared preprocessing and frozen visual models for caching and Policy.act()."""

import re

import torch
import torch.nn.functional as F


IMAGE_SIZE = (240, 320)
MODEL_IDS = {
    "dino": "facebook/dinov2-small",
    "vfm_depth": "depth-anything/Depth-Anything-V2-Small-hf",
    "vfm_seg": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
}
INPUT_SIZES = {
    "dino": (224, 294),
    "vfm_depth": (280, 378),
    "vfm_seg": (480, 640),
}

# Keys are shared class IDs. Values are source label IDs.
CITYSCAPES = {
    1: (0,), 2: (1,), 3: (2, 3, 4, 5), 4: (8, 9),
    6: (13, 14, 15, 16, 17, 18), 7: (11, 12), 8: (6,), 9: (7,),
}
CARLA_LEGACY = {
    1: (6, 7), 2: (8,), 3: (1, 2, 5, 11, 15, 17), 4: (9, 22),
    6: (10,), 7: (4,), 8: (18,), 9: (12,),
}
CARLA_MODERN = {
    1: (1, 24), 2: (2,), 3: (3, 4, 5, 6, 26, 28), 4: (9, 10),
    5: (11,), 6: (14, 15, 16, 17, 18, 19), 7: (12, 13),
    8: (7,), 9: (8,),
}


def remap_labels(labels, groups):
    labels = torch.as_tensor(labels)
    lut = torch.zeros(256, dtype=torch.uint8, device=labels.device)
    for shared_id, source_ids in groups.items():
        lut[list(source_ids)] = shared_id
    return lut[labels.long()]


def remap_carla_seg(labels, carla_version):
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(carla_version))
    if match is None:
        raise ValueError(f"Invalid CARLA version: {carla_version!r}")
    version = tuple(int(value) for value in match.groups())
    groups = CARLA_LEGACY if version < (0, 9, 14) else CARLA_MODERN
    return remap_labels(labels, groups)


def normalize_depth(depth):
    """Normalize each H x W depth map independently to [0, 1]."""
    depth = torch.as_tensor(depth).float()
    lo = depth.amin(dim=(-2, -1), keepdim=True)
    hi = depth.amax(dim=(-2, -1), keepdim=True)
    span = hi - lo
    denominator = torch.where(span > 0, span, torch.ones_like(span))
    return (depth - lo) / denominator


def normalize_gt_depth(depth_m):
    depth_m = torch.as_tensor(depth_m).float()
    return normalize_depth(depth_m.clamp(1.0, 100.0).reciprocal())


def preprocess_rgb(rgb, size, device, dtype):
    """Accept uint8 RGB in HWC or BHWC order and return normalized BCHW."""
    x = torch.as_tensor(rgb, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.dtype != torch.uint8 or x.ndim != 4 or x.shape[1:] != (240, 320, 3):
        raise ValueError(f"Expected uint8 RGB (B, 240, 320, 3); got {x.shape}, {x.dtype}")
    x = x.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(dtype=dtype)


class FrozenVFM:
    """Load only the requested models. Use FP16 on CUDA and FP32 on Mac."""

    def __init__(self, device="cpu", kinds=tuple(MODEL_IDS)):
        from transformers import (
            AutoModel,
            AutoModelForDepthEstimation,
            AutoModelForSemanticSegmentation,
        )

        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        loaders = {
            "dino": AutoModel,
            "vfm_depth": AutoModelForDepthEstimation,
            "vfm_seg": AutoModelForSemanticSegmentation,
        }
        self.models = {}
        for kind in kinds:
            model = loaders[kind].from_pretrained(MODEL_IDS[kind])
            model = model.eval().requires_grad_(False)
            self.models[kind] = model.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def extract(self, kind, rgb):
        x = preprocess_rgb(rgb, INPUT_SIZES[kind], self.device, self.dtype)
        result = self.models[kind](pixel_values=x)

        if kind == "dino":
            patches = result.last_hidden_state[:, 1:, :]
            if patches.shape[1:] != (336, 384):
                raise ValueError(f"Unexpected DINOv2 patch shape: {patches.shape}")
            return patches.transpose(1, 2).reshape(x.shape[0], 384, 16, 21)

        if kind == "vfm_depth":
            depth = result.predicted_depth.float().unsqueeze(1)
            depth = F.interpolate(
                depth, size=IMAGE_SIZE, mode="bicubic", align_corners=False
            ).squeeze(1)
            return normalize_depth(depth)

        labels = result.logits.argmax(dim=1, keepdim=True).float()
        labels = F.interpolate(labels, size=IMAGE_SIZE, mode="nearest").squeeze(1)
        return remap_labels(labels, CITYSCAPES)
