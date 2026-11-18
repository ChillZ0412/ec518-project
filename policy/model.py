"""Five visual encoders with a shared conditional imitation-learning head."""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from policy.vfm import IMAGE_SIZE, preprocess_rgb

VARIANTS = ("rgb", "gt_explicit", "vfm_explicit", "vfm_implicit", "hybrid")


def canonical_commands(commands):
    commands = commands.long()
    if torch.any((commands < 1) | (commands > 6)):
        raise ValueError("Commands must be integers in 1..6")
    return torch.where(commands >= 5, 4, commands)


def resnet_encoder(channels, pretrained):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    encoder = resnet18(weights=weights)
    if channels != 3:
        original = encoder.conv1
        encoder.conv1 = nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            encoder.conv1.weight.copy_(
                original.weight.mean(dim=1, keepdim=True).repeat(1, channels, 1, 1)
                * (3.0 / channels)
            )
    encoder.fc = nn.Identity()
    return encoder


def implicit_encoder():
    return nn.Sequential(
        nn.Conv2d(384, 128, 1), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 512), nn.ReLU(),
    )


class PolicyNetwork(nn.Module):
    def __init__(self, variant, pretrained=True):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.encoders = nn.ModuleDict()
        if variant == "rgb":
            self.encoders["rgb"] = resnet_encoder(3, pretrained)
        if variant in ("gt_explicit", "vfm_explicit", "hybrid"):
            self.encoders["explicit"] = resnet_encoder(11, pretrained)
        if variant in ("vfm_implicit", "hybrid"):
            self.encoders["implicit"] = implicit_encoder()
        self.projection = nn.Linear(1024, 512) if variant == "hybrid" else nn.Identity()
        self.head = nn.Sequential(nn.Linear(513, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.branches = nn.ModuleList([nn.Linear(128, 3) for _ in range(6)])

    def forward(self, inputs, speed, command):
        features = []
        if self.variant == "rgb":
            rgb = preprocess_rgb(inputs["rgb"], IMAGE_SIZE, speed.device, torch.float32)
            features.append(self.encoders["rgb"](rgb))
        if "explicit" in self.encoders:
            seg = F.one_hot(inputs["seg"].long(), num_classes=10).permute(0, 3, 1, 2).float()
            explicit = torch.cat((seg, inputs["depth"].float().unsqueeze(1)), dim=1)
            features.append(self.encoders["explicit"](explicit))
        if "implicit" in self.encoders:
            features.append(self.encoders["implicit"](inputs["dino"].float()))
        embedding = self.projection(torch.cat(features, dim=1))
        hidden = self.head(torch.cat((embedding, speed.float().reshape(-1, 1) / 10.0), dim=1))
        outputs = torch.stack([branch(hidden) for branch in self.branches], dim=1)
        branch_ids = canonical_commands(command).reshape(-1) - 1
        raw = outputs[torch.arange(len(hidden), device=hidden.device), branch_ids]
        return torch.cat((raw[:, :1].tanh(), raw[:, 1:].sigmoid()), dim=1)
