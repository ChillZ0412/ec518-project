"""Online policy using the same preprocessing as offline feature extraction."""

import torch

from policy.data import gt_inputs
from policy.model import PolicyNetwork
from policy.policy_interface import Policy
from policy.vfm import FrozenVFM


class LearnedPolicy(Policy):
    @classmethod
    def load(cls, ckpt_path, device="cuda"):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        policy = cls()
        policy.device = torch.device(device)
        policy.variant = checkpoint["variant"]
        policy.model = PolicyNetwork(policy.variant, pretrained=False).to(policy.device)
        policy.model.load_state_dict(checkpoint["state_dict"])
        policy.model.eval()
        policy.needs_gt = policy.variant == "gt_explicit"
        policy.carla_version = checkpoint["config"].get("carla_version", "0.9.10.1")
        kinds = []
        if policy.variant in ("vfm_implicit", "hybrid"):
            kinds.append("dino")
        if policy.variant in ("vfm_explicit", "hybrid"):
            kinds.extend(("vfm_seg", "vfm_depth"))
        policy.extractor = FrozenVFM(device, kinds) if kinds else None
        return policy

    @torch.inference_mode()
    def act(self, obs):
        inputs = {}
        if self.variant == "rgb":
            inputs["rgb"] = torch.as_tensor(obs["rgb"], device=self.device).unsqueeze(0)
        if self.needs_gt:
            inputs = {key: value.to(self.device).unsqueeze(0) for key, value in
                      gt_inputs(obs["seg"], obs["depth"], self.carla_version).items()}
        if self.extractor is not None:
            for kind in self.extractor.models:
                key = {"dino": "dino", "vfm_seg": "seg", "vfm_depth": "depth"}[kind]
                # Match the cache storage precision before the policy encoder.
                feature = self.extractor.extract(kind, obs["rgb"])
                inputs[key] = feature if kind == "vfm_seg" else feature.half()
        speed = torch.tensor([obs["speed"]], device=self.device, dtype=torch.float32)
        command = torch.tensor([obs["command"]], device=self.device)
        result = self.model(inputs, speed, command)[0]
        if not torch.isfinite(result).all():
            raise FloatingPointError("Policy returned non-finite controls")
        return tuple(result.cpu().tolist())
