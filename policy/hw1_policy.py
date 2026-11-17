"""Wrap the HW1 behavior-cloning model as a Policy, for the RGB baseline before Wednesday.

Fill in the three TODOs by copying from your HW1 test_agent.py (how it builds the model,
preprocesses the image and turns the output into steer/throttle/brake).
Then run the leaderboard with --agent-config=hw1:D:\\path\\to\\your_hw1_model.pth
(load_policy in policy_interface.py already routes the "hw1:" prefix here).
"""
import numpy as np
import torch

from policy.policy_interface import Policy


class HW1Policy(Policy):

    @classmethod
    def load(cls, ckpt_path, device="cuda"):
        self = cls()
        self.device = device
        # TODO 1: build the network exactly as in HW1, e.g.
        #   from your_hw1_module import ClassificationNetwork
        #   self.model = ClassificationNetwork()
        #   self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model = torch.load(ckpt_path, map_location=device)
        self.model.eval()
        return self

    @torch.no_grad()
    def act(self, obs):
        rgb = obs["rgb"]  # uint8 (240, 320, 3), RGB order
        # TODO 2: same preprocessing as HW1 (channel order, normalization, cropping...)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        out = self.model(x)
        # TODO 3: same output decoding as HW1 (e.g. class index -> action)
        steer, throttle, brake = [float(v) for v in out.squeeze(0)[:3].cpu().numpy()]
        return steer, throttle, brake
