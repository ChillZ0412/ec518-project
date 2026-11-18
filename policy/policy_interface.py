"""The contract between training (Yuhao) and evaluation (Z). See docs/interface.md.

The evaluation agent only ever calls:
    policy = load_policy(path, device)
    policy.reset()                      # at the start of each route
    steer, throttle, brake = policy.act(obs)

obs keys:
    "rgb":     uint8   (240, 320, 3), RGB order
    "speed":   float, m/s
    "command": int 1-6 (CARLA RoadOption)
    "seg":     uint8   (240, 320), raw CARLA tag          -- GT-Explicit only
    "depth":   float32 (240, 320), meters                 -- GT-Explicit only
returns steer in [-1, 1], throttle in [0, 1], brake in [0, 1].

Everything model-specific (frozen VFMs, preprocessing, class mapping) lives inside the
Policy, so training-time feature extraction and evaluation-time inference share one code path.
"""


class Policy:
    #: which sensors the evaluation agent must provide for this policy
    needs_gt = False

    @classmethod
    def load(cls, ckpt_path, device="cuda"):
        raise NotImplementedError

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError


class DummyPolicy(Policy):
    """Drives straight at low speed. Used to test the evaluation pipeline end to end."""

    @classmethod
    def load(cls, ckpt_path=None, device="cpu"):
        return cls()

    def act(self, obs):
        throttle = 0.4 if obs["speed"] < 4.0 else 0.0
        return 0.0, throttle, 0.0


def load_policy(path, device="cuda"):
    """Load dummy, HW1 or a variant-aware learned checkpoint."""
    path = str(path)
    if path == "dummy":
        return DummyPolicy.load()
    if path.startswith("hw1:"):  # RGB baseline from HW1, e.g. "hw1:D:\\518\\hw1\\model.pth"
        from policy.hw1_policy import HW1Policy
        return HW1Policy.load(path[4:], device)
    from policy.learned import LearnedPolicy
    return LearnedPolicy.load(path, device)
