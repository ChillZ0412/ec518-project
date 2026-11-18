"""Exercise each variant using real dataset inputs without downloading weights."""

import argparse
import torch
from torch.utils.data import DataLoader

from policy.data import EpisodeDataset
from policy.model import PolicyNetwork, VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(8)
    for variant in VARIANTS:
        data = EpisodeDataset(args.data_root, args.feature_root, variant, "train")
        inputs, speed, command, target = next(iter(DataLoader(data, batch_size=2)))
        model = PolicyNetwork(variant, pretrained=False).to(args.device)
        output = model({key: value.to(args.device) for key, value in inputs.items()},
                       speed.to(args.device), command.to(args.device))
        assert output.shape == (len(target), 3)
        assert torch.isfinite(output).all()
        (output - target.to(args.device)).abs().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print(f"OK: {variant}, train samples={len(data)}, forward/backward", flush=True)
        del model, output, data
    print("ALL FIVE VARIANTS PASSED")


if __name__ == "__main__":
    main()
