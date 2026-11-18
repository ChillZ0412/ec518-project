"""Compare cached validation inference with Policy.act on raw observations."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from policy.data import EpisodeDataset, read_image
from policy.policy_interface import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    policy = load_policy(args.checkpoint, args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = EpisodeDataset(args.data_root, args.feature_root, checkpoint["variant"], "val",
                             checkpoint["config"].get("val_seed", 7))
    indices = np.linspace(0, len(dataset) - 1, min(args.frames, len(dataset)), dtype=int)
    maximum = 0.0
    timings = []
    policy.reset()
    with torch.inference_mode():
        for index in indices:
            inputs, speed, command, _ = dataset[int(index)]
            episode_id, row_index, _ = dataset.samples[int(index)]
            episode = dataset.episodes[episode_id]
            frame = int(episode["table"].iloc[row_index].frame)
            obs = {"rgb": read_image(episode["path"] / "rgb" / f"{frame:06d}.png", rgb=True),
                   "speed": float(speed), "command": int(command)}
            if policy.needs_gt:
                obs["seg"] = read_image(episode["path"] / "seg" / f"{frame:06d}.png")
                obs["depth"] = np.load(episode["path"] / "depth" / f"{frame:06d}.npy")
            prepared = {key: value.unsqueeze(0).to(policy.device) for key, value in inputs.items()}
            cached = policy.model(prepared, speed.reshape(1).to(policy.device),
                                  command.reshape(1).to(policy.device))[0].cpu().numpy()
            if policy.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            online = np.asarray(policy.act(obs))
            if policy.device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            maximum = max(maximum, float(np.abs(cached - online).max()))
    print(f"Frames: {len(indices)}; maximum control difference: {maximum:.6f}")
    print(f"Policy.act mean: {np.mean(timings) * 1000:.2f} ms; "
          f"post-first mean: {np.mean(timings[1:] or timings) * 1000:.2f} ms")
    assert maximum <= 0.01, "Cached/online difference exceeds 0.01"
    print("POLICY CONSISTENCY CHECK PASSED")


if __name__ == "__main__":
    main()
