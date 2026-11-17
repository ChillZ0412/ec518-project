"""Check one episode against docs/interface.md and save a preview image.

Both teammates use this: Z after collecting, Yuhao after downloading.
    python check_episode.py dataset/town01_clearnoon_v50_seed0
Only needs numpy, pandas, pillow (no CARLA), so it also runs on a Mac.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

EXPECTED_COLS = ["frame", "speed", "command", "steer", "throttle", "brake", "x", "y", "yaw", "collision"]


def main(ep_dir):
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    df = pd.read_csv(os.path.join(ep_dir, "measurements.csv"))
    n = len(df)
    print("meta:", json.dumps(meta, indent=1))
    assert list(df.columns) == EXPECTED_COLS, f"columns differ: {list(df.columns)}"
    for d, ext in [("rgb", ".png"), ("seg", ".png"), ("depth", ".npy")]:
        k = len([f for f in os.listdir(os.path.join(ep_dir, d)) if f.endswith(ext)])
        assert k == n, f"{d}: {k} files but {n} rows"

    rgb = np.array(Image.open(os.path.join(ep_dir, "rgb", "000000.png")))
    seg = np.array(Image.open(os.path.join(ep_dir, "seg", "000000.png")))
    depth = np.load(os.path.join(ep_dir, "depth", "000000.npy"))
    print(f"rgb   {rgb.shape} {rgb.dtype}")
    print(f"seg   {seg.shape} {seg.dtype} labels={sorted(np.unique(seg).tolist())}")
    print(f"depth {depth.shape} {depth.dtype} min={depth.min():.1f}m max={depth.max():.1f}m")
    assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
    assert seg.shape == (240, 320) and seg.dtype == np.uint8
    assert depth.shape == (240, 320) and depth.dtype == np.float16

    print(f"frames={n}  speed mean={df.speed.mean():.2f} m/s  max={df.speed.max():.2f}")
    print("command counts:", df.command.value_counts().sort_index().to_dict())
    print(f"steer range [{df.steer.min():.2f}, {df.steer.max():.2f}]  "
          f"brake>0: {(df.brake > 0).mean():.0%} of frames  collision frames: {int(df.collision.sum())}")

    # preview: rgb | seg (random colors) | depth (clipped to 50 m)
    palette = np.random.RandomState(0).randint(0, 255, (256, 3)).astype(np.uint8)
    seg_vis = palette[seg]
    d = np.clip(depth.astype(np.float32), 0, 50) / 50.0
    depth_vis = (np.stack([1 - d] * 3, -1) * 255).astype(np.uint8)
    out = os.path.join(ep_dir, "preview.png")
    Image.fromarray(np.concatenate([rgb, seg_vis, depth_vis], axis=1)).save(out)
    print("OK, preview saved to", out)


if __name__ == "__main__":
    main(sys.argv[1])
