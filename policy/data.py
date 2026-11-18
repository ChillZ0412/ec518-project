"""Episode-level splits, collision truncation and frame-aligned feature loading."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from policy.model import VARIANTS
from policy.vfm import normalize_gt_depth, remap_carla_seg


def read_image(path, rgb=False):
    with Image.open(path) as image:
        if rgb:
            image = image.convert("RGB")
        return np.array(image, copy=True)


def gt_inputs(seg, depth, version):
    return {
        "seg": remap_carla_seg(seg, version),
        "depth": normalize_gt_depth(depth),
    }


class EpisodeDataset(Dataset):
    def __init__(self, data_root, feature_root, variant, split, val_seed=7):
        if variant not in VARIANTS or split not in ("train", "val"):
            raise ValueError("Invalid variant or split")
        self.variant = variant
        self.episodes = []
        self.samples = []
        self.summary = []
        self.jitter = ColorJitter(0.2, 0.2, 0.2, 0.05) if split == "train" else None
        files = sorted(Path(data_root).rglob("measurements.csv"))
        for csv in files:
            meta = json.loads((csv.parent / "meta.json").read_text())
            seed = int(meta["seed"])
            if (seed == val_seed) != (split == "val"):
                continue
            table = pd.read_csv(csv)
            required = ["frame", "speed", "command", "steer", "throttle", "brake", "collision"]
            if table.empty or not np.isfinite(table[required].to_numpy()).all():
                raise ValueError(f"Invalid measurements: {csv}")
            frame_ids = table.frame.to_numpy()
            if np.any(frame_ids != frame_ids.astype(np.int64)) or len(set(frame_ids)) != len(table):
                raise ValueError(f"Invalid frame IDs: {csv}")
            if not table.command.isin(range(1, 7)).all():
                raise ValueError(f"Invalid commands: {csv}")
            collisions = np.flatnonzero(table.collision.to_numpy() != 0)
            keep = int(collisions[0]) if len(collisions) else len(table)
            entry = {"path": csv.parent, "version": meta["carla_version"], "table": table, "cache": {}}
            kinds = []
            if variant in ("vfm_implicit", "hybrid"):
                kinds.append("dino")
            if variant in ("vfm_explicit", "hybrid"):
                kinds.extend(("vfm_seg", "vfm_depth"))
            positions = np.arange(len(table))
            if kinds:
                cache_dir = Path(feature_root) / csv.parent.name
                cached_ids = np.load(cache_dir / "frame_ids.npy", allow_pickle=False)
                if len(set(cached_ids.tolist())) != len(cached_ids):
                    raise ValueError(f"Duplicate cached frame IDs: {cache_dir}")
                lookup = {int(frame): i for i, frame in enumerate(cached_ids)}
                positions = [lookup[int(frame)] for frame in frame_ids[:keep]]
                for kind in kinds:
                    array = np.load(cache_dir / f"{kind}.npy", mmap_mode="r", allow_pickle=False)
                    if len(array) != len(cached_ids):
                        raise ValueError(f"Cache length mismatch: {cache_dir / kind}")
                    entry["cache"][kind] = array
            episode_index = len(self.episodes)
            self.episodes.append(entry)
            self.samples.extend((episode_index, row, int(positions[row])) for row in range(keep))
            self.summary.append({"episode": csv.parent.name, "seed": seed,
                                 "total_frames": len(table), "used_frames": keep})
        if not self.samples:
            raise ValueError(f"No usable {split} samples under {data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_id, row_index, cache_index = self.samples[index]
        episode = self.episodes[episode_id]
        row = episode["table"].iloc[row_index]
        stem = f"{int(row.frame):06d}"
        inputs = {}
        if self.variant == "rgb":
            rgb = read_image(episode["path"] / "rgb" / f"{stem}.png", rgb=True)
            if self.jitter is not None:
                rgb = np.array(self.jitter(Image.fromarray(rgb)), copy=True)
            inputs["rgb"] = torch.from_numpy(rgb)
        if self.variant == "gt_explicit":
            seg = read_image(episode["path"] / "seg" / f"{stem}.png")
            depth = np.load(episode["path"] / "depth" / f"{stem}.npy", allow_pickle=False)
            inputs.update(gt_inputs(seg, depth, episode["version"]))
        for kind, array in episode["cache"].items():
            key = {"dino": "dino", "vfm_seg": "seg", "vfm_depth": "depth"}[kind]
            inputs[key] = torch.from_numpy(np.array(array[cache_index], copy=True))
        target = torch.tensor([row.steer, row.throttle, row.brake], dtype=torch.float32)
        return inputs, torch.tensor(row.speed, dtype=torch.float32), torch.tensor(int(row.command)), target
