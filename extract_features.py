"""Cache visual features for one or more EC518 episodes."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from policy.vfm import FrozenVFM, MODEL_IDS, normalize_gt_depth, remap_carla_seg


OUTPUTS = {
    "dino": ((384, 16, 21), np.float16),
    "vfm_depth": ((240, 320), np.float16),
    "vfm_seg": ((240, 320), np.uint8),
}
PALETTE = np.array([
    [45, 45, 45], [128, 64, 128], [244, 35, 232], [100, 100, 100],
    [107, 142, 35], [70, 130, 180], [0, 0, 142], [220, 20, 60],
    [250, 170, 30], [220, 220, 0],
], dtype=np.uint8)


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), copy=True)


def save_comparison(episode, frame, out_dir, carla_version):
    stem = f"{frame:06d}"
    rgb = read_rgb(episode / "rgb" / f"{stem}.png")
    with Image.open(episode / "seg" / f"{stem}.png") as image:
        gt_seg = remap_carla_seg(np.array(image, copy=True), carla_version).numpy()
    gt_depth = normalize_gt_depth(
        np.load(episode / "depth" / f"{stem}.npy", allow_pickle=False)
    ).numpy()
    pred_seg = np.load(out_dir / "vfm_seg.npy", mmap_mode="r")[0]
    pred_depth = np.load(out_dir / "vfm_depth.npy", mmap_mode="r")[0]

    def depth_image(depth):
        gray = (np.clip(depth.astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")

    panels = [
        ("RGB", Image.fromarray(rgb)),
        ("GT Segmentation", Image.fromarray(PALETTE[gt_seg])),
        ("Predicted Segmentation", Image.fromarray(PALETTE[pred_seg])),
        ("GT Inverse Depth", depth_image(gt_depth)),
        ("Predicted Relative Depth", depth_image(pred_depth)),
    ]
    comparison = Image.new("RGB", (320 * len(panels), 268), "white")
    draw = ImageDraw.Draw(comparison)
    for i, (title, image) in enumerate(panels):
        comparison.paste(image, (320 * i, 28))
        draw.text((320 * i + 8, 7), title, fill="black")
    comparison.save(out_dir / "vfm_comparison.png")


def cache_episode(episode, out_root, extractor, kinds, batch_size, limit):
    with (episode / "measurements.csv").open(newline="") as handle:
        frames = [int(row["frame"]) for row in csv.DictReader(handle)]
    if not frames or len(set(frames)) != len(frames):
        raise ValueError(f"{episode}: expected unique frame IDs and at least one row")
    frames = frames[:limit]
    meta = json.loads((episode / "meta.json").read_text())
    target = out_root / episode.name
    if target.exists():
        raise FileExistsError(f"{target} already exists. Use a different --output-root.")

    started = perf_counter()
    with TemporaryDirectory(prefix=".extract-", dir=out_root) as temporary:
        stage = Path(temporary) / episode.name
        stage.mkdir()
        np.save(stage / "frame_ids.npy", np.asarray(frames, dtype=np.int64))

        for kind in kinds:
            shape, dtype = OUTPUTS[kind]
            path = stage / f"{kind}.npy"
            cache = np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=(len(frames), *shape)
            )
            model_started = perf_counter()
            for start in range(0, len(frames), batch_size):
                batch_ids = frames[start:start + batch_size]
                rgb = np.stack([
                    read_rgb(episode / "rgb" / f"{frame:06d}.png")
                    for frame in batch_ids
                ])
                features = extractor.extract(kind, rgb).cpu().numpy()
                if not np.isfinite(features).all():
                    raise ValueError(f"{kind}: non-finite output near frame {batch_ids[0]}")
                cache[start:start + len(batch_ids)] = features

            cache.flush()
            print(f"{kind}: {cache.shape} {cache.dtype} "
                  f"{(perf_counter() - model_started) * 1000 / len(frames):.1f} ms/frame",
                  flush=True)
            del cache

        if "vfm_seg" in kinds and "vfm_depth" in kinds:
            save_comparison(episode, frames[0], stage, meta["carla_version"])

        stage.rename(target)

    elapsed = perf_counter() - started
    print(f"OK: {len(frames)} frames in {elapsed:.1f}s -> {target}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--output-root", type=Path, default=Path("features"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_IDS),
                        default=list(MODEL_IDS))
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    extractor = FrozenVFM(args.device, args.models)
    for episode in args.episodes:
        cache_episode(
            episode, args.output_root, extractor, args.models, args.batch_size, args.limit
        )


if __name__ == "__main__":
    main()
