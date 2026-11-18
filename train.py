"""Train a frame-based CIL policy using episode-level validation."""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from policy.data import EpisodeDataset
from policy.model import PolicyNetwork, VARIANTS, canonical_commands


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = torch.zeros(3, device=device)
    command_errors = torch.zeros(6, device=device)
    command_counts = torch.zeros(6, device=device)
    count = 0
    with torch.set_grad_enabled(training):
        for inputs, speed, command, target in loader:
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            speed, command, target = [value.to(device, non_blocking=True) for value in (speed, command, target)]
            prediction = model(inputs, speed, command)
            errors = (prediction - target).abs()
            loss = (errors * errors.new_tensor([2.0, 1.0, 1.0])).sum(dim=1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += errors.detach().sum(dim=0)
            count += len(target)
            branch_ids = canonical_commands(command) - 1
            command_errors.scatter_add_(0, branch_ids, errors[:, 0].detach())
            command_counts.scatter_add_(0, branch_ids, torch.ones_like(speed))
    means = (total / count).cpu().tolist()
    metrics = dict(zip(("steer", "throttle", "brake"), means))
    metrics["loss"] = 2 * means[0] + means[1] + means[2]
    per_command = []
    for command_id, (error, number) in enumerate(zip(command_errors.cpu().tolist(), command_counts.cpu().tolist()), 1):
        per_command.append({"command": command_id, "count": int(number),
                            "steer_mae": error / number if number else None})
    return metrics, per_command


def save_plots(history, per_command, directory):
    fig, ax = plt.subplots()
    for split in ("train", "val"):
        ax.plot([row["epoch"] for row in history], [row[f"{split}_loss"] for row in history], label=split)
    ax.set(xlabel="Epoch", ylabel="Weighted L1 (2 steer + throttle + brake)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "loss.png", dpi=180)
    plt.close(fig)
    present = [row for row in per_command if row["count"]]
    fig, ax = plt.subplots()
    ax.bar([str(row["command"]) for row in present], [row["steer_mae"] for row in present])
    ax.set(xlabel="Command (5/6 merged into 4)", ylabel="Validation steer MAE, best checkpoint")
    fig.tight_layout()
    fig.savefig(directory / "steer_by_command.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true", help="For offline smoke tests only")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    directory = args.output_root / f"{args.variant}_seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=False)
    train = EpisodeDataset(args.data_root, args.feature_root, args.variant, "train", args.val_seed)
    val = EpisodeDataset(args.data_root, args.feature_root, args.variant, "val", args.val_seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    versions = {entry["version"] for entry in train.episodes + val.episodes}
    if args.variant == "gt_explicit" and len(versions) != 1:
        raise ValueError("GT policy checkpoint requires one CARLA version")
    config["carla_version"] = sorted(versions)[0]
    config.update({"command_5_6_to_4": True, "speed_scale": 10.0,
                   "loss_weights": [2.0, 1.0, 1.0], "collision_cutoff": "exclude first collision and all later rows"})
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    (directory / "split.json").write_text(json.dumps({"train": train.summary, "val": val.summary}, indent=2))
    print(f"GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}", flush=True)
    print(f"Samples: train={len(train)}, val={len(val)}", flush=True)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, shuffle=True, **options)
    val_loader = DataLoader(val, shuffle=False, **options)
    model = PolicyNetwork(args.variant, pretrained=not args.no_pretrained).to(device)
    encoder_parameters = list(model.encoders.parameters()) + list(model.projection.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 1e-4},
        {"params": list(model.head.parameters()) + list(model.branches.parameters()), "lr": 3e-4},
    ], weight_decay=0.01)
    history = []
    best = float("inf")
    best_epoch = 0
    best_commands = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics, _ = run_epoch(model, train_loader, device, optimizer)
        val_metrics, commands = run_epoch(model, val_loader, device)
        seconds = time.perf_counter() - started
        row = {"epoch": epoch, "seconds": seconds}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        checkpoint = {"variant": args.variant, "state_dict": model.state_dict(), "config": config,
                      "epoch": epoch, "val_metrics": val_metrics}
        if val_metrics["loss"] < best:
            best, best_epoch, best_commands = val_metrics["loss"], epoch, commands
            atomic_save(checkpoint, directory / f"{args.variant}_seed{args.seed}.pt")
            (directory / "best_commands.json").write_text(json.dumps(commands, indent=2))
        atomic_save({**checkpoint, "optimizer": optimizer.state_dict()}, directory / "last.pt")
        with (directory / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:02d}/{args.epochs} train={train_metrics['loss']:.5f} "
              f"val={val_metrics['loss']:.5f} time={seconds:.1f}s best={best:.5f}@{best_epoch}", flush=True)
        save_plots(history, best_commands, directory)
    summary = {"best_val_loss": best, "best_epoch": best_epoch,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
               "train_samples": len(train), "val_samples": len(val)}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE: {directory}", flush=True)


if __name__ == "__main__":
    main()
