"""Plot validation losses and command-specific steer errors across training runs."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for run in args.runs:
        table = pd.read_csv(run / "metrics.csv")
        ax.plot(table.epoch, table.val_loss, label=run.name)
    ax.set(xlabel="Epoch", ylabel="Validation weighted L1 (2 steer + throttle + brake)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "validation_loss.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots()
    commands = (1, 2, 4)
    width = 0.8 / len(args.runs)
    for i, run in enumerate(args.runs):
        rows = {row["command"]: row for row in json.loads((run / "best_commands.json").read_text())}
        errors = [rows[command]["steer_mae"] if rows[command]["count"] else np.nan for command in commands]
        ax.bar(np.arange(3) - 0.4 + width * (i + 0.5), errors, width, label=run.name)
    ax.set_xticks(range(3), ("Left (1)", "Right (2)", "Lane follow (4)"))
    ax.set_ylabel("Validation steer MAE, best checkpoint")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "steer_by_command.png", dpi=200)
    plt.close(fig)
    print(f"Saved figures to {args.output}")


if __name__ == "__main__":
    main()
