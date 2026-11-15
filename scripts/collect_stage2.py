"""Collect the stage 2 checks (overfit and timing runs) into reports/stage2_checks.json.

    python scripts/collect_stage2.py

runs/ is not in git, so the numbers the stage 2 page shows are copied here by script,
never typed by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
OUT = REPO / "reports" / "stage2_checks.json"


def rows(run: str) -> list[dict]:
    with (RUNS / run / "metrics.csv").open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def timing(run: str) -> dict:
    r = rows(run)
    steady = r[1:] or r  # the first epoch includes start-up costs
    return {
        "epochs": len(r),
        "train_seconds_per_epoch": mean(x["train_seconds"] for x in steady),
        "val_seconds_per_epoch": mean(x["val_seconds"] for x in steady),
        "first_epoch_train_loss": r[0]["train_loss"],
        "peak_mb": max(x["peak_mb"] for x in r),
    }


def main() -> None:
    overfit = rows("overfit32")
    summary = json.loads((RUNS / "overfit32" / "summary.json").read_text())
    before, after = timing("timing_r34"), timing("timing_r34_v2")
    data = {
        "overfit": {
            "images": 32,
            "best_epoch": summary["best_epoch"],
            "best_val": summary["val"],
            "curve": [
                {"epoch": int(x["epoch"]), "dice": x["val_dice"], "boundary_iou": x["val_boundary_iou"]}
                for x in overfit
            ],
            "parameters": summary["parameters"]["total"],
        },
        "timing": {
            "model": "smp_unet_r34, 256 px, batch 16, 1292 training and 158 validation images",
            "before": before,
            "after": after,
            "speedup": (before["train_seconds_per_epoch"] + before["val_seconds_per_epoch"])
            / (after["train_seconds_per_epoch"] + after["val_seconds_per_epoch"]),
            "same_first_epoch": abs(before["first_epoch_train_loss"] - after["first_epoch_train_loss"])
            < 1e-6,
            "minutes_per_50_epochs": 50
            * (after["train_seconds_per_epoch"] + after["val_seconds_per_epoch"])
            / 60,
        },
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["timing"], indent=2))
    print("overfit best:", data["overfit"]["best_val"], "epoch", data["overfit"]["best_epoch"])


if __name__ == "__main__":
    main()
