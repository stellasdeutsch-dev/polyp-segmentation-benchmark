"""Score one trained run on the five test sets (stage 4.3; decisions D3, D4, D5).

    python scripts/eval_run.py --run unet_lr3e-4_s0            # Apple GPU
    python scripts/eval_run.py --run unet_lr3e-4_s0 --device cpu

Loads runs/<run>/best.pt, the epoch chosen on validation Dice (D3); the test sets are opened here
for the first time. Each image is resized to the training size and normalised as in validation,
the logits are upsampled bilinearly to the mask's size, and the probability map is scored with
seg.evaluate.score_probability_map: Dice, IoU and Boundary IoU at 0.5 on the raw probability,
PraNet-style mDice on the 8-bit map MyTest.py would have saved. Writes runs/<run>/test.json
(git-ignored with the run; the stage reports collect what they need from it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from seg import data
from seg.env import get_device
from seg.evaluate import score_probability_map, summarise
from seg.models import build_model
from seg.train import TrainConfig

REPO = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def predict(model, sample: dict, device: torch.device) -> np.ndarray:
    logits = model(sample["image"][None].to(device))
    full = sample["mask_full"].shape
    logits = F.interpolate(logits, size=full, mode="bilinear", align_corners=False)
    return torch.sigmoid(logits)[0, 0].float().cpu().numpy()


def evaluate_run(run_dir: Path, device: torch.device, limit: int | None = None) -> dict:
    cfg = TrainConfig.from_yaml(run_dir / "config.yaml")
    model = build_model(cfg.model, pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    out = {"run": run_dir.name, "model": cfg.model, "size": cfg.size, "device": device.type, "sets": []}
    for test_set in data.TEST_SETS:
        items = data.items_for_testset(test_set)[:limit]
        ds = data.PolypDataset(items, data.eval_transform(cfg.size))
        rows, dice_c, iou_c = [], [], []
        start = time.perf_counter()
        for k in range(len(ds)):
            sample = ds[k]
            row, d, i = score_probability_map(
                predict(model, sample, device), sample["mask_full"].numpy().astype(bool), sample["stem"]
            )
            rows.append(row)
            dice_c.append(d)
            iou_c.append(i)
        summary = {"dataset": test_set, **summarise(rows, dice_c, iou_c), "per_image": rows}
        out["sets"].append(summary)
        print(
            f"{test_set:18s} n={summary['n_images']:3d}  mDice {summary['mdice']:.4f}  "
            f"Dice@0.5 {summary['dice']:.4f} [{summary['dice_ci'][0]:.3f}, {summary['dice_ci'][1]:.3f}]  "
            f"BIoU {summary['boundary_iou']:.4f}  {time.perf_counter() - start:.0f}s",
            flush=True,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="folder name under runs/, or a path")
    parser.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"])
    parser.add_argument(
        "--limit", type=int, default=None, help="first N images per set (smoke test; not saved)"
    )
    args = parser.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_dir() else REPO / "runs" / args.run
    result = evaluate_run(run_dir, get_device(args.device), args.limit)
    if args.limit is None:
        (run_dir / "test.json").write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {(run_dir / 'test.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
