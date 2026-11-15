"""Scoring one test set, the same way for every model (decisions D4, D5, D13).

Every prediction ends up as an 8-bit map at ground-truth size: the authors' released PNGs as they
are, and live model output after `metrics.pranet_save_levels`. One function then scores it with
both definitions side by side:
  * ours: Dice, IoU and Boundary IoU at 0.5, per image;
  * PraNet-style: Dice and IoU at 256 thresholds, exactly as eval/main.m.

For a model we run ourselves, `score_probability_map` takes "at 0.5" on the raw probability (D4,
the threshold a deployed model would use and the one model selection uses), and the PraNet-style
scores on the 8-bit map MyTest.py would have saved. `score_saved_map` only has the saved map, so
its "at 0.5" is on the stretched map; found in 4.3, see research/journal.yaml.
"""

from __future__ import annotations

import numpy as np
import torch

from seg import metrics
from seg.stats import bootstrap_ci


def score_saved_map(levels: np.ndarray, gt: np.ndarray, name: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-image row plus the image's Dice and IoU curves over PraNet's thresholds."""
    d, i = metrics.pranet_curves_saved(levels, gt)
    p = torch.from_numpy(levels.astype(np.float32) / 255.0)[None]
    t = torch.from_numpy(metrics.as_foreground(gt))[None]
    row = {
        "image": name,
        "mdice": float(d.mean()),
        "miou": float(i.mean()),
        "dice": float(metrics.dice(p, t)[0]),
        "iou": float(metrics.iou(p, t)[0]),
        "boundary_iou": float(metrics.boundary_iou(p, t)[0]),
    }
    return row, d, i


def score_probability_map(prob: np.ndarray, gt: np.ndarray, name: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-image row for live model output: a probability map at ground-truth size."""
    row, d, i = score_saved_map(metrics.pranet_save_levels(prob), gt, name)
    p = torch.from_numpy(np.asarray(prob, dtype=np.float32))[None]
    t = torch.from_numpy(metrics.as_foreground(gt))[None]
    row["dice"] = float(metrics.dice(p, t)[0])
    row["iou"] = float(metrics.iou(p, t)[0])
    row["boundary_iou"] = float(metrics.boundary_iou(p, t)[0])
    return row, d, i


def summarise(rows: list[dict], dice_curves: list[np.ndarray], iou_curves: list[np.ndarray]) -> dict:
    """Test-set numbers: PraNet-style means and maxima, our means, and bootstrap intervals."""
    s = metrics.pranet_summary(np.stack(dice_curves), np.stack(iou_curves))
    per = {k: np.array([r[k] for r in rows]) for k in ("mdice", "dice", "iou", "boundary_iou")}
    return {
        "n_images": len(rows),
        "mdice": s.mean_dice,
        "miou": s.mean_iou,
        "max_dice": s.max_dice,
        "mdice_ci": bootstrap_ci(per["mdice"]),
        "dice": float(per["dice"].mean()),
        "dice_ci": bootstrap_ci(per["dice"]),
        "iou": float(per["iou"].mean()),
        "boundary_iou": float(per["boundary_iou"].mean()),
        "dice_curve": np.stack(dice_curves).mean(axis=0).round(5).tolist(),
    }
