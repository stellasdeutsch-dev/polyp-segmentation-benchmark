"""Draw real training images as the models will see them, with the mask outline on top.

    make preview

Writes reports/figures/augmentations.png: one row per image; the first column is the
evaluation view (resize only), the others are random training augmentations (decision D10).
If the outline does not sit on the polyp, the pipeline is broken.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seg.data import IMAGENET_MEAN, IMAGENET_STD, PolypDataset, eval_transform, split_items, train_transform

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "figures" / "augmentations.png"


def to_rgb(image) -> np.ndarray:
    mean, std = np.array(IMAGENET_MEAN)[:, None, None], np.array(IMAGENET_STD)[:, None, None]
    return np.clip(image.numpy() * std + mean, 0, 1).transpose(1, 2, 0)


def outline(rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.85, 0.0)) -> np.ndarray:
    m = (mask > 0.5).astype(np.uint8)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out = rgb.copy()
    out[edge] = color
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, _ = split_items()
    rng = np.random.default_rng(args.seed)
    kvasir = [i for i in train if i.source == "Kvasir"]
    clinic = [i for i in train if i.source == "CVC-ClinicDB"]
    picks = [kvasir[j] for j in rng.choice(len(kvasir), 2, replace=False)]
    picks += [clinic[j] for j in rng.choice(len(clinic), 2, replace=False)]

    plain = PolypDataset(picks, eval_transform(args.size))
    aug = PolypDataset(picks, train_transform(args.size))
    cols = 1 + args.draws
    fig, axes = plt.subplots(len(picks), cols, figsize=(1.9 * cols, 1.9 * len(picks)), dpi=150)
    for r in range(len(picks)):
        s = plain[r]
        axes[r, 0].imshow(outline(to_rgb(s["image"]), s["mask"][0].numpy()))
        for c in range(1, cols):
            np.random.seed(args.seed * 100 + r * 10 + c)
            s = aug[r]
            axes[r, c].imshow(outline(to_rgb(s["image"]), s["mask"][0].numpy()))
        axes[r, 0].set_ylabel(f"{picks[r].source}\n{picks[r].stem[:10]}", fontsize=7)
    axes[0, 0].set_title("evaluation view", fontsize=8)
    for c in range(1, cols):
        axes[0, c].set_title(f"training draw {c}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
