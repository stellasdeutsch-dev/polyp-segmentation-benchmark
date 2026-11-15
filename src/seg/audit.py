"""Data audit helpers: image features, nearest-neighbour similarity, and a rank-based AUC.

Used by scripts/make_splits.py to check the CVC-ClinicDB sequence table against the images
and to measure how close each test image is to its nearest training image.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


@torch.inference_mode()
def image_features(paths: Sequence[Path], device: torch.device, batch_size: int = 64) -> np.ndarray:
    """L2-normalised ResNet-18 (ImageNet) features, one row per image.

    A generic feature extractor, used only to compare images with each other.
    """
    import torchvision
    from PIL import Image

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    transform = weights.transforms()
    out = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]]).to(
            device
        )
        out.append(torch.nn.functional.normalize(net(x), dim=1).cpu())
    return torch.cat(out).numpy()


def nearest_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each query row, the highest cosine similarity to any reference row (rows are unit length)."""
    return (query @ reference.T).max(axis=1)


def auc_low_is_positive(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """Probability that a random positive scores lower than a random negative (ties count half)."""
    pos, neg = scores[is_positive], scores[~is_positive]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need at least one positive and one negative")
    less = (pos[:, None] < neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(less + 0.5 * ties)
