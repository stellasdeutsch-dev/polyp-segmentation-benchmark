"""Training losses (decision D6). All take raw logits (B, 1, H, W) and binary targets (B, 1, H, W).

  dice_bce            default for every model: BCE + soft Dice, weight 1 each (nnU-Net's default,
                      nnUNetTrainer.py:474-482 at commit 90cc65a)
  bce                 ablation: cross-entropy only
  structure           ablation: PraNet's boundary-weighted BCE + IoU, as intended
  structure_pranet    ablation: the same loss exactly as published, including a bug (see below)

The bug. PraNet (MyTrain.py:14 at commit 95f12bd) and Polyp-PVT (Train.py:17) compute the BCE with
`reduce='none'`, a deprecated boolean argument. PyTorch reads the string as "reduce", returns one
averaged number, and the boundary weights then cancel out: the "weighted" BCE equals the plain mean
BCE. Only the IoU half of the published loss is boundary-weighted. Checked in sub-step 2.5.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SMOOTH = 1e-5  # nnU-Net's Dice smoothing (nnUNetTrainer.py:476)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = SMOOTH) -> torch.Tensor:
    """1 - soft Dice, computed per image and averaged over the batch."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    total = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + smooth) / (total + smooth)).mean()


def bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return bce(logits, target) + soft_dice_loss(logits, target)


def boundary_weights(target: torch.Tensor, kernel: int = 31, strength: float = 5.0) -> torch.Tensor:
    """PraNet's weight map: 1 + 5 * |avgpool31(mask) - mask|. 1 in flat regions, larger near edges."""
    pooled = F.avg_pool2d(target, kernel_size=kernel, stride=1, padding=kernel // 2)
    return 1 + strength * torch.abs(pooled - target)


def _weighted_iou(logits: torch.Tensor, target: torch.Tensor, weit: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = ((p * target) * weit).sum(dim=(2, 3))
    union = ((p + target) * weit).sum(dim=(2, 3))
    return 1 - (inter + 1) / (union - inter + 1)


def structure(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PraNet's structure loss with the boundary weights applied to both terms, as intended."""
    weit = boundary_weights(target)
    wbce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return (wbce + _weighted_iou(logits, target, weit)).mean()


def structure_pranet(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PraNet's structure loss exactly as published: the BCE term ends up unweighted."""
    weit = boundary_weights(target)
    wbce = F.binary_cross_entropy_with_logits(logits, target)  # what reduce='none' actually returns
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))  # = the plain mean, weights cancel
    return (wbce + _weighted_iou(logits, target, weit)).mean()


LOSSES = {"dice_bce": dice_bce, "bce": bce, "structure": structure, "structure_pranet": structure_pranet}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; expected one of {sorted(LOSSES)}")
    return LOSSES[name]
