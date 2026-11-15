"""Model registry (decision D12). Every model maps (B, 3, S, S) images to (B, 1, S, S) logits.

Rungs are added in the stage that builds them:
  unet           rung 1: U-Net written from scratch, 64-1,024 channels (stage 4, seg.unet)
  smp_unet_r34   rung 2: U-Net with an ImageNet ResNet-34 encoder, library defaults (stage 5;
                 registered in 2.5 so the training loop can be checked end to end)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _smp_unet_r34(pretrained: bool = True) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )


def _unet(pretrained: bool = False) -> nn.Module:
    if pretrained:
        raise ValueError("the from-scratch U-Net has no pretrained weights; set pretrained: false")
    from seg.unet import UNet

    return UNet(in_channels=3, classes=1)


MODELS: dict[str, Callable[..., nn.Module]] = {
    "unet": _unet,
    "smp_unet_r34": _smp_unet_r34,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


@torch.inference_mode()
def check_output_shape(model: nn.Module, size: int = 64) -> None:
    """Fail early if a model does not return one logit map at input resolution."""
    model.eval()
    out = model(torch.zeros(1, 3, size, size))
    if out.shape != (1, 1, size, size):
        raise ValueError(f"model returned {tuple(out.shape)}, expected (1, 1, {size}, {size})")
