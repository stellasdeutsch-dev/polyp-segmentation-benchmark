"""The from-scratch U-Net (stage 4.1): shapes, size, initialisation, and an exact match with
milesial/Pytorch-UNet when its code has been fetched (`make milesial`)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

from seg.models import build_model
from seg.unet import WIDTHS, UNet

REPO = Path(__file__).resolve().parents[1]
MILESIAL = REPO / "third_party" / "milesial"


def expected_parameters(c_in: int = 3, classes: int = 1, widths=WIDTHS) -> int:
    """Count by hand: conv weights (no bias), BatchNorm scale and shift, transposed convs, head."""

    def double(a, b):
        return 9 * a * b + 9 * b * b + 4 * b

    n = double(c_in, widths[0])
    n += sum(double(a, b) for a, b in zip(widths[:-1], widths[1:], strict=True))
    for a, b in zip(widths[::-1][:-1], widths[::-1][1:], strict=True):
        n += 4 * a * (a // 2) + a // 2  # 2x2 transposed conv with bias
        n += double(a, b)
    return n + widths[0] * classes + classes


def test_parameter_count_matches_the_hand_count():
    model = UNet()
    assert sum(p.numel() for p in model.parameters()) == expected_parameters() == 31_037_633


@pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 256, 256), (1, 3, 250, 190), (1, 3, 33, 47)])
def test_output_has_the_input_size(shape):
    model = UNet().eval()
    with torch.no_grad():
        out = model(torch.randn(shape))
    assert out.shape == (shape[0], 1, *shape[2:])


def test_he_initialisation():
    torch.manual_seed(0)
    model = UNet()
    conv = model.downs[3][1][0]  # 512 -> 1024, fan_in 4,608: about 4.7M weights
    std = conv.weight.std().item()
    assert abs(std - (2 / conv.weight[0].numel()) ** 0.5) / std < 0.01
    assert model.head.bias.abs().max() == 0


def test_every_level_reaches_the_output():
    """A gradient from the output must reach every encoder level, including the deepest."""
    model = UNet()
    model(torch.randn(2, 3, 32, 32)).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_registry_refuses_pretrained_weights():
    with pytest.raises(ValueError, match="no pretrained"):
        build_model("unet", pretrained=True)
    assert isinstance(build_model("unet", pretrained=False), UNet)


@pytest.fixture(scope="module")
def milesial_unet():
    if not (MILESIAL / "unet" / "unet_model.py").exists():
        pytest.skip("milesial code not fetched; run `make milesial`")
    sys.path.insert(0, str(MILESIAL))
    try:
        return importlib.import_module("unet").UNet
    finally:
        sys.path.remove(str(MILESIAL))


def copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Both register their layers in the same order; copy tensor by tensor, checking shapes."""
    a, b = src.state_dict(), dst.state_dict()
    assert len(a) == len(b)
    for (ka, va), (kb, vb) in zip(a.items(), b.items(), strict=True):
        assert va.shape == vb.shape, f"{ka} {tuple(va.shape)} vs {kb} {tuple(vb.shape)}"
        b[kb] = va.clone()
    dst.load_state_dict(b)


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
def test_same_function_as_milesial(milesial_unet, train):
    torch.manual_seed(0)
    ref, ours = milesial_unet(n_channels=3, n_classes=1, bilinear=False), UNet()
    copy_weights(ref, ours)
    ref.train(train)
    ours.train(train)
    x = torch.randn(2, 3, 72, 88)
    y_ref, y_ours = ref(x), ours(x)
    assert torch.equal(y_ref, y_ours)
    if train:  # BatchNorm uses batch statistics here; gradients must agree too
        y_ref.square().mean().backward()
        y_ours.square().mean().backward()
        for p, q in zip(ref.parameters(), ours.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)


def test_milesial_memory_fallback_crashes(milesial_unet):
    """milesial's train.py calls use_checkpointing() after a CUDA out-of-memory error; it fails."""
    with pytest.raises(TypeError, match="not callable"):
        milesial_unet(n_channels=3, n_classes=1).use_checkpointing()
