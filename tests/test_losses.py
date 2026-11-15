"""Tests for seg.losses."""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from seg.losses import LOSSES, boundary_weights, get_loss, soft_dice_loss, structure, structure_pranet


def batch(seed=0, b=3, s=40):
    g = torch.Generator().manual_seed(seed)
    target = torch.zeros(b, 1, s, s)
    target[:, :, 10:28, 12:30] = 1
    logits = torch.randn(b, 1, s, s, generator=g) * 2
    return logits, target


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_every_loss_is_finite_scalar_with_gradients(name):
    logits, target = batch()
    logits.requires_grad_(True)
    loss = get_loss(name)(logits, target)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_perfect_prediction_scores_lower_than_random(name):
    logits, target = batch()
    perfect = (target * 2 - 1) * 20  # confident and right everywhere
    fn = get_loss(name)
    assert fn(perfect, target) < fn(logits, target)


def test_soft_dice_of_a_confident_perfect_prediction_is_near_zero():
    _, target = batch()
    assert soft_dice_loss((target * 2 - 1) * 30, target) < 1e-6


def test_boundary_weights_are_one_inside_flat_regions_and_peak_at_edges():
    target = torch.zeros(1, 1, 80, 80)
    target[..., 20:60, 20:60] = 1
    w = boundary_weights(target)
    assert w[0, 0, 40, 40] == pytest.approx(1.0)  # deep inside
    assert w[0, 0, 2, 2] == pytest.approx(1.0)  # far outside
    edge = w[0, 0, 40, 20]  # on a straight edge the 31x31 window is about half polyp
    assert 3.3 < edge < 3.7
    assert w.max() <= 6.0


def test_published_structure_loss_equals_pranet_code():
    """structure_pranet must reproduce PraNet's MyTrain.py:12-21 line for line, warts included."""
    logits, target = batch(seed=1)

    def pranet_original(pred, mask):  # copied from PraNet MyTrain.py:12-21 (commit 95f12bd)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    torch.testing.assert_close(structure_pranet(logits, target), pranet_original(logits, target))


def test_published_bce_term_ignores_the_boundary_weights():
    """The bug, pinned down: in the published loss, changing the weights cannot change the BCE term."""
    logits, target = batch(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        published = F.binary_cross_entropy_with_logits(logits, target, reduce="none")
    assert published.dim() == 0  # one averaged number, not a per-pixel map
    torch.testing.assert_close(published, F.binary_cross_entropy_with_logits(logits, target))


def test_intended_structure_loss_differs_from_published():
    logits, target = batch(seed=3)
    assert not torch.isclose(structure(logits, target), structure_pranet(logits, target))


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError):
        get_loss("focal")
