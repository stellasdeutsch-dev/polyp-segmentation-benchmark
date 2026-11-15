"""Tests for seg.metrics.

Three kinds of evidence:
  1. hand-worked cases with known answers;
  2. agreement with independent implementations (SMP, MONAI, the official Boundary IoU code);
  3. a slow, literal port of PraNet's MATLAB loop, checked against the fast version.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from seg.metrics import (
    MATLAB_THRESHOLDS,
    boundary_dilation,
    boundary_iou,
    boundary_region,
    confusion,
    dice,
    iou,
    matlab_colon,
    pranet_curves,
    pranet_curves_saved,
    pranet_quantise,
    pranet_save_levels,
    pranet_summary,
)
from tests.reference.boundary_iou_official import mask_to_boundary

RNG = np.random.default_rng(0)


def blobs(n: int, h: int, w: int, rng=RNG) -> torch.Tensor:
    """Random masks with blob-like shapes, including some empty and some full-frame."""
    out = np.zeros((n, h, w), dtype=bool)
    yy, xx = np.mgrid[:h, :w]
    for i in range(n):
        kind = i % 10
        if kind == 0:
            continue  # empty
        if kind == 1:
            out[i] = True  # full frame, touches every edge
            continue
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
            ry, rx = rng.uniform(2, h / 2), rng.uniform(2, w / 2)
            out[i] |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
    return torch.from_numpy(out)


# ---------------------------------------------------------------- hand-worked cases


def test_dice_and_iou_on_a_known_case():
    pred = torch.zeros(1, 4, 4, dtype=torch.bool)
    target = torch.zeros(1, 4, 4, dtype=torch.bool)
    pred[0, :2, :2] = True  # 4 pixels
    target[0, :2, :3] = True  # 6 pixels, 4 shared
    tp, fp, fn = confusion(pred, target)
    assert (tp.item(), fp.item(), fn.item()) == (4, 0, 2)
    assert dice(pred, target).item() == pytest.approx(2 * 4 / (4 + 6))
    assert iou(pred, target).item() == pytest.approx(4 / 6)


def test_empty_prediction_on_empty_mask_scores_one():
    z = torch.zeros(2, 8, 8, dtype=torch.bool)
    assert dice(z, z).tolist() == [1.0, 1.0]
    assert iou(z, z).tolist() == [1.0, 1.0]
    assert boundary_iou(z, z).tolist() == [1.0, 1.0]


def test_missing_the_object_entirely_scores_zero():
    target = torch.zeros(1, 8, 8, dtype=torch.bool)
    target[0, 2:5, 2:5] = True
    empty = torch.zeros_like(target)
    assert dice(empty, target).item() == 0.0
    assert iou(empty, target).item() == 0.0


def test_threshold_is_inclusive():
    prob = torch.tensor([[[0.5, 0.49]]])
    target = torch.tensor([[[True, False]]])
    assert dice(prob, target).item() == 1.0  # 0.5 counts as foreground, 0.49 does not
    assert dice(prob, target, threshold=0.6).item() == 0.0


def test_iou_and_dice_are_linked():
    pred, target = blobs(40, 32, 32), blobs(40, 32, 32, np.random.default_rng(1))
    d, j = dice(pred, target), iou(pred, target)
    torch.testing.assert_close(j, d / (2 - d), rtol=0, atol=1e-12)


def test_shape_errors():
    with pytest.raises(ValueError, match="mismatch"):
        dice(torch.zeros(1, 4, 4), torch.zeros(1, 4, 5))
    with pytest.raises(ValueError, match="batch"):
        dice(torch.zeros(4, 4), torch.zeros(4, 4))


@pytest.mark.parametrize(
    ("h", "w", "d"), [(352, 352, 10), (256, 256, 7), (288, 384, 10), (20, 20, 1), (5, 5, 1)]
)
def test_boundary_dilation(h, w, d):
    assert boundary_dilation(h, w) == d


def test_boundary_of_a_square_is_its_inner_ring():
    mask = torch.zeros(1, 20, 20, dtype=torch.bool)
    mask[0, 5:15, 5:15] = True
    ring = boundary_region(mask, dilation=2)
    expected = mask.clone()
    expected[0, 7:13, 7:13] = False  # interior beyond 2 pixels from the background
    assert torch.equal(ring, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_metrics_run_on_mps_and_match_cpu():
    """MPS has no float64; counting runs on the GPU, the division on the CPU."""
    prob = torch.from_numpy(RNG.uniform(size=(8, 64, 64))).float()
    target = blobs(8, 64, 64)
    for fn in (dice, iou, boundary_iou):
        on_gpu = fn(prob.to("mps"), target.to("mps"))
        assert on_gpu.device.type == "cpu"
        torch.testing.assert_close(on_gpu, fn(prob, target))


# ---------------------------------------------------------------- agreement with SMP and MONAI


@pytest.mark.parametrize("size", [(16, 16), (37, 53), (64, 64)])
def test_dice_and_iou_match_smp(size):
    smp = pytest.importorskip("segmentation_models_pytorch")
    pred, target = blobs(60, *size), blobs(60, *size, np.random.default_rng(2))
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred.long().unsqueeze(1), target.long().unsqueeze(1), mode="binary"
    )
    ref_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").squeeze(1).double()
    ref_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").squeeze(1).double()
    torch.testing.assert_close(dice(pred, target), ref_dice)
    torch.testing.assert_close(iou(pred, target), ref_iou)


def test_dice_and_iou_match_monai():
    monai_metrics = pytest.importorskip("monai.metrics")
    pred, target = blobs(60, 48, 40), blobs(60, 48, 40, np.random.default_rng(3))
    p, t = pred.float().unsqueeze(1), target.float().unsqueeze(1)
    ref_dice = (
        monai_metrics.compute_dice(p, t, include_background=True, ignore_empty=False).squeeze(1).double()
    )
    ref_iou = monai_metrics.compute_iou(p, t, include_background=True, ignore_empty=False).squeeze(1).double()
    torch.testing.assert_close(dice(pred, target), ref_dice)
    torch.testing.assert_close(iou(pred, target), ref_iou)


def test_dice_from_probabilities_matches_smp_threshold():
    smp = pytest.importorskip("segmentation_models_pytorch")
    prob = torch.from_numpy(RNG.uniform(size=(20, 24, 24))).float()
    target = blobs(20, 24, 24)
    tp, fp, fn, tn = smp.metrics.get_stats(
        prob.unsqueeze(1), target.long().unsqueeze(1), mode="binary", threshold=0.5
    )
    ref = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").squeeze(1).double()
    # SMP uses prob > threshold; ties at exactly 0.5 have probability zero with continuous noise
    torch.testing.assert_close(dice(prob, target), ref)


# ---------------------------------------------------------------- agreement with official Boundary IoU code


@pytest.mark.parametrize("size", [(20, 20), (64, 48), (288, 384), (352, 352)])
def test_boundary_region_matches_official_code(size):
    masks = blobs(30, *size)
    d = boundary_dilation(*size)
    ours = boundary_region(masks, d).numpy()
    for i, m in enumerate(masks.numpy()):
        ref = mask_to_boundary(m.astype(np.uint8)).astype(bool)
        assert np.array_equal(ours[i], ref), f"mask {i} differs"


def test_boundary_iou_matches_official_boundaries():
    pred, target = blobs(30, 96, 128), blobs(30, 96, 128, np.random.default_rng(4))
    ours = boundary_iou(pred, target)
    for i in range(len(pred)):
        pb = mask_to_boundary(pred[i].numpy().astype(np.uint8)).astype(bool)
        tb = mask_to_boundary(target[i].numpy().astype(np.uint8)).astype(bool)
        union = (pb | tb).sum()
        ref = (pb & tb).sum() / union if union else 1.0
        assert ours[i].item() == pytest.approx(ref)


def test_boundary_iou_is_harsher_than_iou_on_large_objects():
    target = torch.zeros(1, 200, 200, dtype=torch.bool)
    target[0, 20:180, 20:180] = True
    pred = torch.zeros_like(target)
    pred[0, 26:186, 26:186] = True  # same shape, shifted by 6 pixels
    assert iou(pred, target).item() > 0.85
    assert boundary_iou(pred, target).item() < 0.5


# ---------------------------------------------------------------- PraNet-style evaluation


def literal_pranet(prob: np.ndarray, gt: np.ndarray, exact: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Slow line-by-line port of MyTest.py:37 + main.m:73-107 + Fmeasure_calu.m.

    exact=True compares 8-bit levels with integer thresholds (level >= 255 - k), which is what the
    thresholds mean. exact=False compares floats (level/255 >= 1 - k/255), where rounding drops
    pixels lying exactly on 66 of the 256 thresholds. MATLAB builds 1:-1/255:0 with its own rounding,
    so its tie behaviour is unknown here; the two variants bound it.
    """
    res = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)  # MyTest.py:37
    lo, hi = res.min(), res.max()  # scipy imsave rescales to 0..255 and rounds
    png = np.clip((res - lo) / (hi - lo) * 255 + 0.5, 0, 255).astype(np.uint8)
    resmap = png.astype(np.float64) / 255  # im2double
    resmap = (resmap - resmap.min()) / (resmap.max() - resmap.min())  # mapminmax(…, 0, 1)
    g = gt > 128
    levels = np.round(resmap * 255).astype(int)
    thresholds = 1 + np.arange(256) * (-1 / 255)  # 1:-1/255:0
    dices, ious = [], []
    for k, t in enumerate(thresholds):
        label = levels >= 255 - k if exact else resmap >= min(t, 1)
        tp = (label & g).sum()
        if tp == 0:
            dices.append(0.0)
            ious.append(0.0)
            continue
        fn, num_rec = g.sum() - tp, label.sum()
        dices.append(2 * tp / (g.sum() + label.sum()))
        ious.append(tp / (fn + num_rec))
    return np.array(dices), np.array(ious)


def test_pranet_quantise_spans_full_range_and_keeps_order():
    prob = RNG.uniform(0.2, 0.7, size=(30, 40))
    q = pranet_quantise(prob)
    assert q.dtype == np.uint8 and q.min() == 0 and q.max() == 255
    order = np.argsort(prob.ravel(), kind="stable")
    assert np.all(np.diff(q.ravel()[order].astype(int)) >= 0)


def test_pranet_quantise_constant_map_is_zero():
    assert not pranet_quantise(np.full((5, 5), 0.3)).any()


@pytest.mark.parametrize("seed", range(6))
def test_pranet_curves_match_literal_port(seed):
    rng = np.random.default_rng(seed)
    gt = np.zeros((48, 64), dtype=np.uint8)
    gt[10:30, 15:45] = 255
    prob = np.clip(gt / 255 * rng.uniform(0.3, 0.9) + rng.normal(0, 0.25, gt.shape), 0, 1)
    fast_d, fast_i = pranet_curves(prob, gt)
    slow_d, slow_i = literal_pranet(prob, gt, exact=True)
    np.testing.assert_allclose(fast_d, slow_d, atol=1e-12)
    np.testing.assert_allclose(fast_i, slow_i, atol=1e-12)
    # float-style ties at a threshold move the per-image mean by far less than our 0.005 trust tolerance
    float_d, _ = literal_pranet(prob, gt, exact=False)
    assert abs(fast_d.mean() - float_d.mean()) < 0.002


def test_pranet_scores_no_overlap_as_zero():
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[2:5, 2:5] = 255
    prob = np.zeros((10, 10))
    prob[7:9, 7:9] = 1.0  # confident, but in the wrong place
    d, i = pranet_curves(prob, gt)
    assert d[:255].max() == 0  # every threshold above 0 misses the polyp
    assert d[255] == pytest.approx(2 * 9 / (9 + 100))  # threshold 0 predicts everything


def test_pranet_summary_averages_over_images_then_thresholds():
    curves = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    s = pranet_summary(curves, curves)
    assert s.mean_dice == pytest.approx(0.5)
    assert s.max_dice == pytest.approx(0.5)  # max of per-threshold means, not mean of per-image maxima


# ---------------------------------------------------------------- saved 8-bit maps (stage 3.2)


def test_matlab_colon_has_256_thresholds_from_1_to_0():
    t = MATLAB_THRESHOLDS
    assert t.shape == (256,) and t[0] == 1.0 and t[-1] == 0.0
    assert np.all(np.diff(t) < 0)
    np.testing.assert_allclose(t, 1 - np.arange(256) / 255, atol=1e-15)


def test_matlab_colon_builds_second_half_from_the_end():
    t = matlab_colon(0.0, 0.1, 1.0)  # even count: the middle element is (start + stop) / 2
    assert t[5] == 0.5 and t[-1] == 1.0 and t[-2] == 1.0 - 0.1


def literal_main_m(levels: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """main.m:85-105 for one saved map, a threshold at a time, as Fmeasure_calu.m does it."""
    x = levels.astype(np.float64) / 255  # im2double
    gain, offset = (1 / (x.max() - x.min()), x.min()) if x.max() > x.min() else (1.0, 0.0)
    resmap = (x - offset) * gain  # mapminmax(resmap, 0, 1)
    g = gt > 128
    dices, ious = [], []
    for t in MATLAB_THRESHOLDS:
        label = resmap >= min(t, 1)
        tp = (label & g).sum()
        if tp == 0:
            dices.append(0.0)
            ious.append(0.0)
            continue
        dices.append(2 * tp / (g.sum() + label.sum()))
        ious.append(tp / (g.sum() - tp + label.sum()))
    return np.array(dices), np.array(ious)


@pytest.mark.parametrize("lo,hi", [(0, 255), (0, 200), (37, 255), (90, 140)])
def test_pranet_curves_saved_match_literal_main_m(lo, hi):
    rng = np.random.default_rng(lo + hi)
    gt = np.zeros((40, 50), dtype=np.uint8)
    gt[8:30, 10:35] = 255
    levels = rng.integers(lo, hi + 1, size=gt.shape).astype(np.uint8)
    levels[gt > 0] = np.maximum(levels[gt > 0], (lo + hi) // 2)
    fast = pranet_curves_saved(levels, gt)
    slow = literal_main_m(levels, gt)
    np.testing.assert_array_equal(fast[0], slow[0])
    np.testing.assert_array_equal(fast[1], slow[1])


def test_pranet_curves_saved_keep_a_constant_map():
    gt = np.zeros((6, 6), dtype=np.uint8)
    gt[:3] = 255
    d, _ = pranet_curves_saved(np.full((6, 6), 255, dtype=np.uint8), gt)
    assert d[0] == pytest.approx(2 * 18 / (18 + 36))  # everything predicted at every threshold


def test_pranet_curves_saved_reject_float_maps():
    with pytest.raises(ValueError):
        pranet_curves_saved(np.zeros((4, 4)), np.zeros((4, 4), dtype=np.uint8))


@pytest.mark.parametrize("seed", range(4))
def test_boundary_region_equals_square_max_pool(seed):
    """Window doubling gives exactly the (2d+1)x(2d+1) max-pool it replaces."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(50):
        h, w = torch.randint(1, 40, (2,), generator=g).tolist()
        mask = torch.rand(2, h, w, generator=g) > torch.rand(1, generator=g).item()
        d = int(torch.randint(1, 12, (1,), generator=g))
        background = F.pad((~mask).float().unsqueeze(1), (d,) * 4, value=1.0)
        expected = mask & (F.max_pool2d(background, 2 * d + 1, stride=1).squeeze(1) > 0)
        assert torch.equal(boundary_region(mask, d), expected)


# ---------------------------------------------------------------- saving like MyTest.py (stage 3.3)


def scipy_bytescale(data: np.ndarray) -> np.ndarray:
    """scipy.misc.bytescale as scipy 1.1 wrote it (cmin/cmax from the data, low 0, high 255)."""
    cmin, cmax = data.min(), data.max()
    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1
    scale = 255.0 / cscale
    return ((data - cmin) * scale).clip(0, 255) + 0.5


def test_pranet_save_levels_match_mytest_and_scipy():
    prob = RNG.uniform(0.1, 0.9, size=(37, 53))
    res = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)  # MyTest.py:37
    expected = scipy_bytescale(res).astype(np.uint8)
    np.testing.assert_array_equal(pranet_save_levels(prob), expected)


def test_pranet_save_levels_span_0_to_255_and_keep_order():
    prob = RNG.uniform(0.3, 0.6, size=(20, 30))
    lv = pranet_save_levels(prob)
    assert lv.dtype == np.uint8 and lv.min() == 0 and lv.max() == 255
    order = np.argsort(prob.ravel(), kind="stable")
    assert np.all(np.diff(lv.ravel()[order].astype(int)) >= 0)


def test_pranet_save_levels_of_a_constant_map_are_zero():
    assert not pranet_save_levels(np.full((4, 4), 0.7)).any()
