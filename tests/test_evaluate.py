"""Tests for seg.evaluate: one scoring path for every model."""

from __future__ import annotations

import numpy as np
import pytest

from seg.evaluate import score_saved_map, summarise


def square(h: int = 40, w: int = 50) -> np.ndarray:
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[10:30, 12:38] = 255
    return gt


def test_perfect_map_scores_one_everywhere():
    gt = square()
    row, d, i = score_saved_map(gt.copy(), gt, "a")
    assert row["dice"] == row["iou"] == row["boundary_iou"] == 1.0
    assert row["mdice"] == pytest.approx(d.mean())
    assert d.shape == i.shape == (256,)
    # a binary 0/255 map is predicted at every threshold above 0, so all but the last are perfect
    assert np.all(d[:-1] == 1.0)


def test_shifted_map_scores_below_one_and_boundary_is_harsher():
    gt = square()
    pred = np.roll(gt, 4, axis=1)
    row, _, _ = score_saved_map(pred, gt, "b")
    assert 0 < row["dice"] < 1 and row["iou"] < row["dice"]
    assert row["boundary_iou"] < row["iou"]


def test_summary_averages_images_and_gives_intervals():
    gt = square()
    rows, dc, ic = [], [], []
    for k, shift in enumerate([0, 2, 6]):
        r, d, i = score_saved_map(np.roll(gt, shift, axis=0), gt, str(k))
        rows.append(r)
        dc.append(d)
        ic.append(i)
    s = summarise(rows, dc, ic)
    assert s["n_images"] == 3
    assert s["dice"] == pytest.approx(np.mean([r["dice"] for r in rows]))
    assert s["mdice"] == pytest.approx(np.mean([r["mdice"] for r in rows]))
    lo, hi = s["dice_ci"]
    assert lo <= s["dice"] <= hi
    assert len(s["dice_curve"]) == 256


def test_probability_map_scores_at_raw_half_and_pranet_on_saved_levels():
    from seg import metrics
    from seg.evaluate import score_probability_map, score_saved_map

    rng = np.random.default_rng(3)
    gt = np.zeros((40, 50), dtype=np.uint8)
    gt[10:30, 12:35] = 255
    # a timid model: everything below 0.5, so raw Dice at 0.5 is 0, but the stretched map is fine
    prob = np.clip(0.1 + 0.3 * (gt > 0) + rng.normal(0, 0.03, gt.shape), 0, 1)
    row, d, _ = score_probability_map(prob, gt, "x")
    saved, d_saved, _ = score_saved_map(metrics.pranet_save_levels(prob), gt, "x")
    assert row["dice"] == 0.0 and row["iou"] == 0.0
    assert saved["dice"] > 0.9  # the stretched map crosses 0.5 where the polyp is
    assert row["mdice"] == saved["mdice"] and np.array_equal(d, d_saved)


def test_zero_one_integer_masks_are_refused():
    """Our loader returns 0/1 masks; read as 8-bit they would be empty (found in 4.3)."""
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 5:15] = 1
    prob = np.full((20, 20), 0.2)
    prob[5:15, 5:15] = 0.9
    with pytest.raises(ValueError, match="bool"):
        score_saved_map((prob * 255).astype(np.uint8), gt, "x")
    from seg.evaluate import score_probability_map

    row, _, _ = score_probability_map(prob, gt.astype(bool), "x")
    assert row["dice"] == 1.0 and row["mdice"] > 0.5


def test_test_scoring_equals_validation_scoring():
    """scripts/eval_run.py and the training loop's validation give the same Dice at 0.5, so
    the checkpoint chosen on validation is scored on test by the same rule (D3, D4)."""
    import importlib.util
    from pathlib import Path

    import torch

    from seg.evaluate import score_probability_map
    from seg.train import evaluate
    from seg.unet import UNet

    spec = importlib.util.spec_from_file_location(
        "eval_run", Path(__file__).parents[1] / "scripts" / "eval_run.py"
    )
    eval_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_run)

    torch.manual_seed(0)
    model = UNet(widths=(4, 8, 16, 32, 64)).eval()
    images = torch.randn(3, 3, 32, 32)
    masks = [torch.rand(40 + 7 * k, 50) > 0.6 for k in range(3)]  # originals of different sizes
    loader = [{"image": images, "mask_full": masks}]
    expected = evaluate(model, loader, torch.device("cpu"))["dice"]
    scores = []
    for k in range(3):
        prob = eval_run.predict(model, {"image": images[k], "mask_full": masks[k]}, torch.device("cpu"))
        scores.append(score_probability_map(prob, masks[k].numpy(), str(k))[0]["dice"])
    assert abs(np.mean(scores) - expected) < 1e-6
