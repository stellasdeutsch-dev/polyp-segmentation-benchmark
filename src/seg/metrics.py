"""Segmentation metrics (decision D4).

Main numbers, per image, at a fixed threshold:
  * Dice and IoU.
  * Boundary IoU (Cheng et al., CVPR 2021): IoU restricted to mask pixels within d of the contour,
    d = 2% of the image diagonal.
Comparison with the literature only:
  * PraNet-style mDice / mIoU: per-image min-max stretch, 8-bit quantisation, Dice at 256
    thresholds, averaged. A port of PraNet's MyTest.py + eval/main.m + eval/Fmeasure_calu.m.

Conventions:
  * A pixel is foreground when probability >= threshold (as in PraNet's Fmeasure_calu.m).
  * Per-image metrics, averaged over images by the caller (not pooled over pixels).
  * Empty prediction on an empty mask scores 1 for Dice, IoU and Boundary IoU. PraNet's
    evaluation scores it 0; `pranet_curves` keeps that rule so its numbers stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_THRESHOLD = 0.5
BOUNDARY_DILATION_RATIO = 0.02


def _as_bool(x: torch.Tensor, threshold: float) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x
    return x >= threshold


def _check(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: prediction {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.dim() != 3:
        raise ValueError(f"expected a batch of 2D masks (B, H, W), got {tuple(pred.shape)}")


def confusion(pred: torch.Tensor, target: torch.Tensor, threshold: float = DEFAULT_THRESHOLD):
    """Per-image true positives, false positives and false negatives, each of shape (B,)."""
    _check(pred, target)
    p, t = _as_bool(pred, threshold), _as_bool(target, 0.5)
    tp = (p & t).flatten(1).sum(1)
    fp = (p & ~t).flatten(1).sum(1)
    fn = (~p & t).flatten(1).sum(1)
    return tp, fp, fn


def _ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """num / den per image, with 0/0 defined as 1 (empty prediction on empty mask).

    Pixel counting happens on the input's device; the per-image counts are moved to the CPU and
    divided in float64 there, because Apple's MPS backend has no float64. Returns a CPU tensor.
    """
    num, den = num.cpu().double(), den.cpu().double()
    return torch.where(den > 0, num / den.clamp_min(1), torch.ones_like(den))


def dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> torch.Tensor:
    """Per-image Dice = 2TP / (2TP + FP + FN), shape (B,)."""
    tp, fp, fn = confusion(pred, target, threshold)
    return _ratio(2 * tp, 2 * tp + fp + fn)


def iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> torch.Tensor:
    """Per-image IoU = TP / (TP + FP + FN), shape (B,)."""
    tp, fp, fn = confusion(pred, target, threshold)
    return _ratio(tp, tp + fp + fn)


def boundary_dilation(height: int, width: int, ratio: float = BOUNDARY_DILATION_RATIO) -> int:
    """Boundary width d in pixels: round(ratio * image diagonal), at least 1 (as in the official code)."""
    return max(1, int(round(ratio * (height**2 + width**2) ** 0.5)))


def _sliding_any(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """True where any of k consecutive elements along `dim` is True (output shorter by k - 1).

    Window doubling: after j steps each element covers 2^j inputs; two overlapping windows of the
    largest power of two <= k then cover exactly k. About log2(k) element-wise ORs on any device.
    """
    out_len = x.shape[dim] - k + 1
    m, w = x, 1
    while 2 * w <= k:
        n = m.shape[dim] - w
        m = m.narrow(dim, 0, n) | m.narrow(dim, w, n)
        w *= 2
    return m.narrow(dim, 0, out_len) | m.narrow(dim, k - w, out_len)


def boundary_region(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    """Mask pixels within chessboard distance `dilation` of the background ("G_d ∩ G" in the paper).

    Same result as the official implementation (erode `dilation` times with a 3x3 square after
    padding with background), computed as one (2d+1)x(2d+1) window max over the background: a row
    pass then a column pass, each by window doubling (`_sliding_any`). Found in 3.2: PyTorch's CPU
    max-pool with stride 1 took 50 s for d = 31 on one 1225 x 966 image; this takes milliseconds.
    Pixels outside the image count as background, so a mask cut by the image edge has a boundary there.
    """
    if mask.dim() != 3 or mask.dtype != torch.bool:
        raise ValueError("expected a boolean (B, H, W) mask")
    padded = F.pad((~mask).float(), (dilation,) * 4, value=1.0) > 0
    k = 2 * dilation + 1
    near_background = _sliding_any(_sliding_any(padded, k, dim=2), k, dim=1)
    return mask & near_background


def boundary_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    dilation_ratio: float = BOUNDARY_DILATION_RATIO,
) -> torch.Tensor:
    """Per-image Boundary IoU, shape (B,). Both boundaries empty scores 1."""
    _check(pred, target)
    p, t = _as_bool(pred, threshold), _as_bool(target, 0.5)
    d = boundary_dilation(p.shape[-2], p.shape[-1], dilation_ratio)
    pb, tb = boundary_region(p, d), boundary_region(t, d)
    inter = (pb & tb).flatten(1).sum(1)
    union = (pb | tb).flatten(1).sum(1)
    return _ratio(inter, union)


# ----------------------------------------------------------------------------------------------
# PraNet-style evaluation, for comparison with published numbers only
# ----------------------------------------------------------------------------------------------

N_THRESHOLDS = 256  # eval/main.m: Thresholds = 1:-1/255:0


def pranet_quantise(prob: np.ndarray) -> np.ndarray:
    """What PraNet does to a probability map before scoring, as 8-bit levels 0-255.

    MyTest.py:37 stretches the map to [0, 1] and saves it as a PNG (8 bits); main.m:85-88 reads
    it back and stretches it again. A constant map has nothing to stretch and becomes all zeros.
    """
    prob = np.asarray(prob, dtype=np.float64)
    lo, hi = prob.min(), prob.max()
    if hi - lo <= 0:
        return np.zeros(prob.shape, dtype=np.uint8)
    levels = np.clip((prob - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lo8, hi8 = int(levels.min()), int(levels.max())
    if hi8 == lo8:
        return np.zeros(prob.shape, dtype=np.uint8)
    # second stretch (mapminmax in main.m:88), kept in integer levels
    return np.round((levels.astype(np.float64) - lo8) / (hi8 - lo8) * 255.0).astype(np.uint8)


def pranet_curves(prob: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dice and IoU of one image at each of the 256 PraNet thresholds, highest threshold first.

    `prob`: probability map at the ground-truth resolution. `gt`: ground-truth image (0-255 or bool);
    foreground is > 128 for 8-bit masks, as in main.m:73. Follows Fmeasure_calu.m: a pixel is
    predicted when its level >= threshold, and both scores are 0 whenever there is no overlap.
    """
    levels = pranet_quantise(prob).ravel()
    g = np.asarray(gt)
    fg = as_foreground(g).ravel()
    if levels.shape != fg.shape:
        raise ValueError("probability map and ground truth must have the same size")
    fg_hist = np.bincount(levels[fg], minlength=256)
    all_hist = np.bincount(levels, minlength=256)
    # threshold index k = 0..255 means level >= 255 - k, i.e. thresholds 1, 254/255, ..., 0
    tp = np.cumsum(fg_hist[::-1])
    predicted = np.cumsum(all_hist[::-1])
    n_fg = fg.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        dice_c = np.where(tp > 0, 2.0 * tp / (n_fg + predicted), 0.0)
        iou_c = np.where(tp > 0, tp / (n_fg + predicted - tp), 0.0)
    return dice_c, iou_c


@dataclass(frozen=True)
class PraNetSummary:
    mean_dice: float
    mean_iou: float
    max_dice: float
    max_iou: float


def pranet_summary(dice_curves: np.ndarray, iou_curves: np.ndarray) -> PraNetSummary:
    """Dataset-level numbers as main.m reports them: average over images per threshold, then
    mean (meanDic) or max (maxDic) over thresholds."""
    d, i = np.asarray(dice_curves).mean(axis=0), np.asarray(iou_curves).mean(axis=0)
    return PraNetSummary(float(d.mean()), float(i.mean()), float(d.max()), float(i.max()))


def as_foreground(gt: np.ndarray) -> np.ndarray:
    """Ground truth as a boolean foreground mask: a bool array, or an 8-bit mask where > 128 is
    foreground (main.m:73).

    An integer mask whose largest value is 1 is refused: read as 8-bit it would be all
    background. Found in 4.3, when our loader's 0/1 masks scored every test image as empty.
    """
    g = np.asarray(gt)
    if g.dtype == bool:
        return g
    if g.dtype.kind not in "iu":
        raise ValueError(f"ground truth must be bool or 8-bit, got {g.dtype}")
    if g.max() == 1:
        raise ValueError("0/1 integer mask: pass it as a bool array")
    return g > 128


def matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB's `start:step:stop` in float64, element by element.

    MATLAB builds the first half forward from `start` and the second half backward from `stop`
    (Moler, "colonop"), so its values differ from `start + k * step` in the last bits. Those bits
    decide ties when a stretched map value sits exactly on a threshold.
    """
    n = int(round((stop - start) / step))
    k = np.arange(n + 1)
    out = np.where(k <= n // 2, start + k * step, stop - (n - k) * step)
    if n % 2 == 0:
        out[n // 2] = (start + stop) / 2
    return out


MATLAB_THRESHOLDS = matlab_colon(1.0, -1 / 255, 0.0)  # eval/main.m:30


def pranet_curves_saved(levels: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dice and IoU at PraNet's 256 thresholds for a map already saved as an 8-bit PNG.

    The exact path of eval/main.m for one image: `im2double` (level / 255), one `mapminmax`
    stretch to [0, 1] in float64, then `map >= t` for t in MATLAB's `1:-1/255:0`, scored by
    Fmeasure_calu.m (both scores 0 when nothing overlaps). Use this on the authors' published maps;
    `pranet_curves` covers our own float maps, which still have to go through MyTest.py's saving.

    mapminmax leaves a constant map unchanged (its gain falls back to 1), unlike `pranet_quantise`.
    """
    lv = np.asarray(levels)
    if lv.dtype != np.uint8:
        raise ValueError("expected an 8-bit map")
    g = np.asarray(gt)
    fg = as_foreground(g).ravel()
    lv = lv.ravel()
    if lv.shape != fg.shape:
        raise ValueError("map and ground truth must have the same size")
    x = np.arange(256, dtype=np.float64) / 255.0  # value of every possible level after im2double
    xmin, xmax = x[lv.min()], x[lv.max()]
    # mapminmax with ymin = 0, ymax = 1; for a zero range it sets gain = 1 and offset = ymin
    gain, offset = (1.0 / (xmax - xmin), xmin) if xmax > xmin else (1.0, 0.0)
    stretched = (x - offset) * gain
    predicted_at = stretched[None, :] >= np.minimum(MATLAB_THRESHOLDS, 1.0)[:, None]  # 256 x 256
    fg_hist = np.bincount(lv[fg], minlength=256)
    all_hist = np.bincount(lv, minlength=256)
    tp = predicted_at @ fg_hist
    predicted = predicted_at @ all_hist
    n_fg = fg.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        dice_c = np.where(tp > 0, 2.0 * tp / (n_fg + predicted), 0.0)
        iou_c = np.where(tp > 0, tp / (n_fg + predicted - tp), 0.0)
    return dice_c, iou_c


def pranet_save_levels(prob: np.ndarray) -> np.ndarray:
    """The 8-bit PNG levels MyTest.py writes for a probability map at ground-truth size.

    MyTest.py:37 stretches the map, `(res - min) / (max - min + 1e-8)`, and `scipy.misc.imsave`
    (scipy < 1.2) byte-scales it again: `(x - min) * 255 / (max - min) + 0.5`, clipped, truncated.
    A constant map has no range and becomes all zeros, as bytescale does. Score the result with
    `pranet_curves_saved`, exactly as main.m scores the PNG.
    """
    res = np.asarray(prob, dtype=np.float64)
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # MyTest.py:37
    lo, hi = res.min(), res.max()
    if hi - lo <= 0:
        return np.zeros(res.shape, dtype=np.uint8)
    return (np.clip((res - lo) * (255.0 / (hi - lo)), 0, 255) + 0.5).astype(np.uint8)
