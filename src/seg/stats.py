"""Uncertainty over test images (decision D5).

A test set is a sample of images, so a mean score has sampling error. We resample images with
replacement and read the 2.5th and 97.5th percentiles of the resampled means (percentile
bootstrap). The generator is seeded, so the same scores always give the same interval.
"""

from __future__ import annotations

import numpy as np

N_RESAMPLES = 10_000


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = N_RESAMPLES, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-image scores."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_resamples: int = N_RESAMPLES, level: float = 0.95, seed: int = 0
) -> dict:
    """Mean difference a - b over the same images, its interval, and the share of resamples where
    the sign flips (a two-sided bootstrap p-value)."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("paired scores must cover the same images")
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "p": float(min(1.0, p))}
