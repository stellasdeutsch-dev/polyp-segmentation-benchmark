"""Tests for seg.stats (bootstrap intervals over test images)."""

from __future__ import annotations

import numpy as np
import pytest

from seg.stats import bootstrap_ci, paired_bootstrap


def test_interval_contains_the_mean_and_repeats_with_the_seed():
    v = np.random.default_rng(0).uniform(0, 1, 200)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi
    assert bootstrap_ci(v) == (lo, hi)


def test_interval_narrows_with_more_images():
    rng = np.random.default_rng(1)
    small, large = rng.uniform(0, 1, 50), rng.uniform(0, 1, 2000)
    w_small = np.subtract(*bootstrap_ci(small)[::-1])
    w_large = np.subtract(*bootstrap_ci(large)[::-1])
    assert w_large < w_small / 3  # width shrinks like 1 / sqrt(n)


def test_interval_matches_the_normal_approximation():
    v = np.random.default_rng(2).normal(0.8, 0.1, 400)
    lo, hi = bootstrap_ci(v)
    half = 1.96 * v.std(ddof=1) / np.sqrt(v.size)
    assert (hi - lo) / 2 == pytest.approx(half, rel=0.1)


def test_paired_difference_detects_a_real_gap_and_ignores_none():
    rng = np.random.default_rng(3)
    base = rng.uniform(0.5, 0.9, 300)
    better = paired_bootstrap(base + 0.02 + rng.normal(0, 0.01, 300), base)
    assert better["mean"] == pytest.approx(0.02, abs=0.003)
    assert better["ci"][0] > 0 and better["p"] < 0.001
    same = paired_bootstrap(base + rng.normal(0, 0.01, 300), base)
    assert same["ci"][0] < 0 < same["ci"][1] and same["p"] > 0.05


def test_paired_needs_the_same_images():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4))
