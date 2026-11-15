"""Tests for seg.env: device choice, seeding, cost measurement, run record."""

from __future__ import annotations

import pytest
import torch

from seg.env import environment_report, get_device, git_state, measure, seed_everything

HAS_ACCELERATOR = torch.backends.mps.is_available() or torch.cuda.is_available()


def test_get_device_default_is_available():
    device = get_device()
    assert device.type in {"mps", "cuda", "cpu"}
    torch.zeros(1, device=device)  # usable


def test_get_device_cpu_always_works():
    assert get_device("cpu").type == "cpu"


def test_get_device_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_get_device_refuses_unavailable_preference():
    with pytest.raises(RuntimeError):
        get_device("cuda")


def test_seed_makes_cpu_draws_repeatable():
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    seed_everything(124)
    c = torch.randn(5)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_measure_reports_time_on_cpu_and_no_peak():
    with measure(torch.device("cpu")) as cost:
        sum(range(10_000))
    assert cost.seconds > 0
    assert cost.peak_bytes is None


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="needs MPS or CUDA")
def test_measure_peak_matches_known_allocation():
    device = get_device()
    size = 64 * 2**20  # 64 MB
    with measure(device) as cost:
        x = torch.empty(size // 4, device=device)
        del x
    assert abs(cost.peak_bytes - size) < 2**20


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="needs MPS or CUDA")
def test_measure_resets_peak_between_blocks():
    device = get_device()
    with measure(device):
        big = torch.empty(128 * 2**20 // 4, device=device)
        del big
    with measure(device) as second:
        small = torch.empty(8 * 2**20 // 4, device=device)
        del small
    assert second.peak_mb < 16


def test_environment_report_has_reproducibility_fields():
    report = environment_report(torch.device("cpu"))
    for key in ("python", "torch", "torchvision", "numpy", "device", "git"):
        assert key in report
    assert set(report["git"]) == {"commit", "dirty"}


def test_git_state_outside_a_repo_returns_none(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None}
