"""Device selection, seeding, peak-memory measurement and the run environment record.

Every training and evaluation script starts with these helpers, so device choice,
seeding and cost measurement are done the same way everywhere (decisions D14, D15).
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_device(prefer: str | None = None) -> torch.device:
    """Return the device to run on: the one asked for, else MPS, else CUDA, else CPU.

    `prefer` accepts "mps", "cuda", "cpu" or None. An unavailable preference raises,
    so a run never silently falls back to a slower device than the config says.
    """
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if prefer is not None:
        if prefer not in available:
            raise ValueError(f"unknown device {prefer!r}; expected one of {sorted(available)}")
        if not available[prefer]:
            raise RuntimeError(f"device {prefer!r} was requested but is not available")
        return torch.device(prefer)
    for name in ("mps", "cuda", "cpu"):
        if available[name]:
            return torch.device(name)
    raise AssertionError("unreachable: cpu is always available")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU, CUDA and MPS).

    On MPS this makes runs repeatable in distribution, not bit for bit: some Metal
    kernels are non-deterministic. We rely on three seeds and confidence intervals
    instead of identical reruns (decision D14).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    os.environ["PYTHONHASHSEED"] = str(seed)


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work, so timings and memory readings are complete."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class CostReading:
    """What a measured block of work cost. Peak memory is None on CPU."""

    seconds: float = 0.0
    peak_bytes: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def peak_mb(self) -> float | None:
        return None if self.peak_bytes is None else self.peak_bytes / 2**20


@contextmanager
def measure(device: torch.device) -> Iterator[CostReading]:
    """Measure wall-clock time and peak tensor memory of the enclosed block.

    Uses torch.accelerator's peak counter, which on PyTorch 2.15 works for MPS and
    CUDA (checked in sub-step 2.1: it tracks allocations exactly and catches
    short-lived activations). It counts memory held by tensors, not the Metal
    driver's own reserve, which grows in coarse 1 GB steps.
    """
    reading = CostReading()
    on_accelerator = device.type in ("mps", "cuda")
    synchronize(device)
    if on_accelerator:
        torch.accelerator.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield reading
    finally:
        synchronize(device)
        reading.seconds = time.perf_counter() - start
        if on_accelerator:
            reading.peak_bytes = int(torch.accelerator.max_memory_allocated())


def git_state(repo: Path = REPO_ROOT) -> dict:
    """Commit hash and whether the working tree has uncommitted changes."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def environment_report(device: torch.device | None = None) -> dict:
    """Everything needed to reproduce a run, saved next to its results (decision D14)."""
    import torchvision

    device = device or get_device()
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": device.type,
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "git": git_state(),
    }
    if device.type == "mps":
        report["mps_recommended_max_gb"] = round(torch.mps.recommended_max_memory() / 2**30, 2)
    return report
