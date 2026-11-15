"""Print the environment and run three quick checks on the GPU backend.

    make check-env

1. Device: which backend is used and how much GPU memory macOS recommends.
2. Correctness: a matrix product on the GPU matches the CPU result.
3. Memory probe: allocating a known amount moves the peak counter by that amount.
"""

from __future__ import annotations

import json
import sys

import torch

from seg.env import environment_report, get_device, measure


def main() -> int:
    device = get_device()
    report = environment_report(device)
    print(json.dumps(report, indent=2))
    ok = True

    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    ref = a @ b
    got = (a.to(device) @ b.to(device)).cpu()
    max_err = (ref - got).abs().max().item()
    match = torch.allclose(ref, got, rtol=1e-3, atol=1e-3)
    verdict = "ok" if match else "MISMATCH"
    print(f"\n[correctness] {device.type} vs cpu matmul: max abs diff {max_err:.2e} -> {verdict}")
    ok &= match

    if device.type in ("mps", "cuda"):
        size_mb = 256
        with measure(device) as cost:
            x = torch.empty(size_mb * 2**20 // 4, device=device)
            del x
        within = abs(cost.peak_mb - size_mb) < 1
        verdict = "ok" if within else "OFF"
        print(f"[memory] allocated {size_mb} MB, peak counter read {cost.peak_mb:.1f} MB -> {verdict}")
        ok &= within
    else:
        print("[memory] no accelerator: peak memory is not measured on CPU")

    if device.type != "mps":
        print(
            "\nnote: MPS is not in use. On a Mac this usually means torch was installed for the wrong Python."
        )
    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
