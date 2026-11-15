"""Measure what one training step of the from-scratch U-Net costs, and what makes it cheaper.

    make unet-timing

Stage 4.2. The first full epoch of the 64-1,024-channel U-Net took 396 s of training, eleven
times the 37 s of the ResNet-34 U-Net (decision D8), so a 50-epoch run would take about 6 hours.
This script times synthetic training steps (batch 16, 256 px, Dice + BCE, AdamW) for the variants
that could change that, next to the ResNet-34 U-Net as the anchor, and counts each model's
floating-point operations. Writes reports/stage4_timing.json.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from seg.env import get_device, measure
from seg.losses import get_loss
from seg.models import build_model, count_parameters
from seg.unet import UNet

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "stage4_timing.json"
BATCH, SIZE, WARMUP, STEPS, TRAIN_IMAGES = 16, 256, 3, 12, 1292

VARIANTS = [  # name, model factory, memory format, autocast dtype
    ("U-Net 64-1024", lambda: UNet(), torch.contiguous_format, None),
    ("U-Net 64-1024, channels-last", lambda: UNet(), torch.channels_last, None),
    ("U-Net 64-1024, float16 autocast", lambda: UNet(), torch.contiguous_format, torch.float16),
    ("U-Net 64-1024, bfloat16 autocast", lambda: UNet(), torch.contiguous_format, torch.bfloat16),
    ("U-Net 32-512 (nnU-Net widths)", lambda: UNet(widths=(32, 64, 128, 256, 512)),
     torch.contiguous_format, None),
    ("ResNet-34 U-Net (rung 2, anchor)", lambda: build_model("smp_unet_r34", pretrained=False),
     torch.contiguous_format, None),
]  # fmt: skip


def gflops_per_image(model: torch.nn.Module) -> float:
    """Forward-pass FLOPs for one 256 px image (a multiply-add counts as two), on the CPU."""
    model = model.cpu().eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 3, SIZE, SIZE))
    return counter.get_total_flops() / 1e9


def time_steps(model, memory_format, dtype, device) -> dict:
    model = model.to(device).to(memory_format=memory_format).train()
    loss_fn = get_loss("dice_bce")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(BATCH, 3, SIZE, SIZE, generator=g).to(device).to(memory_format=memory_format)
    y = (torch.rand(BATCH, 1, SIZE, SIZE, generator=g) > 0.7).float().to(device)
    autocast = torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()

    def step():
        with autocast:
            loss = loss_fn(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    for _ in range(WARMUP):
        step()
    torch.mps.synchronize() if device.type == "mps" else None
    with measure(device) as cost:
        start = time.perf_counter()
        for _ in range(STEPS):
            loss = step()
        loss.item()  # waits for the GPU
        seconds = time.perf_counter() - start
    images_per_s = STEPS * BATCH / seconds
    return {
        "images_per_second": images_per_s,
        "seconds_per_epoch": TRAIN_IMAGES / images_per_s,
        "peak_mb": cost.peak_mb,
        "finite_loss": bool(torch.isfinite(loss).item()),
    }


def main() -> None:
    device = get_device()
    rows = []
    for name, make, fmt, dtype in VARIANTS:
        torch.manual_seed(0)
        model = make()
        row = {"variant": name, "parameters": count_parameters(model)["total"]}
        row["gflops_per_image"] = gflops_per_image(model)
        row.update(time_steps(model, fmt, dtype, device))
        rows.append(row)
        print(
            f"{name:36s} {row['parameters'] / 1e6:5.1f}M  {row['gflops_per_image']:6.1f} GFLOP  "
            f"{row['images_per_second']:5.1f} img/s  {row['seconds_per_epoch']:5.0f} s/epoch  "
            f"{(row['peak_mb'] or 0) / 1024:4.1f} GB",
            flush=True,
        )
        del model
        torch.mps.empty_cache() if device.type == "mps" else None
    OUT.write_text(json.dumps({"device": device.type, "batch": BATCH, "size": SIZE, "steps": STEPS,
                               "variants": rows}, indent=2) + "\n")  # fmt: skip
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
