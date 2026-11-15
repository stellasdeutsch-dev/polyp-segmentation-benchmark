"""Training loop shared by every model (decisions D3, D6, D7, D8, D14, D15).

    python -m seg.train --config configs/smp_unet_r34.yaml --set lr=3e-4 seed=1

One run = one model, one learning rate, one seed. It writes to runs/<name>/:
  config.yaml    the exact settings
  env.json       versions, device, git commit (D14)
  metrics.csv    one row per epoch: train loss, validation Dice / IoU / Boundary IoU, lr, time, memory
  best.pt        weights of the epoch with the best validation Dice at 0.5 (D3)
  last.pt        full training state for --set resume=true; deleted when the run finishes
  summary.json   best epoch, its validation scores, total and median epoch time, peak memory,
                 parameter count, and the epochs after which the run was resumed, if any

Validation is scored at the original mask resolution (D4). Test sets are never touched here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import typing
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from seg.data import (
    DEFAULT_NUM_WORKERS,
    PolypDataset,
    eval_transform,
    make_loader,
    split_items,
    train_transform,
)
from seg.env import environment_report, get_device, measure, seed_everything
from seg.losses import get_loss
from seg.metrics import boundary_iou, dice, iou
from seg.models import build_model, count_parameters

REPO = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    name: str = "debug"
    model: str = "smp_unet_r34"
    pretrained: bool = True
    size: int = 256  # D9
    batch_size: int = 16
    epochs: int = 50
    lr: float = 3e-4  # D7: picked per model from {1e-4, 3e-4, 1e-3}
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    loss: str = "dice_bce"  # D6
    augment: bool = True  # D10
    seed: int = 0
    num_workers: int = DEFAULT_NUM_WORKERS
    device: str | None = None  # None = best available
    max_minutes: float | None = None  # stop after the epoch that crosses this, resume later
    resume: bool = False
    overfit: int = 0  # >0: train and validate on the first N training images, no augmentation
    out_root: str = "runs"

    @classmethod
    def from_yaml(cls, path: Path | None, overrides: list[str] | None = None) -> TrainConfig:
        values = yaml.safe_load(Path(path).read_text()) if path else {}
        names = {f.name for f in fields(cls)}
        for item in overrides or []:
            key, _, raw = item.partition("=")
            if key not in names:
                raise ValueError(f"unknown setting {key!r}")
            values[key] = yaml.safe_load(raw)
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        return cls(**{k: _coerce(cls, k, v) for k, v in values.items()})

    @property
    def out_dir(self) -> Path:
        return REPO / self.out_root / self.name


def _coerce(cls, name: str, value):
    """Convert a YAML value to the field's declared type.

    PyYAML follows YAML 1.1, where `3e-4` (no decimal point) is a string, so `--set lr=3e-4` would
    otherwise reach the optimizer as text. Caught by tests/test_train.py in 2.5.
    """
    hint = typing.get_type_hints(cls)[name]
    allowed = typing.get_args(hint) or (hint,)
    if value is None:
        if type(None) in allowed:
            return None
        raise ValueError(f"{name} cannot be empty")
    for t in allowed:
        if t is bool:
            if isinstance(value, bool):
                return value
            raise ValueError(f"{name} must be true or false, got {value!r}")
        if t in (int, float, str):
            try:
                converted = t(value)
            except (TypeError, ValueError):
                continue
            if t is int and isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{name} must be a whole number, got {value!r}")
            return converted
    raise ValueError(f"cannot read {name}={value!r} as {hint}")


def warmup_cosine(total_steps: int, warmup_fraction: float):
    """Learning-rate multiplier per step: linear warmup to 1, then cosine decay to 0 (D7)."""
    warmup = max(1, int(round(total_steps * warmup_fraction)))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


@torch.inference_mode()
def evaluate(model, loader, device) -> dict:
    """Mean per-image Dice, IoU and Boundary IoU at threshold 0.5, at original resolution."""
    model.eval()
    scores = {"dice": [], "iou": [], "boundary_iou": []}
    for batch in loader:
        logits = model(batch["image"].to(device))
        for k, full in enumerate(batch["mask_full"]):
            prob = torch.sigmoid(
                F.interpolate(logits[k : k + 1], size=full.shape, mode="bilinear", align_corners=False)
            )
            pred, target = prob[:, 0], full.to(device).bool()[None]
            scores["dice"].append(dice(pred, target).item())
            scores["iou"].append(iou(pred, target).item())
            scores["boundary_iou"].append(boundary_iou(pred, target).item())
    return {k: sum(v) / len(v) for k, v in scores.items()}


def loader_seed(seed: int, start_epoch: int) -> int:
    """Seed for the training loader. A run that starts at epoch 0 uses the run's seed.

    A resumed run must not: the loader would replay the shuffles and augmentations of epochs
    1, 2, ... from the resume point on (found in the stage 4 project check). It gets a seed of
    its own, fixed by the run's seed and the epoch it resumes at.
    """
    return seed if start_epoch == 0 else seed + 1_000_003 * start_epoch


def build_data(cfg: TrainConfig, start_epoch: int = 0):
    train_items, val_items = split_items()
    if cfg.overfit:
        train_items = train_items[: cfg.overfit]
        val_items = train_items
    train_tf = train_transform(cfg.size) if cfg.augment and not cfg.overfit else eval_transform(cfg.size)
    train_ds = PolypDataset(train_items, train_tf)
    val_ds = PolypDataset(val_items, eval_transform(cfg.size))
    train_loader = make_loader(
        train_ds,
        cfg.batch_size,
        shuffle=True,
        seed=loader_seed(cfg.seed, start_epoch),
        num_workers=cfg.num_workers,
    )
    val_loader = make_loader(
        val_ds, cfg.batch_size, shuffle=False, seed=cfg.seed, num_workers=cfg.num_workers
    )
    return train_loader, val_loader


def run(cfg: TrainConfig) -> dict:
    device = get_device(cfg.device)
    seed_everything(cfg.seed)
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    (out / "env.json").write_text(json.dumps(environment_report(device), indent=2) + "\n")

    last_path, metrics_path = out / "last.pt", out / "metrics.csv"
    state = None
    if cfg.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
    start_epoch = state["epoch"] + 1 if state else 0
    train_loader, val_loader = build_data(cfg, start_epoch)
    model = build_model(cfg.model, pretrained=cfg.pretrained).to(device)
    params = count_parameters(model)
    loss_fn = get_loss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps, cfg.warmup_fraction))

    best, history, resumed_at = {"dice": -1.0}, [], []
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, history = state["best"], state["history"]
        resumed_at = [*state.get("resumed_at", []), start_epoch]  # epochs completed at each resume
        print(f"resumed {cfg.name} after epoch {start_epoch}")
        del state

    session_start = time.perf_counter()
    peak_mb = max([h["peak_mb"] or 0 for h in history], default=0)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, n = torch.zeros((), device=device), 0
        with measure(device) as cost:
            for batch in train_loader:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                loss = loss_fn(model(images), masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running += loss.detach() * len(images)  # no .item() here: it would sync every step
                n += len(images)
        with measure(device) as val_cost:
            val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1,  # counted from 1 everywhere a person reads it
            "train_loss": running.item() / n,
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "val_boundary_iou": val["boundary_iou"],
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": cost.seconds,
            "val_seconds": val_cost.seconds,
            "peak_mb": cost.peak_mb,
        }
        history.append(row)
        peak_mb = max(peak_mb, cost.peak_mb or 0)
        if val["dice"] > best["dice"]:
            best = {"epoch": epoch + 1, **val}
            torch.save(model.state_dict(), out / "best.pt")
        print(
            f"[{cfg.name}] epoch {epoch + 1}/{cfg.epochs}  loss {row['train_loss']:.4f}  "
            f"val dice {val['dice']:.4f}  iou {val['iou']:.4f}  biou {val['boundary_iou']:.4f}  "
            f"{cost.seconds:.0f}s+{val_cost.seconds:.0f}s  {cost.peak_mb or 0:.0f} MB",
            flush=True,
        )
        with metrics_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best": best,
                "history": history,
                "resumed_at": resumed_at,
            },
            last_path,
        )
        minutes = (time.perf_counter() - session_start) / 60
        if cfg.max_minutes is not None and minutes > cfg.max_minutes and epoch + 1 < cfg.epochs:
            print(f"[{cfg.name}] stopping after {minutes:.1f} min; continue with --set resume=true")
            return {"finished": False, "epoch": epoch + 1, "best": best}

    summary = {
        "finished": True,
        "best_epoch": best["epoch"],
        "val": {k: best[k] for k in ("dice", "iou", "boundary_iou")},
        "resumed_after_epochs": resumed_at,
        "median_train_seconds_per_epoch": sorted(h["train_seconds"] for h in history)[len(history) // 2],
        "train_minutes": sum(h["train_seconds"] for h in history) / 60,
        "val_minutes": sum(h["val_seconds"] for h in history) / 60,
        "peak_mb": peak_mb,
        "parameters": params,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    last_path.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model on the PraNet benchmark.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config, args.set)
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
