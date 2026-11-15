"""Run one model through decision D7's protocol: a learning-rate search on seed 0, then seeds 1-2.

    python scripts/run_queue.py --config configs/unet.yaml            # all of it
    python scripts/run_queue.py --config configs/unet.yaml --phase lr  # only the search

Run names are <config name>_lr<lr>_s<seed>, under runs/. A finished run (summary.json present)
is skipped and an interrupted one resumes from last.pt, so the queue can be stopped and started
again at any time. The best learning rate is the one with the highest best validation Dice at 0.5
on seed 0 (D3, D7). Test sets are never touched here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
LRS = ("1e-4", "3e-4", "1e-3")


def run_name(base: str, lr: str, seed: int) -> str:
    return f"{base}_lr{lr}_s{seed}"


def summary(name: str) -> dict | None:
    path = REPO / "runs" / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def train(config: Path, name: str, lr: str, seed: int, extra: list[str]) -> bool:
    """Train or resume one run. Returns False if it was already finished."""
    if summary(name):
        print(f"{name}: finished, skipped", flush=True)
        return False
    resume = (REPO / "runs" / name / "last.pt").exists()
    settings = [f"name={name}", f"lr={lr}", f"seed={seed}", f"resume={'true' if resume else 'false'}"]
    cmd = [sys.executable, "-m", "seg.train", "--config", str(config), "--set", *settings, *extra]
    print(f"{name}: {'resuming' if resume else 'starting'}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)
    return True


def best_lr(base: str) -> str:
    scores = {lr: summary(run_name(base, lr, 0)) for lr in LRS}
    missing = [lr for lr, s in scores.items() if s is None]
    if missing:
        raise SystemExit(f"learning-rate search not finished: {missing}")
    return max(LRS, key=lambda lr: scores[lr]["val"]["dice"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["lr", "seeds", "all"], default="all")
    parser.add_argument("--rest-minutes", type=float, default=0.0, help="pause between runs, to cool down")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="passed on to seg.train")
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text())["name"]
    queue: list[tuple[str, int]] = []
    if args.phase in ("lr", "all"):
        queue += [(lr, 0) for lr in LRS]
    trained = False  # rest only between two runs that actually trained
    for lr, seed in queue:
        if trained and args.rest_minutes and not summary(run_name(base, lr, seed)):
            time.sleep(args.rest_minutes * 60)
        trained = train(args.config, run_name(base, lr, seed), lr, seed, args.set) or trained
    if args.phase in ("seeds", "all"):
        lr = best_lr(base)
        print(f"best learning rate on seed 0: {lr}", flush=True)
        for seed in (1, 2):
            if trained and args.rest_minutes and not summary(run_name(base, lr, seed)):
                time.sleep(args.rest_minutes * 60)
            trained = train(args.config, run_name(base, lr, seed), lr, seed, args.set) or trained
    return 0


if __name__ == "__main__":
    sys.exit(main())
