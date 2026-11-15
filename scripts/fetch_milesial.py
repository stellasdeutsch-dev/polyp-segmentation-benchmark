"""Fetch milesial/Pytorch-UNet's model code, the reference our from-scratch U-Net is checked against.

    make milesial

Downloads the three files of its `unet` package at the commit read in stage 1 (1.2) into
third_party/milesial/ (ignored by git; the project is GPL-3.0 and is only run, never copied).
tests/test_unet.py loads it when present and compares it with seg.unet layer by layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

from seg.download import http, record

REPO = Path(__file__).resolve().parents[1]
COMMIT = "21d7850f2af30a9695bbeea75f3136aa538cfc4a"
FILES = ("unet/__init__.py", "unet/unet_parts.py", "unet/unet_model.py")
CODE_DIR = REPO / "third_party" / "milesial"


def main() -> int:
    for rel in FILES:
        out = CODE_DIR / rel
        if not out.exists():
            http(f"https://raw.githubusercontent.com/milesial/Pytorch-UNet/{COMMIT}/{rel}", out)
        info = record(f"milesial:{rel}", out, commit=COMMIT)
        print(f"{rel}: {info['bytes']} bytes, sha256 {info['sha256'][:12]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
