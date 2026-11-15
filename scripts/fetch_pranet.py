"""Fetch the PraNet reference: model code, released weights, and the authors' prediction maps.

    make pranet

PraNet's licence allows research and education use but says nothing about redistribution, so
its code is not copied into this repository. This script downloads the two model files at the
pinned commit into third_party/PraNet/ (ignored by git). Sources, all from the PraNet README
at commit 95f12bd:
  lib/PraNet_Res2Net.py, lib/Res2Net_v1b.py   model definition
  PraNet-19.pth (124.6 MB)                    released weights      -> checkpoints/pranet/
  pre-computed maps (61.6 MB)                 the authors' outputs   -> data/pranet_maps/
"""

from __future__ import annotations

import sys
from pathlib import Path

from seg.download import gdrive, http, record, safe_extract

REPO = Path(__file__).resolve().parents[1]
COMMIT = "95f12bd4e32ec954b5069654abae7de9473fd1f5"
CODE_FILES = ("lib/PraNet_Res2Net.py", "lib/Res2Net_v1b.py")
CODE_DIR = REPO / "third_party" / "PraNet"
WEIGHTS = REPO / "checkpoints" / "pranet" / "PraNet-19.pth"
WEIGHTS_ID = "1lJv8XVStsp3oNKZHaSr42tawdMOq6FLP"
MAPS_ZIP = REPO / "data" / "raw" / "PraNet-maps.zip"
MAPS_ID = "1CJ6CTUdenumgiKXieuKXFohefRJwyFPY"
MAPS_DIR = REPO / "data" / "pranet_maps"


def main() -> int:
    for rel in CODE_FILES:
        out = CODE_DIR / rel
        if not out.exists():
            http(f"https://raw.githubusercontent.com/DengPingFan/PraNet/{COMMIT}/{rel}", out)
        info = record(f"PraNet:{rel}", out, commit=COMMIT)
        print(f"{rel}: {info['bytes']} bytes, sha256 {info['sha256'][:12]}…")

    if not WEIGHTS.exists():
        print("downloading PraNet-19.pth (README says 124.6MB)")
        gdrive(WEIGHTS_ID, WEIGHTS)
    info = record("PraNet-19.pth", WEIGHTS, drive_id=WEIGHTS_ID)
    print(f"PraNet-19.pth: {info['bytes'] / 1e6:.1f} MB, sha256 {info['sha256'][:12]}…")

    if not MAPS_ZIP.exists():
        print("downloading pre-computed maps (README says 61.6MB)")
        gdrive(MAPS_ID, MAPS_ZIP)
    info = record("PraNet-maps", MAPS_ZIP, drive_id=MAPS_ID)
    n = safe_extract(MAPS_ZIP, MAPS_DIR)
    print(f"pre-computed maps: {info['bytes'] / 1e6:.1f} MB, {n} files -> {MAPS_DIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
