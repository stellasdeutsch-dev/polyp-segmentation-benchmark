"""Why does released PraNet differ from the authors' maps only on Kvasir? (stage 3.3, macOS only)

    python scripts/check_kvasir_decoding.py

Kvasir is the only test set whose originals are JPEG; the four lossless sets reproduce the
authors' maps within one 8-bit level. This script tests two explanations:
  1. the benchmark's PNGs are not the original images: compares them with the original
     Kvasir-SEG JPEGs (downloaded from Simula, 46 MB) decoded by Pillow;
  2. JPEG decoding changes the result: decodes the same JPEGs with Apple's ImageIO (`sips`), runs
     PraNet on both decodings and compares mDice and the distance to the authors' maps.

Output: reports/stage3_kvasir_decoding.json
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from run_pranet_weights import predict_levels
from seg.download import http, record, safe_extract
from seg.env import get_device
from seg.evaluate import score_saved_map
from seg.pranet import load_pranet

REPO = Path(__file__).resolve().parents[1]
URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ZIP = REPO / "data" / "raw" / "kvasir-seg.zip"
ORIGINAL = REPO / "data" / "kvasir_seg_original"
TEST = REPO / "data" / "TestDataset" / "Kvasir"
AUTHORS = REPO / "data" / "pranet_maps" / "PraNet" / "Kvasir"
OUT = REPO / "reports" / "stage3_kvasir_decoding.json"


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def main() -> None:
    if not ZIP.exists():
        http(URL, ZIP)
    record("Kvasir-SEG original", ZIP, url=URL)
    if not ORIGINAL.exists():
        safe_extract(ZIP, ORIGINAL)
    jpgs = ORIGINAL / "Kvasir-SEG" / "images"
    stems = sorted(p.stem for p in (TEST / "images").glob("*.png"))

    pixel = [np.abs(rgb(TEST / "images" / f"{s}.png") - rgb(jpgs / f"{s}.jpg")) for s in stems]
    device = get_device()
    model = load_pranet(device)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for s in stems:
            apple = Path(tmp) / f"{s}.png"
            subprocess.run(["sips", "-s", "format", "png", str(jpgs / f"{s}.jpg"), "--out", str(apple)],
                           check=True, capture_output=True)  # fmt: skip
            gt = np.asarray(Image.open(TEST / "masks" / f"{s}.png").convert("L"))
            authors = np.asarray(Image.open(AUTHORS / f"{s}.png")).astype(np.int16)
            row = {
                "image": s,
                "apple_vs_pillow_pixels": float(np.abs(rgb(apple) - rgb(jpgs / f"{s}.jpg")).mean()),
            }
            for name, src in (("pillow", jpgs / f"{s}.jpg"), ("apple", apple)):
                levels, _ = predict_levels(model, Image.open(src), gt.shape, device)
                score, _, _ = score_saved_map(levels, gt, s)
                row[f"{name}_mdice"] = score["mdice"]
                row[f"{name}_level_diff_to_authors"] = float(np.abs(levels.astype(np.int16) - authors).mean())
            rows.append(row)

    gap = np.array([r["pillow_mdice"] - r["apple_mdice"] for r in rows])
    report = {
        "benchmark_png_equals_pillow_jpeg": all(int(p.max()) == 0 for p in pixel),
        "apple_vs_pillow_pixel_diff_mean": float(np.mean([r["apple_vs_pillow_pixels"] for r in rows])),
        "mdice_pillow": float(np.mean([r["pillow_mdice"] for r in rows])),
        "mdice_apple": float(np.mean([r["apple_mdice"] for r in rows])),
        "decoder_mdice_gap_mean": float(gap.mean()),
        "decoder_mdice_gap_max_image": float(np.abs(gap).max()),
        "level_diff_to_authors_pillow": float(np.mean([r["pillow_level_diff_to_authors"] for r in rows])),
        "level_diff_to_authors_apple": float(np.mean([r["apple_level_diff_to_authors"] for r in rows])),
        "per_image": rows,
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    for k, v in report.items():
        if k != "per_image":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
