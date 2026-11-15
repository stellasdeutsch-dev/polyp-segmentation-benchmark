"""Draw stage 3's qualitative figure: our PraNet maps next to the authors' maps.

    make stage3-figures

Writes reports/figures/stage3_maps.png. One row per test set, on the image whose mDice is the
set's median, plus the Kvasir image where our map differs most from the authors'. Columns: the
image with the true outline, our map (Apple GPU), the authors' released map, and the difference
in 8-bit levels. Needs `make pranet-run` first (the maps in runs/ are not published).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from seg import data

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports" / "stage3_pranet_weights.json"
OURS = REPO / "runs" / "pranet_released" / "mps"
AUTHORS = REPO / "data" / "pranet_maps" / "PraNet"
OUT = REPO / "reports" / "figures" / "stage3_maps.png"
DIFF_MAX = 64  # levels; larger differences are drawn at full colour
SHORT = {"ETIS-LaribPolypDB": "ETIS"}
# one sequential hue, white (no difference) to dark blue, as in the site's charts
DIFF_CMAP = LinearSegmentedColormap.from_list("diff", ["#ffffff", "#9ec5f4", "#3987e5", "#184f95"])


def outline(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = (mask > 128).astype(np.uint8)
    width = max(2, round(0.006 * max(m.shape)))
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((width, width), np.uint8)) > 0
    out = rgb.copy()
    out[edge] = (255, 214, 0)
    return out


def pick_rows(report: dict) -> list[tuple[str, str, str]]:
    run = next(r for r in report["runs"] if r["device"] == "mps")
    rows = []
    for s in run["sets"]:
        per = sorted(s["per_image"], key=lambda r: r["mdice"])
        median = per[len(per) // 2]
        rows.append((s["dataset"], median["image"], "median mDice"))
    kvasir = next(s for s in run["sets"] if s["dataset"] == "Kvasir")
    worst = max(kvasir["per_image"], key=lambda r: r["level_diff_mean"])
    rows.append(("Kvasir", worst["image"], "largest difference"))
    return rows


def main() -> None:
    report = json.loads(REPORT.read_text())
    rows = pick_rows(report)
    items = {(t, i.stem): i for t in data.TEST_SETS for i in data.items_for_testset(t)}
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.4, 1.75 * len(rows)), dpi=200)
    for r, (test_set, stem, why) in enumerate(rows):
        item = items[(test_set, stem)]
        gt = np.asarray(Image.open(item.mask).convert("L"))
        rgb = np.asarray(Image.open(item.image).convert("RGB").resize(gt.shape[::-1]))
        ours = np.asarray(Image.open(OURS / test_set / f"{stem}.png"))
        authors = np.asarray(Image.open(AUTHORS / test_set / f"{stem}.png"))
        diff = np.abs(ours.astype(np.int16) - authors.astype(np.int16))
        panels = [
            (outline(rgb, gt), {}),
            (ours, {"cmap": "gray", "vmin": 0, "vmax": 255}),
            (authors, {"cmap": "gray", "vmin": 0, "vmax": 255}),
            (diff, {"cmap": DIFF_CMAP, "vmin": 0, "vmax": DIFF_MAX}),
        ]
        for c, (img, kw) in enumerate(panels):
            axes[r, c].imshow(img, interpolation="antialiased", **kw)
        name = SHORT.get(test_set, test_set)
        axes[r, 0].set_ylabel(f"{name}\n{why}", fontsize=7, color="#1d1d1f")
        axes[r, 3].text(
            1.04, 0.5, f"mean {diff.mean():.2f}\nmax {diff.max()}", transform=axes[r, 3].transAxes,
            fontsize=6.5, va="center", ha="left", color="#424245",
        )  # fmt: skip
    titles = ["Image, true outline", "Our map (Apple GPU)", "Authors' map"]
    titles.append(f"Difference, 0-{DIFF_MAX}+ levels")
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=7.5, color="#1d1d1f")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#d2d2d7")
            side.set_linewidth(0.6)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5)
    fig.subplots_adjust(right=0.9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT.relative_to(REPO)}")
    for test_set, stem, why in rows:
        print(f"  {test_set}: {stem} ({why})")


if __name__ == "__main__":
    main()
