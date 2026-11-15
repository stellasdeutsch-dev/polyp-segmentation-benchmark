"""Plot published polyp-segmentation Dice from research/published_results.csv.

Cleveland dot plot: one row per test set, one marker per method, a hairline
spanning U-Net to the best model, and the gap printed at the right.
Writes light and dark variants (PNG + SVG) to research/figures/.

Usage:  python research/plot_published.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "published_results.csv"
OUT_DIR = HERE / "figures"

# Each method is shown with the numbers from its own paper; U-Net as reported by PraNet.
METHODS = [  # (method, source paper in the CSV, marker)
    ("U-Net", "PraNet paper", "o"),
    ("PraNet", "PraNet paper", "s"),
    ("Polyp-PVT", "Polyp-PVT paper", "D"),
]
ROWS = [  # (dataset, label) top to bottom
    ("Kvasir", "Kvasir"),
    ("CVC-ClinicDB", "CVC-ClinicDB"),
    ("CVC-ColonDB", "CVC-ColonDB"),
    ("ETIS-LaribPolypDB", "ETIS"),
    ("CVC-300", "CVC-300"),
]
GROUPS = [("Seen hospitals", 0, 1), ("Unseen hospitals", 2, 4)]  # (label, first row, last row)

# Validated categorical slots 1-3 (dataviz reference palette), light and dark steps.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series=["#2a78d6", "#eb6834", "#1baf7a"],
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series=["#3987e5", "#d95926", "#199e70"],
    ),
}


def pick_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("SF Pro Display", "SF Pro Text", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def load() -> dict[tuple[str, str], float]:
    wanted = {(m, src) for m, src, _ in METHODS}
    values = {}
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if r["protocol"] == "pranet" and (r["method"], r["reported_in"]) in wanted:
                values[(r["method"], r["dataset"])] = float(r["mdice"])
    return values


def plot(theme: str, values: dict[tuple[str, str], float]) -> None:
    t = THEMES[theme]
    plt.rcParams.update({"font.family": pick_font(), "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    # extra vertical space between the two groups
    y_of = {}
    y = 0.0
    for i, (ds, _) in enumerate(ROWS):
        if i == GROUPS[1][1]:
            y += 0.9
        y_of[ds] = -y
        y += 1.0

    for ds, label in ROWS:
        yy = y_of[ds]
        present = [(m, values[(m, ds)]) for m, _, _ in METHODS if (m, ds) in values]
        lo, hi = min(v for _, v in present), max(v for _, v in present)
        ax.plot([lo, hi], [yy, yy], color=t["axis"], lw=2, solid_capstyle="round", zorder=1)
        for (m, _, marker), color in zip(METHODS, t["series"], strict=True):
            if (m, ds) in values:
                ax.scatter(
                    values[(m, ds)],
                    yy,
                    s=70,
                    marker=marker,
                    color=color,
                    edgecolors=t["surface"],
                    linewidths=2,
                    zorder=3,
                )
        best = max(present, key=lambda p: p[1])[0]
        gap = hi - values[("U-Net", ds)]
        note = f"+{gap:.2f}" + ("" if best == "Polyp-PVT" else "*")
        ax.text(1.005, yy, note, va="center", ha="left", fontsize=10, color=t["ink2"])
        ax.text(0.285, yy, label, va="center", ha="right", fontsize=10.5, color=t["ink"])

    for glabel, first, _last in GROUPS:
        top = y_of[ROWS[first][0]] + 0.65
        ax.text(
            0.285, top, glabel, va="bottom", ha="right", fontsize=9.5, color=t["muted"], fontweight="medium"
        )

    ax.text(
        1.005,
        y_of[ROWS[0][0]] + 0.65,
        "Gap to U-Net",
        va="bottom",
        ha="left",
        fontsize=9.5,
        color=t["muted"],
        fontweight="medium",
    )

    ax.set_xlim(0.30, 1.0)
    ax.set_ylim(min(y_of.values()) - 0.6, max(y_of.values()) + 1.0)
    ax.set_yticks([])
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9.5, length=0, pad=6)
    ax.grid(axis="x", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.set_xlabel(
        "Mean Dice as published (PraNet-style, averaged over thresholds)",
        color=t["ink2"],
        fontsize=9.5,
        labelpad=8,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker=mk,
            ls="",
            markersize=8,
            color=c,
            markeredgecolor=t["surface"],
            markeredgewidth=1.5,
            label=m,
        )
        for (m, _, mk), c in zip(METHODS, t["series"], strict=True)
    ]
    leg = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.905),
        ncol=3,
        frameon=False,
        fontsize=10,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    for txt in leg.get_texts():
        txt.set_color(t["ink"])

    fig.text(
        0.02,
        0.965,
        "U-Net trails newer models by about 0.1 Dice on seen hospitals and up to 0.39 on unseen ones",
        fontsize=12.5,
        color=t["ink"],
        fontweight="semibold",
        va="top",
    )
    fig.text(
        0.02,
        0.02,
        "Sources: PraNet (MICCAI 2020) Tables 1-2; Polyp-PVT (CAAI AIR 2023) Tables IV-V. "
        "* vs PraNet: Polyp-PVT reports no CVC-300 result.",
        fontsize=8.5,
        color=t["muted"],
    )
    fig.subplots_adjust(left=0.2, right=0.88, top=0.8, bottom=0.2)

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"published_dice_{theme}.{ext}", facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    values = load()
    for theme in THEMES:
        plot(theme, values)
    print(f"wrote {sorted(p.name for p in OUT_DIR.glob('published_dice_*'))}")


if __name__ == "__main__":
    main()
