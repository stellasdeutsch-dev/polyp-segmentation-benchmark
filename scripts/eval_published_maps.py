"""Score the authors' published prediction maps with our evaluation (stage 3.2).

    make published-maps

PraNet's authors released the maps behind their Tables 1-2 for PraNet, U-Net, U-Net++ and SFA.
Scoring the same files the authors scored isolates the evaluation code: if our PraNet-style mDice
matches the published table within 0.005 (decision D13), our ruler measures what theirs measures.

For every method and test set this writes:
  * mDice / mIoU exactly as eval/main.m computes them, on exactly the files main.m reads
    (it lists only *.png and resizes maps of the wrong size, bicubic);
  * the same with our integer thresholds, to measure what MATLAB's float thresholds change;
  * our main metrics on the saved maps: Dice, IoU and Boundary IoU at 0.5;
  * per-image values, so the page can show intervals and spread.

Outputs: reports/stage3_published_maps.json, reports/stage3_published_maps.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from seg import metrics
from seg.evaluate import score_saved_map, summarise

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "data" / "pranet_maps"
TEST = REPO / "data" / "TestDataset"
OUT_JSON = REPO / "reports" / "stage3_published_maps.json"
OUT_MD = REPO / "reports" / "stage3_published_maps.md"
METHODS = {"PraNet": "PraNet", "UNet": "U-Net", "UNet++": "U-Net++", "SFA": "SFA"}
SETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
TOLERANCE = 0.005  # decision D13


def published() -> dict[tuple[str, str], dict]:
    """PraNet paper, Tables 1-2, from research/published_results.csv."""
    out = {}
    with (REPO / "research" / "published_results.csv").open() as f:
        for r in csv.DictReader(f):
            if r["reported_in"] == "PraNet paper":
                out[(r["method"], r["dataset"])] = {
                    "mdice": float(r["mdice"]),
                    "miou": float(r["miou"]) if r["miou"] else None,
                }
    return out


def read_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    return np.asarray(im if im.mode == "L" else im.convert("L"))


def load_map(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """The map at the ground-truth size. main.m:79-83 resizes with MATLAB's imresize, whose default
    is bicubic with a = -0.5, the same kernel as Pillow's BICUBIC."""
    im = Image.open(path)
    im = im if im.mode == "L" else im.convert("L")
    if im.size == size:
        return np.asarray(im), False
    return np.asarray(im.resize(size, Image.Resampling.BICUBIC)), True


def score_folder(folder: Path, test_set: str, include_other_formats: bool) -> dict:
    masks = TEST / test_set / "masks"
    files = sorted(folder.glob("*.png"))  # main.m:52 dir([resMapPath '*.png'])
    if include_other_formats:
        stems = {p.stem for p in files}
        files += sorted(p for p in folder.iterdir() if p.suffix in {".tif", ".jpg"} and p.stem not in stems)
    dice_c, iou_c, dice_int, rows = [], [], [], []
    resized = 0
    for path in files:
        gt = read_gray(masks / f"{path.stem}.png")
        levels, was_resized = load_map(path, (gt.shape[1], gt.shape[0]))
        resized += was_resized
        row, d, i = score_saved_map(levels, gt, path.stem)
        dn, _ = metrics.pranet_curves(levels.astype(np.float64) / 255.0, gt)  # integer thresholds
        rows.append(row)
        dice_c.append(d)
        iou_c.append(i)
        dice_int.append(dn)
    return {
        **summarise(rows, dice_c, iou_c),
        "n_resized": resized,
        "mdice_integer_thresholds": float(np.stack(dice_int).mean()),
        "per_image": rows,
    }


def main() -> None:
    paper = published()
    results = []
    for folder_name, method in METHODS.items():
        for test_set in SETS:
            folder = MAPS / folder_name / test_set
            r = score_folder(folder, test_set, include_other_formats=False)
            n_gt = len(list((TEST / test_set / "masks").glob("*.png")))
            entry = {"method": method, "dataset": test_set, "n_ground_truth": n_gt, **r}
            if r["n_images"] < n_gt:  # maps main.m never reads: score them too, separately
                full = score_folder(folder, test_set, include_other_formats=True)
                entry["all_files"] = {
                    k: full[k] for k in ("n_images", "mdice", "miou", "dice", "boundary_iou")
                }
            ref = paper.get((method, test_set))
            if ref:
                entry["paper"] = ref
                entry["diff_mdice"] = r["mdice"] - ref["mdice"]
                entry["diff_miou"] = r["miou"] - ref["miou"] if ref["miou"] is not None else None
                entry["within_tolerance"] = abs(entry["diff_mdice"]) <= TOLERANCE
            results.append(entry)
            paper_txt = f"paper {ref['mdice']:.3f} ({entry['diff_mdice']:+.4f})" if ref else "no paper value"
            print(
                f"{method:8s} {test_set:18s} {r['n_images']:3d}/{n_gt:3d} images  "
                f"mDice {r['mdice']:.4f}, {paper_txt}, Dice at 0.5 {r['dice']:.4f}"
            )
    OUT_JSON.write_text(json.dumps({"tolerance": TOLERANCE, "results": results}, indent=1) + "\n")
    write_markdown(results)
    print(f"wrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)}")


def fmt(x: float | None, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def write_markdown(results: list[dict]) -> None:
    header = [
        "Method", "Test set", "Images", "mDice ours", "mDice paper", "Diff",
        "mIoU ours", "mIoU paper", "Dice at 0.5", "Boundary IoU", "Pass",
    ]  # fmt: skip
    lines = [
        "# Published maps scored with our evaluation (stage 3.2)",
        "",
        "Generated by `scripts/eval_published_maps.py`. mDice and mIoU follow PraNet's eval/main.m",
        "exactly; the paper columns are PraNet's Tables 1-2.",
        f"A row passes when |ours - paper| <= {TOLERANCE} (decision D13).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for e in results:
        p = e.get("paper", {})
        cells = [
            e["method"],
            e["dataset"],
            f"{e['n_images']}/{e['n_ground_truth']}",
            fmt(e["mdice"], ".4f"),
            fmt(p.get("mdice"), ".3f"),
            fmt(e.get("diff_mdice"), "+.4f"),
            fmt(e["miou"], ".4f"),
            fmt(p.get("miou"), ".3f"),
            fmt(e["dice"], ".4f"),
            fmt(e["boundary_iou"], ".4f"),
            "yes" if e.get("within_tolerance") else "no",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Notes generated from the same run", ""]
    for e in results:
        if "all_files" in e:
            a = e["all_files"]
            lines.append(
                f"- {e['method']} on {e['dataset']}: main.m lists only *.png, so it scores "
                f"{e['n_images']} of {e['n_ground_truth']} maps (mDice {e['mdice']:.4f}). With all "
                f"{a['n_images']} maps, mDice is {a['mdice']:.4f} ({a['mdice'] - e['mdice']:+.4f})."
            )
    worst = max(abs(e["mdice_integer_thresholds"] - e["mdice"]) for e in results)
    lines.append(
        f"- MATLAB's float thresholds against exact 8-bit thresholds: at most {worst:.5f} mDice on any row."
    )
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
