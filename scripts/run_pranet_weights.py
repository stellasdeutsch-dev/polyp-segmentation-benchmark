"""Run the released PraNet weights on the five test sets and score them (stage 3.3).

    make pranet-run                     # Apple GPU, then the CPU
    python scripts/run_pranet_weights.py --devices mps

Follows MyTest.py step by step: resize the image to 352 x 352 (Pillow bilinear, through
torchvision as the authors do), ImageNet normalisation, the last of PraNet's four outputs,
bilinear upsampling to the ground-truth size, sigmoid, stretch, and the 8-bit PNG levels scipy
would have saved. The levels are scored exactly as eval/main.m scores the PNG.

Three comparisons:
  * against PraNet's Tables 1-2 (decision D13, tolerance 0.005);
  * against the authors' released maps, image by image: how far our run's maps are from theirs;
  * Apple GPU against CPU: what the device changes.

Outputs: reports/stage3_pranet_weights.json and .md; maps in runs/pranet_released/<device>/
(git-ignored) for the stage 3 figures.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from seg import data, metrics
from seg.env import get_device
from seg.evaluate import score_saved_map, summarise
from seg.pranet import TEST_SIZE, load_pranet
from seg.stats import paired_bootstrap

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "reports" / "stage3_pranet_weights.json"
OUT_MD = REPO / "reports" / "stage3_pranet_weights.md"
RUNS = REPO / "runs" / "pranet_released"
AUTHOR_MAPS = REPO / "data" / "pranet_maps" / "PraNet"
PUBLISHED = REPO / "reports" / "stage3_published_maps.json"
TOLERANCE = 0.005
WARMUP = 5

# utils/dataloader.py:73-77, test_dataset
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((TEST_SIZE, TEST_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD),
    ]
)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def predict_levels(
    model: torch.nn.Module, image: Image.Image, gt_shape: tuple[int, int], device: torch.device
):
    """MyTest.py:28-37 for one image. Returns the 8-bit levels and the forward-pass time."""
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    sync(device)
    start = time.perf_counter()
    res = model(x)[3]  # res5, res4, res3, res2 = model(image); res = res2
    sync(device)
    seconds = time.perf_counter() - start
    res = F.interpolate(res, size=gt_shape, mode="bilinear", align_corners=False)
    prob = res.sigmoid().cpu().numpy().squeeze()
    return metrics.pranet_save_levels(prob), seconds


def run_device(device: torch.device, paper: dict) -> dict:
    model = load_pranet(device)
    warm = TRANSFORM(Image.new("RGB", (64, 64))).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(WARMUP):  # first passes compile kernels and allocate memory
            model(warm)
    sync(device)
    out = {"device": device.type, "sets": []}
    for test_set in data.TEST_SETS:
        rows, dice_c, iou_c, times, vs_authors = [], [], [], [], []
        save_dir = RUNS / device.type / test_set
        save_dir.mkdir(parents=True, exist_ok=True)
        for item in data.items_for_testset(test_set):
            gt = np.asarray(Image.open(item.mask).convert("L"))
            levels, seconds = predict_levels(model, Image.open(item.image), gt.shape, device)
            Image.fromarray(levels).save(save_dir / f"{item.stem}.png")
            row, d, i = score_saved_map(levels, gt, item.stem)
            authors = np.asarray(Image.open(AUTHOR_MAPS / test_set / f"{item.stem}.png"))
            diff = np.abs(levels.astype(np.int16) - authors.astype(np.int16))
            row["level_diff_mean"] = float(diff.mean())
            row["level_diff_max"] = int(diff.max())
            row["pixels_off_by_more_than_2"] = float((diff > 2).mean())
            rows.append(row)
            dice_c.append(d)
            iou_c.append(i)
            times.append(seconds)
            vs_authors.append(row["level_diff_mean"])
        entry = {"dataset": test_set, **summarise(rows, dice_c, iou_c)}
        entry["forward_ms_median"] = 1000 * float(np.median(times))
        entry["forward_ms_p90"] = 1000 * float(np.percentile(times, 90))
        entry["level_diff_mean"] = float(np.mean(vs_authors))
        entry["level_diff_max"] = max(r["level_diff_max"] for r in rows)
        entry["pixels_off_by_more_than_2"] = float(np.mean([r["pixels_off_by_more_than_2"] for r in rows]))
        ref = paper.get(test_set)
        if ref:
            entry["paper"] = ref["paper"]
            entry["authors_maps_mdice"] = ref["mdice"]
            entry["diff_paper"] = entry["mdice"] - ref["paper"]["mdice"]
            entry["diff_authors_maps"] = entry["mdice"] - ref["mdice"]
            entry["within_tolerance"] = abs(entry["diff_paper"]) <= TOLERANCE
            mine = np.array([r["mdice"] for r in rows])
            theirs = {r["image"]: r["mdice"] for r in ref["per_image"]}
            entry["paired_vs_authors_maps"] = paired_bootstrap(
                mine, np.array([theirs[r["image"]] for r in rows])
            )
        entry["per_image"] = rows
        out["sets"].append(entry)
        print(
            f"{device.type:4s} {test_set:18s} mDice {entry['mdice']:.4f} "
            f"(paper {entry.get('paper', {}).get('mdice', float('nan')):.3f}, "
            f"authors' maps {entry.get('authors_maps_mdice', float('nan')):.4f}) "
            f"Dice@0.5 {entry['dice']:.4f}  {entry['forward_ms_median']:.0f} ms  "
            f"map diff {entry['level_diff_mean']:.2f} levels"
        )
    return out


def device_gap(a: dict, b: dict) -> list[dict]:
    """Image-by-image difference between two devices' runs."""
    out = []
    for sa, sb in zip(a["sets"], b["sets"], strict=True):
        ma = np.array([r["mdice"] for r in sa["per_image"]])
        mb = np.array([r["mdice"] for r in sb["per_image"]])
        out.append({
            "dataset": sa["dataset"],
            "mdice_diff": float(ma.mean() - mb.mean()),
            "max_image_diff": float(np.abs(ma - mb).max()),
        })  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    args = parser.parse_args()
    published = json.loads(PUBLISHED.read_text())["results"]
    paper = {r["dataset"]: r for r in published if r["method"] == "PraNet"}
    runs = []
    for name in args.devices:
        device = get_device(name)  # raises if the device is not available
        runs.append(run_device(device, paper))
    report = {
        "tolerance": TOLERANCE,
        "machine": f"{platform.machine()}, {platform.platform()}",
        "torch": torch.__version__,
        "runs": runs,
        "device_gap": device_gap(runs[0], runs[1]) if len(runs) == 2 else None,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1) + "\n")
    write_markdown(report)
    print(f"wrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)}")


def write_markdown(report: dict) -> None:
    lines = [
        "# Released PraNet weights, run on the Mac and scored by our evaluation (stage 3.3)",
        "",
        "Generated by `scripts/run_pranet_weights.py`,",
        f"PyTorch {report['torch']} on {report['machine']}.",
        'mDice follows eval/main.m exactly. "Authors\' maps" is the same metric on the maps the authors',
        f"released (stage 3.2). A row passes when |ours - paper| <= {report['tolerance']} (decision D13).",
        "",
    ]
    for run in report["runs"]:
        lines += [
            f"## {run['device'].upper()}",
            "",
            "| Test set | mDice | Paper | Diff | Authors' maps | Dice at 0.5 | Boundary IoU | "
            "Map difference (levels) | Forward pass | Pass |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for e in run["sets"]:
            lines.append(
                f"| {e['dataset']} | {e['mdice']:.4f} | {e['paper']['mdice']:.3f} | {e['diff_paper']:+.4f} | "
                f"{e['authors_maps_mdice']:.4f} | {e['dice']:.4f} | {e['boundary_iou']:.4f} | "
                f"{e['level_diff_mean']:.2f} mean, {e['level_diff_max']} max | "
                f"{e['forward_ms_median']:.0f} ms | {'yes' if e['within_tolerance'] else 'no'} |"
            )
        lines.append("")
    if report["device_gap"]:
        lines += [
            "## Apple GPU against CPU",
            "",
            "| Test set | mDice difference | Largest per-image difference |",
            "|---|---|---|",
        ]
        for g in report["device_gap"]:
            row = [g["dataset"], f"{g['mdice_diff']:+.6f}", f"{g['max_image_diff']:.6f}"]
            lines.append("| " + " | ".join(row) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
