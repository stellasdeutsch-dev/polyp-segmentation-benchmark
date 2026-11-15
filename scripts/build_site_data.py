"""Build site/data/journal.json, the data behind the Journal page.

    make site-data

Merges four sources, so nothing on the page is typed by hand:
  research/journal.yaml        what happened in each iteration, and every problem found
  pytest --collect-only        the current size of the test suite, per file
  real forward passes          tensor shapes and parameter counts of the models built so far
  reports/*.json               results of the evaluation scripts (stage 3 onwards)
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "data" / "journal.json"
KINDS = ("ours", "published", "library", "data")
STAGES = {
    1: "Study of real projects",
    2: "Skeleton and data",
    3: "Reproduce the reference",
    4: "U-Net from scratch",
    5: "U-Net with a pretrained encoder",
    6: "Frozen foundation model",
    7: "Experiments",
    8: "Demo, report, publication",
    9: "Server stage",
}


def load_journal() -> list[dict]:
    items = yaml.safe_load((REPO / "research" / "journal.yaml").read_text())["iterations"]
    for it in items:
        for key in ("id", "stage", "date", "title", "summary", "why", "results", "problems", "files"):
            if key not in it:
                raise ValueError(f"iteration {it.get('id')} is missing {key!r}")
        for p in it["problems"]:
            if p["kind"] not in KINDS:
                raise ValueError(f"iteration {it['id']}: unknown problem kind {p['kind']!r}")
        it["date"] = it["date"].isoformat() if isinstance(it["date"], date) else str(it["date"])
        for f in it["files"]:
            if not (REPO / f).exists():
                raise ValueError(f"iteration {it['id']}: file {f} does not exist")
    return items


def test_counts() -> dict:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    per_file = {}
    for line in out.splitlines():
        if line.startswith("tests/") and ": " in line:
            name, n = line.rsplit(": ", 1)
            per_file[name.removeprefix("tests/").removesuffix(".py")] = int(n)
    return {"total": sum(per_file.values()), "per_file": per_file}


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def shape(t: torch.Tensor) -> list[int]:
    return list(t.shape[1:])


@torch.inference_mode()
def smp_unet_graph() -> dict:
    from seg.models import build_model

    model = build_model("smp_unet_r34", pretrained=False).eval()
    x = torch.zeros(1, 3, 256, 256)
    feats = model.encoder(x)
    names = ["input", "stem", "layer1", "layer2", "layer3", "layer4"]
    encoder = [{"name": n, "shape": shape(f)} for n, f in zip(names, feats, strict=False)]
    seen: list[list[int]] = []  # decoder block outputs, recorded during the real forward pass
    hooks = [b.register_forward_hook(lambda m, i, o: seen.append(shape(o))) for b in model.decoder.blocks]
    dec_out = {}
    hooks.append(model.decoder.register_forward_hook(lambda m, i, o: dec_out.__setitem__("x", shape(o))))
    out = model(x)
    for h in hooks:
        h.remove()
    decoder = [{"name": f"up {i + 1}", "shape": s} for i, s in enumerate(seen)]
    dec = torch.zeros([1, *dec_out["x"]])
    return {
        "id": "smp_unet_r34",
        "rung": "2",
        "title": "U-Net with a ResNet-34 encoder",
        "status": "built",
        "input": [3, 256, 256],
        "encoder": encoder[1:],
        "decoder": decoder,
        "output": shape(out),
        "decoder_output": shape(dec),
        "params": {
            "total": n_params(model),
            "encoder": n_params(model.encoder),
            "decoder": n_params(model.decoder),
            "head": n_params(model.segmentation_head),
        },
        "note": "Measured by a forward pass of a 256 x 256 input. ImageNet-pretrained encoder (stage 5).",
    }


@torch.inference_mode()
def pranet_graph() -> dict | None:
    try:
        from seg.pranet import load_pranet
    except ImportError:
        return None
    try:
        model = load_pranet(torch.device("cpu"))
    except FileNotFoundError:
        return None
    seen = {}
    hooks = []
    for name in (
        "resnet.layer1",
        "resnet.layer2",
        "resnet.layer3",
        "resnet.layer4",
        "rfb2_1",
        "rfb3_1",
        "rfb4_1",
        "agg1",
    ):
        mod = dict(model.named_modules())[name]
        hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: seen.__setitem__(n, shape(o))))
    outs = model(torch.zeros(1, 3, 352, 352))
    for h in hooks:
        h.remove()
    backbone = n_params(model.resnet)
    return {
        "id": "pranet",
        "rung": "reference",
        "title": "PraNet (released weights)",
        "status": "built",
        "input": [3, 352, 352],
        "encoder": [
            {"name": n.split(".")[-1], "shape": seen[n]}
            for n in ("resnet.layer1", "resnet.layer2", "resnet.layer3", "resnet.layer4")
        ],
        "decoder": [
            {"name": "RFB on layer2-4", "shape": seen["rfb2_1"]},
            {"name": "partial decoder", "shape": seen["agg1"]},
            {"name": "reverse attention x3", "shape": shape(outs[3])},
        ],
        "output": shape(outs[3]),
        "params": {
            "total": n_params(model),
            "encoder": backbone,
            "decoder": n_params(model) - backbone,
            "head": 0,
        },
        "note": (
            "Measured by a forward pass of a 352 x 352 input with the released weights. Res2Net-50 backbone."
        ),
    }


@torch.inference_mode()
def unet_graph() -> dict:
    """Rung 1, measured: shapes from a forward pass of the model the registry builds."""
    from seg.models import build_model

    model = build_model("unet").eval()
    seen: list[list[int]] = []

    def record(m, i, o):
        seen.append(shape(o))

    hooks = [m.register_forward_hook(record) for m in (model.inc, *model.downs, *model.ups)]
    out = model(torch.zeros(1, 3, 256, 256))
    for h in hooks:
        h.remove()
    n_enc = 1 + len(model.downs)
    encoder_params = n_params(model.inc) + n_params(model.downs)
    return {
        "id": "unet_scratch",
        "rung": "1",
        "title": "U-Net from scratch",
        "status": "built",
        "input": [3, 256, 256],
        "encoder": [
            {"name": f"down {i}" if i else "input conv", "shape": s} for i, s in enumerate(seen[:n_enc])
        ],
        "decoder": [{"name": f"up {i + 1}", "shape": s} for i, s in enumerate(seen[n_enc:])],
        "output": shape(out),
        "params": {
            "total": n_params(model),
            "encoder": encoder_params,
            "decoder": n_params(model.ups),
            "head": n_params(model.head),
        },
        "note": "Measured by a forward pass of a 256 x 256 input. Same function as milesial.",
    }


def planned_graphs() -> list[dict]:
    """Rungs not built yet: their documented design, no measured numbers."""
    return [
        {
            "id": "dinov3_linear",
            "rung": "3",
            "title": "Linear head on frozen DINOv3 ViT-S/16",
            "status": "planned",
            "input": [3, 256, 256],
            "encoder": [{"name": "12 transformer blocks", "shape": [384, 16, 16]}],
            "decoder": [
                {"name": "BatchNorm + 1x1 conv", "shape": [1, 16, 16]},
                {"name": "bilinear upsampling", "shape": [1, 256, 256]},
            ],
            "output": [1, 256, 256],
            "params": None,
            "note": (
                "Shapes follow from patch size 16 and width 384 (DINOv3 README: ViT-S/16, 21M parameters). "
                "Measured in stage 6."
            ),
        },
    ]


def published_maps() -> dict | None:
    """Stage 3.2: our scores of the authors' maps next to the paper, without per-image rows."""
    path = REPO / "reports" / "stage3_published_maps.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    keep = ("method", "dataset", "n_images", "n_ground_truth", "mdice", "miou", "mdice_ci", "dice",
            "boundary_iou", "paper", "diff_mdice", "diff_miou", "within_tolerance", "all_files")  # fmt: skip
    return {
        "tolerance": data["tolerance"],
        "rows": [{k: r[k] for k in keep if k in r} for r in data["results"]],
    }


def pranet_weights() -> dict | None:
    """Stage 3.3: the released weights run on the Mac, per device and test set, without per-image rows."""
    path = REPO / "reports" / "stage3_pranet_weights.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    runs = []
    for run in data["runs"]:
        sets = []
        for e in run["sets"]:
            levels = [r["level_diff_max"] for r in e["per_image"]]
            sets.append({
                "dataset": e["dataset"],
                "n_images": e["n_images"],
                "mdice": e["mdice"],
                "paper": e["paper"]["mdice"],
                "authors_maps_mdice": e["authors_maps_mdice"],
                "diff_paper": e["diff_paper"],
                "forward_ms": e["forward_ms_median"],
                "maps_identical": sum(v == 0 for v in levels),
                "maps_within_one_level": sum(v == 1 for v in levels),
                "maps_differ": sum(v > 1 for v in levels),
            })  # fmt: skip
        runs.append({"device": run["device"], "sets": sets})
    return {"tolerance": data["tolerance"], "runs": runs, "device_gap": data["device_gap"]}


def stage3_page() -> dict | None:
    """Everything the stage 3 page draws, from the three stage 3 reports, in one small file."""
    names = ("published_maps", "pranet_weights", "kvasir_decoding")
    paths = {k: REPO / "reports" / f"stage3_{k}.json" for k in names}
    if not all(p.exists() for p in paths.values()):
        return None
    pub, run, dec = (json.loads(p.read_text()) for p in paths.values())
    by_device = {r["device"]: {s["dataset"]: s for s in r["sets"]} for r in run["runs"]}
    pranet_pub = {r["dataset"]: r for r in pub["results"] if r["method"] == "PraNet"}
    sets = []
    for name, mps in by_device["mps"].items():
        cpu = by_device["cpu"][name]
        levels = [r["level_diff_max"] for r in mps["per_image"]]
        sets.append({
            "dataset": name,
            "n_images": mps["n_images"],
            "paper": mps["paper"]["mdice"],
            "authors_maps": pranet_pub[name]["mdice"],
            "ours": mps["mdice"],
            "ours_ci": mps["mdice_ci"],
            "dice": mps["dice"],
            "boundary_iou": mps["boundary_iou"],
            "maps_within_one_level": sum(v <= 1 for v in levels),
            "forward_ms": {"mps": mps["forward_ms_median"], "cpu": cpu["forward_ms_median"]},
            "device_gap_max": next(g["max_image_diff"] for g in run["device_gap"] if g["dataset"] == name),
        })  # fmt: skip
    authors_kvasir = {r["image"]: r["mdice"] for r in pranet_pub["Kvasir"]["per_image"]}
    kvasir = by_device["mps"]["Kvasir"]
    return {
        "tolerance": run["tolerance"],
        "published_rows": len(pub["results"]),
        "published_within": sum(r["within_tolerance"] for r in pub["results"]),
        "sets": sets,
        "kvasir": {
            "paired": kvasir["paired_vs_authors_maps"],
            "png_equals_jpeg": dec["benchmark_png_equals_pillow_jpeg"],
            "images": [
                {"image": r["image"], "authors": round(authors_kvasir[r["image"]], 5),
                 "pillow": round(r["pillow_mdice"], 5), "apple": round(r["apple_mdice"], 5)}
                for r in dec["per_image"]
            ],
        },
    }  # fmt: skip


def stage4_page() -> dict:
    """What the stage 4 page draws: the U-Net's measured shapes, the cost benchmark, and every
    planned training run with its validation curve so far (runs/ itself is never published)."""
    import csv

    timing_path = REPO / "reports" / "stage4_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    runs = []
    plan = [("1e-4", 0), ("3e-4", 0), ("1e-3", 0), (None, 1), (None, 2)]
    search_done = all((REPO / "runs" / f"unet_lr{lr}_s0" / "summary.json").exists() for lr, _ in plan[:3])
    best_lr = None
    if search_done:
        scores = {
            lr: json.loads((REPO / "runs" / f"unet_lr{lr}_s0" / "summary.json").read_text())
            for lr, _ in plan[:3]
        }
        best_lr = max(scores, key=lambda k: scores[k]["val"]["dice"])
    for lr, seed in plan:
        lr = lr or best_lr
        name = f"unet_lr{lr}_s{seed}" if lr else None
        run_dir = REPO / "runs" / name if name else None
        curve = []
        if run_dir and (run_dir / "metrics.csv").exists():
            with (run_dir / "metrics.csv").open() as f:
                curve = [
                    {
                        "epoch": int(r["epoch"]),
                        "dice": round(float(r["val_dice"]), 4),
                        "biou": round(float(r["val_boundary_iou"]), 4),
                        "seconds": round(float(r["train_seconds"])),
                    }
                    for r in csv.DictReader(f)
                ]
        finished = bool(run_dir and (run_dir / "summary.json").exists())
        status = "done" if finished else "running" if curve else "waiting"
        runs.append({"lr": lr, "seed": seed, "status": status, "epochs": 50, "curve": curve})
    return {"generated": date.today().isoformat(), "model": unet_graph(), "timing": timing, "runs": runs,
            "best_lr": best_lr}  # fmt: skip


def main() -> None:
    items = load_journal()
    tests = test_counts()
    by_stage: dict[int, Counter] = {}
    for it in items:
        c = by_stage.setdefault(it["stage"], Counter())
        for p in it["problems"]:
            c[p["kind"]] += 1
    timeline = [{"id": it["id"], "tests": it["tests"]} for it in items if "tests" in it]
    if timeline and timeline[-1]["tests"] != tests["total"]:
        print(
            f"note: journal says {timeline[-1]['tests']} tests at {timeline[-1]['id']}, "
            f"pytest collects {tests['total']} now"
        )
    graphs = [g for g in (unet_graph(), smp_unet_graph(), pranet_graph()) if g] + planned_graphs()
    stages_done = sorted({it["stage"] for it in items})
    data = {
        "generated": date.today().isoformat(),
        "stages": [
            {"stage": s, "title": t, "iterations": sum(1 for it in items if it["stage"] == s)}
            for s, t in STAGES.items()
        ],
        "stages_started": stages_done,
        "iterations": items,
        "problems_by_stage": [
            {"stage": s, **{k: by_stage.get(s, Counter())[k] for k in KINDS}} for s in stages_done
        ],
        "problem_totals": {k: sum(by_stage[s][k] for s in by_stage) for k in KINDS},
        "tests": tests,
        "tests_timeline": timeline,
        "models": graphs,
        "published_maps": published_maps(),
        "pranet_weights": pranet_weights(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    (OUT.parent / "stage4.json").write_text(json.dumps(stage4_page(), indent=1) + "\n")
    stage3 = stage3_page()
    if stage3:
        (OUT.parent / "stage3.json").write_text(json.dumps(stage3, indent=1) + "\n")
    n_problems = sum(data["problem_totals"].values())
    print(
        f"wrote {OUT.relative_to(REPO)}: {len(items)} iterations, {n_problems} problems, "
        f"{tests['total']} tests, {len(graphs)} models"
    )


if __name__ == "__main__":
    main()
