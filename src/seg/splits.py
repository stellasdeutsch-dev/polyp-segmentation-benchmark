"""Dataset listing, CVC-ClinicDB video sequences, and the saved validation split (decision D2).

The PraNet training folder mixes two datasets:
  * Kvasir-SEG images have random alphanumeric names, e.g. "cju0qkwl35piu0993l0dewei2".
  * CVC-ClinicDB images are numbered frames "1" ... "612" taken from 29 colonoscopy videos.

Frames of the same video look alike. A validation split that puts frame 101 in training and
frame 102 in validation measures memory, not generalisation. So CVC-ClinicDB is split by whole
video sequences, using the frame-to-sequence table published with the dataset.
"""

from __future__ import annotations

import bisect
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# First frame of each of the 29 CVC-ClinicDB sequences. Source: "Table 1: Correspondence between
# number of frames and sequences in CVC-ClinicDB" on https://polyp.grand-challenge.org/CVCClinicDB/
# (an image there), transcribed on https://datasetninja.com/cvc-612. Sequence 29 ends at frame 612.
# Sub-step 2.2 checked it against the images: ResNet-18 similarity of consecutive frames separates
# these 28 boundaries from within-sequence pairs with AUC 0.965 (reports/data_inventory.md).
CLINICDB_SEQUENCE_STARTS = (
    1, 26, 51, 68, 79, 104, 127, 152, 178, 200, 206, 228, 253, 278, 298,
    318, 343, 364, 384, 409, 429, 448, 467, 479, 504, 529, 547, 572, 592,
)  # fmt: skip
CLINICDB_LAST_FRAME = 612


def clinicdb_sequence(frame: int) -> int:
    """Sequence number (1-29) of a CVC-ClinicDB frame number (1-612)."""
    if not 1 <= frame <= CLINICDB_LAST_FRAME:
        raise ValueError(f"CVC-ClinicDB frames are numbered 1-{CLINICDB_LAST_FRAME}, got {frame}")
    return bisect.bisect_right(CLINICDB_SEQUENCE_STARTS, frame)


def list_pairs(images_dir: Path, masks_dir: Path) -> list[str]:
    """Stems that have both an image and a mask, sorted. Raises if any side is unmatched."""
    images = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    masks = {p.stem for p in masks_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    only_img, only_mask = images - masks, masks - images
    if only_img or only_mask:
        raise ValueError(
            f"unmatched files: {len(only_img)} images without mask, {len(only_mask)} masks without image"
        )
    return sorted(images, key=sort_key)


def sort_key(stem: str) -> tuple[int, int | str]:
    """Numbered frames first in numeric order, then names alphabetically."""
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def source_of(stem: str) -> str:
    """Which dataset a training image comes from, judged by its file name."""
    return "CVC-ClinicDB" if stem.isdigit() else "Kvasir"


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    meta: dict

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta, "train": self.train, "val": self.val}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> Split:
        d = json.loads(path.read_text())
        return cls(train=d["train"], val=d["val"], meta=d["meta"])


def make_validation_split(
    stems: Iterable[str],
    group_of: dict[str, int],
    fraction: float = 0.10,
    seed: int = 0,
    preferred_groups: set[int] | None = None,
    source: Callable[[str], str] = source_of,
) -> Split:
    """Hold out about `fraction` of each source. Sources with group ids are split by whole groups.

    Kvasir images are independent, so they are sampled one by one. For a grouped source
    (CVC-ClinicDB sequences) the groups are shuffled with the seed, `preferred_groups` first,
    and taken whole until the target count is reached; the last group may overshoot.
    """
    stems = sorted(stems, key=sort_key)
    preferred = preferred_groups or set()
    rng = random.Random(seed)
    by_source: dict[str, list[str]] = {}
    for s in stems:
        by_source.setdefault(source(s), []).append(s)

    val: set[str] = set()
    meta: dict = {"seed": seed, "fraction": fraction, "sources": {}}
    for src, items in sorted(by_source.items()):
        target = round(len(items) * fraction)
        grouped = [s for s in items if s in group_of]
        if grouped and len(grouped) != len(items):
            raise ValueError(f"{src}: group ids given for some images but not all")
        if not grouped:
            picked = rng.sample(items, target)
            meta["sources"][src] = {
                "total": len(items),
                "target": target,
                "val": len(picked),
                "unit": "image",
            }
            val.update(picked)
            continue

        groups: dict[int, list[str]] = {}
        for s in items:
            groups.setdefault(group_of[s], []).append(s)
        first = sorted(g for g in groups if g in preferred)
        rest = sorted(g for g in groups if g not in preferred)
        rng.shuffle(first)
        rng.shuffle(rest)
        chosen, taken = [], 0
        for g in first + rest:
            if taken >= target:
                break
            chosen.append(g)
            taken += len(groups[g])
        picked = [s for g in chosen for s in groups[g]]
        meta["sources"][src] = {
            "total": len(items),
            "target": target,
            "val": len(picked),
            "unit": "sequence",
            "val_groups": sorted(chosen),
            "n_groups": len(groups),
        }
        val.update(picked)

    train = [s for s in stems if s not in val]
    return Split(train=train, val=sorted(val, key=sort_key), meta=meta)
