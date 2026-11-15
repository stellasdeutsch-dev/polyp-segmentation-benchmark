"""Datasets and transforms for the PraNet polyp benchmark (decisions D1, D9, D10, D11).

Every model reads data through this module, so all of them see exactly the same images,
masks, resizing and augmentation.

Each sample is a dict:
  image      float32 (3, S, S), ImageNet-normalised
  mask       float32 (1, S, S), values exactly 0 or 1
  mask_full  uint8 (H, W), the original-resolution binary mask, for scoring at full size (D4)
  stem       file name without extension
  source     "Kvasir", "CVC-ClinicDB" or the test-set name
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from seg.splits import Split, list_pairs, source_of

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
TRAIN_IMAGES = DATA / "TrainDataset" / "image"  # the archive uses image/, the test sets use images/
TRAIN_MASKS = DATA / "TrainDataset" / "masks"
TEST_ROOT = DATA / "TestDataset"
DEFAULT_SPLIT = DATA / "splits" / "val_seed0.json"
TEST_SETS = ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300")

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # PraNet utils/dataloader.py:22-23, also DINOv3's input statistics
IMAGENET_STD = (0.229, 0.224, 0.225)

# Loader worker processes. Measured in 2.4 on the M3 Air (training transform, 256 px, batch 16):
# 0 workers 231 img/s, 2 -> 481, 4 -> 746, 6 -> 1040. Four keeps a model fed while leaving cores
# for the GPU work; checked against real training speed in 2.5.
DEFAULT_NUM_WORKERS = 4


def train_transform(size: int, seed: int | None = None) -> A.Compose:
    """Decision D10: one light, joint image-mask augmentation, identical for every model.

    Resize first so the augmentations work at training resolution (cheaper). Every geometric
    step moves image and mask together; masks always use nearest-neighbour interpolation, so
    they stay strictly binary (D11). Pixels revealed by rotation or scaling are filled with 0,
    which is background in the mask and black in the image, like the dark border of an endoscope frame.

    albumentations 2.x draws from its own generator: `np.random.seed` does not reach it, `seed`
    here does. Loader workers are reseeded individually by `worker_init_fn` (checked in 2.5).
    """
    return A.Compose(
        [
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.75, 1.25),
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
        seed=seed,
    )


def eval_transform(size: int) -> A.Compose:
    """Resize and normalise only. Deterministic."""
    return A.Compose(
        [
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    """Binary uint8 mask: 1 where the stored mask is above 127 (D11)."""
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8)


@dataclass(frozen=True)
class Item:
    image: Path
    mask: Path
    stem: str
    source: str


class PolypDataset(Dataset):
    """Image-mask pairs with a transform. `items` fixes the order, so indices are stable."""

    def __init__(self, items: Sequence[Item], transform: A.Compose):
        if not items:
            raise ValueError("empty dataset")
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        image, mask = load_image(item.image), load_mask(item.mask)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"{item.stem}: image {image.shape[:2]} and mask {mask.shape} differ in size")
        out = self.transform(image=image, mask=mask)
        return {
            "image": torch.from_numpy(out["image"].transpose(2, 0, 1).copy()).float(),
            "mask": torch.from_numpy(out["mask"][None].copy()).float(),
            "mask_full": torch.from_numpy(mask),
            "stem": item.stem,
            "source": item.source,
        }


def _png(folder: Path, stem: str) -> Path:
    return folder / f"{stem}.png"


def split_items(split_path: Path = DEFAULT_SPLIT) -> tuple[list[Item], list[Item]]:
    """Training and validation items from the saved split (D2)."""
    split = Split.from_json(split_path)

    def make(stems: list[str]) -> list[Item]:
        return [Item(_png(TRAIN_IMAGES, s), _png(TRAIN_MASKS, s), s, source_of(s)) for s in stems]

    return make(split.train), make(split.val)


def items_for_testset(name: str, root: Path = TEST_ROOT) -> list[Item]:
    """All items of one PraNet test set."""
    if name not in TEST_SETS:
        raise ValueError(f"unknown test set {name!r}; expected one of {TEST_SETS}")
    images, masks = root / name / "images", root / name / "masks"
    return [Item(_png(images, s), _png(masks, s), s, name) for s in list_pairs(images, masks)]


def collate(batch: list[dict]) -> dict:
    """Stack fixed-size tensors; keep full-resolution masks and names as lists (their sizes differ)."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "mask_full": [b["mask_full"] for b in batch],
        "stem": [b["stem"] for b in batch],
        "source": [b["source"] for b in batch],
    }


def worker_init_fn(worker_id: int) -> None:
    """Give each loader worker its own augmentation stream.

    Without this, every worker starts from the same albumentations seed and repeats the others'
    draws: with 2 workers only half the augmentations were distinct (measured in 2.5). PyTorch
    hands each worker a distinct seed derived from the loader's generator, so this stays reproducible.
    """
    info = torch.utils.data.get_worker_info()
    transform = getattr(info.dataset, "transform", None)
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(info.seed % 2**32)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
    drop_last: bool = False,
) -> DataLoader:
    """A seeded DataLoader: same seed, same order and same augmentations, any number of workers."""
    if num_workers == 0 and hasattr(getattr(dataset, "transform", None), "set_random_seed"):
        dataset.transform.set_random_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
        drop_last=drop_last,
        # Keep workers alive across epochs: on macOS each new worker re-imports torch and
        # albumentations, which cost ~20 s per epoch in 2.5. Each worker's augmentation stream
        # simply continues, so draws stay distinct and reproducible.
        persistent_workers=num_workers > 0,
    )
