"""Tests for seg.data: transforms keep image and mask aligned, masks stay binary, shapes are right."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from seg.data import (
    DEFAULT_SPLIT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TEST_SETS,
    Item,
    PolypDataset,
    collate,
    eval_transform,
    items_for_testset,
    split_items,
    train_transform,
)

HAS_DATA = DEFAULT_SPLIT.exists() and (DEFAULT_SPLIT.parents[1] / "TrainDataset").exists()


def synthetic_pair(tmp_path, h=300, w=360, seed=0) -> Item:
    """Dark noisy image with a red ellipse; the mask is exactly that ellipse."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
    ry, rx = rng.uniform(0.12, 0.3) * h, rng.uniform(0.12, 0.3) * w
    inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
    img = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
    img[inside] = (230, 30, 30)
    Image.fromarray(img).save(tmp_path / f"img{seed}.png")
    Image.fromarray(inside.astype(np.uint8) * 255).save(tmp_path / f"mask{seed}.png")
    return Item(tmp_path / f"img{seed}.png", tmp_path / f"mask{seed}.png", f"s{seed}", "synthetic")


def denormalise(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return ((image * std + mean).clamp(0, 1) * 255).numpy().transpose(1, 2, 0)


def red_region(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r - (g + b) / 2) > 60


@pytest.mark.parametrize("seed", range(4))
def test_augmentation_keeps_image_and_mask_aligned(tmp_path, seed):
    """Over many random augmentations, the red ellipse in the image and the mask must coincide."""
    ds = PolypDataset([synthetic_pair(tmp_path, seed=seed)], train_transform(128, seed=seed))
    worst = 1.0
    for _ in range(40):  # each call draws a new augmentation from the transform's own generator
        sample = ds[0]
        mask = sample["mask"][0].numpy() > 0.5
        red = red_region(denormalise(sample["image"]))
        union = (mask | red).sum()
        agreement = (mask & red).sum() / union if union else 1.0
        worst = min(worst, agreement)
    # edges are interpolated in the image but not in the mask, so a thin band can disagree
    assert worst > 0.9, f"image and mask drifted apart (worst IoU {worst:.3f})"


def test_augmentation_actually_varies(tmp_path):
    ds = PolypDataset([synthetic_pair(tmp_path)], train_transform(96))
    masks = {ds[0]["mask"].numpy().tobytes() for _ in range(20)}
    assert len(masks) > 5


@pytest.mark.parametrize("transform", [train_transform, eval_transform])
def test_masks_stay_strictly_binary(tmp_path, transform):
    ds = PolypDataset([synthetic_pair(tmp_path)], transform(100))
    for _ in range(15):
        values = set(np.unique(ds[0]["mask"].numpy()).tolist())
        assert values <= {0.0, 1.0}


def test_sample_shapes_and_types(tmp_path):
    sample = PolypDataset([synthetic_pair(tmp_path, h=288, w=384)], eval_transform(256))[0]
    assert sample["image"].shape == (3, 256, 256) and sample["image"].dtype == torch.float32
    assert sample["mask"].shape == (1, 256, 256) and sample["mask"].dtype == torch.float32
    assert sample["mask_full"].shape == (288, 384) and sample["mask_full"].dtype == torch.uint8
    assert set(np.unique(sample["mask_full"].numpy()).tolist()) <= {0, 1}


def test_eval_transform_is_deterministic(tmp_path):
    ds = PolypDataset([synthetic_pair(tmp_path)], eval_transform(128))
    torch.testing.assert_close(ds[0]["image"], ds[0]["image"])


def test_normalisation_uses_imagenet_statistics(tmp_path):
    grey = np.full((64, 64, 3), 128, dtype=np.uint8)
    Image.fromarray(grey).save(tmp_path / "g.png")
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(tmp_path / "gm.png")
    sample = PolypDataset([Item(tmp_path / "g.png", tmp_path / "gm.png", "g", "x")], eval_transform(64))[0]
    expected = [(128 / 255 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)]
    np.testing.assert_allclose(sample["image"].mean(dim=(1, 2)).numpy(), expected, atol=1e-4)


def test_mismatched_sizes_are_rejected(tmp_path):
    Image.fromarray(np.zeros((40, 50, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((40, 40), dtype=np.uint8)).save(tmp_path / "am.png")
    ds = PolypDataset([Item(tmp_path / "a.png", tmp_path / "am.png", "a", "x")], eval_transform(32))
    with pytest.raises(ValueError, match="differ in size"):
        ds[0]


def test_collate_keeps_full_masks_as_list(tmp_path):
    items = [synthetic_pair(tmp_path, h=200 + 20 * i, w=240, seed=i) for i in range(3)]
    batch = collate([PolypDataset(items, eval_transform(64))[i] for i in range(3)])
    assert batch["image"].shape == (3, 3, 64, 64)
    assert batch["mask"].shape == (3, 1, 64, 64)
    assert [m.shape[0] for m in batch["mask_full"]] == [200, 220, 240]
    assert batch["stem"] == ["s0", "s1", "s2"]


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError):
        PolypDataset([], eval_transform(32))


@pytest.mark.skipif(not HAS_DATA, reason="needs `make data` and the committed split")
def test_real_split_and_test_sets_load():
    train, val = split_items()
    assert (len(train), len(val)) == (1292, 158)
    assert not {i.stem for i in train} & {i.stem for i in val}
    expected = {
        "Kvasir": 100,
        "CVC-ClinicDB": 62,
        "CVC-ColonDB": 380,
        "ETIS-LaribPolypDB": 196,
        "CVC-300": 60,
    }
    for name in TEST_SETS:
        assert len(items_for_testset(name)) == expected[name]
    sample = PolypDataset(train[:1], train_transform(256))[0]
    assert sample["image"].shape == (3, 256, 256)
    assert set(np.unique(sample["mask"].numpy()).tolist()) <= {0.0, 1.0}


def test_unknown_test_set_is_rejected():
    with pytest.raises(ValueError, match="unknown test set"):
        items_for_testset("NotADataset")


def test_alignment_check_catches_a_broken_pipeline(tmp_path):
    """Negative control: augmenting image and mask with separate random draws must be detected."""
    item = synthetic_pair(tmp_path)
    image = np.asarray(Image.open(item.image).convert("RGB"))
    mask = (np.asarray(Image.open(item.mask)) > 127).astype(np.uint8)
    t = train_transform(128, seed=0)
    scores = []
    for _ in range(40):
        img_out = t(image=image, mask=mask)["image"]  # one random draw for the image ...
        mask_out = t(image=image, mask=mask)["mask"]  # ... an independent one for the mask
        red = red_region(denormalise(torch.from_numpy(img_out.transpose(2, 0, 1).copy())))
        m = mask_out > 0
        scores.append((m & red).sum() / max((m | red).sum(), 1))
    assert min(scores) < 0.7  # the same check that passes for the real pipeline fails here


def _augmentation_fingerprints(items, workers: int, seed: int) -> list[int]:
    from seg.data import make_loader

    ds = PolypDataset(items, train_transform(64))
    loader = make_loader(ds, batch_size=1, shuffle=False, seed=seed, num_workers=workers)
    return [int(b["mask"].sum().item() * 1000 + b["image"].abs().sum().item()) for b in loader]


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_augmentations_are_distinct_and_reproducible(tmp_path, workers):
    """Same image eight times: every copy must get its own augmentation, and a rerun the same ones."""
    items = [synthetic_pair(tmp_path, seed=0)] * 8
    first = _augmentation_fingerprints(items, workers, seed=3)
    again = _augmentation_fingerprints(items, workers, seed=3)
    other = _augmentation_fingerprints(items, workers, seed=4)
    assert first == again  # reproducible
    assert len(set(first)) == 8  # workers do not repeat each other's draws
    assert first != other  # the seed matters
