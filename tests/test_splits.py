"""Tests for seg.splits and seg.audit on synthetic data (no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from seg.audit import auc_low_is_positive, nearest_similarity
from seg.splits import (
    CLINICDB_SEQUENCE_STARTS,
    Split,
    clinicdb_sequence,
    list_pairs,
    make_validation_split,
    sort_key,
    source_of,
)


def test_source_of_tells_datasets_apart():
    assert source_of("123") == "CVC-ClinicDB"
    assert source_of("cju0qkwl35piu0993l0dewei2") == "Kvasir"


def test_sort_key_orders_frames_numerically():
    assert sorted(["10", "2", "abc", "1"], key=sort_key) == ["1", "2", "10", "abc"]


def test_list_pairs_requires_matching_masks(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (4, 4)).save(tmp_path / "images" / f"{name}.png")
    Image.new("L", (4, 4)).save(tmp_path / "masks" / "a.png")
    with pytest.raises(ValueError, match="unmatched"):
        list_pairs(tmp_path / "images", tmp_path / "masks")
    Image.new("L", (4, 4)).save(tmp_path / "masks" / "b.png")
    assert list_pairs(tmp_path / "images", tmp_path / "masks") == ["a", "b"]


def test_sequence_table_shape():
    assert len(CLINICDB_SEQUENCE_STARTS) == 29
    assert list(CLINICDB_SEQUENCE_STARTS) == sorted(CLINICDB_SEQUENCE_STARTS)


@pytest.mark.parametrize(
    ("frame", "sequence"),
    [(1, 1), (25, 1), (26, 2), (205, 10), (206, 11), (591, 28), (592, 29), (612, 29)],
)
def test_clinicdb_sequence_boundaries(frame, sequence):
    assert clinicdb_sequence(frame) == sequence


@pytest.mark.parametrize("frame", [0, 613, -5])
def test_clinicdb_sequence_rejects_out_of_range(frame):
    with pytest.raises(ValueError):
        clinicdb_sequence(frame)


def make_stems():
    kvasir = [f"k{i:03d}" for i in range(90)]
    clinic = [str(i) for i in range(1, 56)]  # 55 frames in 11 groups of 5
    group_of = {s: (int(s) - 1) // 5 for s in clinic}
    return kvasir + clinic, group_of


def test_split_is_deterministic_and_disjoint():
    stems, group_of = make_stems()
    a = make_validation_split(stems, group_of, fraction=0.2, seed=0)
    b = make_validation_split(stems, group_of, fraction=0.2, seed=0)
    c = make_validation_split(stems, group_of, fraction=0.2, seed=1)
    assert a == b
    assert a.val != c.val
    assert not set(a.train) & set(a.val)
    assert sorted(a.train + a.val, key=sort_key) == sorted(stems, key=sort_key)


def test_split_keeps_groups_whole_and_stratifies():
    stems, group_of = make_stems()
    split = make_validation_split(stems, group_of, fraction=0.2, seed=0)
    val_groups = {group_of[s] for s in split.val if s in group_of}
    train_groups = {group_of[s] for s in split.train if s in group_of}
    assert not val_groups & train_groups  # no group on both sides
    meta = split.meta["sources"]
    assert meta["Kvasir"]["val"] == 18  # 20% of 90, image by image
    assert meta["CVC-ClinicDB"]["unit"] == "sequence"
    assert meta["CVC-ClinicDB"]["val"] >= meta["CVC-ClinicDB"]["target"]


def test_split_takes_preferred_groups_first():
    stems, group_of = make_stems()
    split = make_validation_split(stems, group_of, fraction=0.2, seed=0, preferred_groups={3, 7})
    # target is 11 frames: groups 3 and 7 give 10, so exactly one more group is added
    chosen = split.meta["sources"]["CVC-ClinicDB"]["val_groups"]
    assert {3, 7} <= set(chosen)
    assert len(chosen) == 3


def test_split_rejects_partial_group_ids():
    stems, group_of = make_stems()
    group_of.pop("1")
    with pytest.raises(ValueError, match="group ids"):
        make_validation_split(stems, group_of)


def test_split_round_trips_through_json(tmp_path):
    stems, group_of = make_stems()
    split = make_validation_split(stems, group_of)
    split.to_json(tmp_path / "val.json")
    assert Split.from_json(tmp_path / "val.json") == split


def test_nearest_similarity_picks_the_closest_row():
    ref = np.eye(3, dtype=np.float32)
    query = np.array([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(nearest_similarity(query, ref), [0.8, 1.0])


def test_auc_low_is_positive():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_low_is_positive(scores, np.array([True, True, False, False])) == 1.0
    assert auc_low_is_positive(scores, np.array([False, False, True, True])) == 0.0
    assert auc_low_is_positive(np.array([0.5, 0.5]), np.array([True, False])) == 0.5
    with pytest.raises(ValueError):
        auc_low_is_positive(scores, np.zeros(4, dtype=bool))
