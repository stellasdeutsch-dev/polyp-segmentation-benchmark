"""Tests for seg.train: schedule, config parsing, and a short real run with stop-and-resume."""

from __future__ import annotations

import csv
import json

import pytest
import torch

from seg.data import DEFAULT_SPLIT
from seg.train import TrainConfig, run, warmup_cosine

HAS_DATA = DEFAULT_SPLIT.exists() and (DEFAULT_SPLIT.parents[1] / "TrainDataset").exists()


def test_warmup_then_cosine_to_zero():
    f = warmup_cosine(total_steps=100, warmup_fraction=0.1)
    assert f(0) == pytest.approx(0.1)  # warmup starts small
    assert f(9) == pytest.approx(1.0)  # reaches the full rate at the end of warmup
    assert f(10) == pytest.approx(1.0)
    assert f(55) == pytest.approx(0.5, abs=0.02)  # halfway through the cosine
    assert f(99) < 0.01  # almost zero at the end
    values = [f(s) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))  # never increases after warmup


def test_config_overrides_are_typed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nlr: 1.0e-3\n")
    cfg = TrainConfig.from_yaml(path, ["lr=3e-4", "epochs=7", "augment=false", "device=cpu"])
    assert (cfg.lr, cfg.epochs, cfg.augment, cfg.device) == (3e-4, 7, False, "cpu")


def test_config_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_yaml(None, ["learning_rate=0.1"])
    path = tmp_path / "c.yaml"
    path.write_text("nmae: typo\n")
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_yaml(path)


@pytest.mark.skipif(not HAS_DATA, reason="needs `make data`")
def test_short_run_stops_and_resumes(tmp_path):
    """Four images, 64 px, three epochs: stop after the first, resume, finish, and clean up."""
    base = dict(
        name="t",
        model="smp_unet_r34",
        pretrained=False,
        size=64,
        batch_size=2,
        epochs=3,
        lr=1e-3,
        overfit=4,
        num_workers=0,
        device="cpu",
        out_root=str(tmp_path),
    )
    first = run(TrainConfig(**base, max_minutes=0))
    assert first == {"finished": False, "epoch": 1, "best": first["best"]}
    out = tmp_path / "t"
    assert (out / "last.pt").exists() and (out / "best.pt").exists()

    summary = run(TrainConfig(**base, resume=True))
    assert summary["finished"] is True
    assert not (out / "last.pt").exists()  # removed once the run is complete
    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3]  # resumed, not restarted
    assert summary["resumed_after_epochs"] == [1]  # the interruption is on record
    assert json.loads((out / "summary.json").read_text())["parameters"]["total"] > 1e6
    assert "torch" in json.loads((out / "env.json").read_text())
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert any(k.startswith("encoder.") for k in state)


def test_config_rejects_wrong_types():
    with pytest.raises(ValueError, match="true or false"):
        TrainConfig.from_yaml(None, ["augment=maybe"])
    with pytest.raises(ValueError, match="whole number"):
        TrainConfig.from_yaml(None, ["epochs=2.5"])
    with pytest.raises(ValueError, match="cannot read"):
        TrainConfig.from_yaml(None, ["lr=fast"])
    assert TrainConfig.from_yaml(None, ["max_minutes=null"]).max_minutes is None


def test_resumed_run_gets_its_own_loader_seed():
    """A resumed run must not replay the shuffles and augmentations of the first epochs."""
    from seg.train import loader_seed

    assert loader_seed(0, 0) == 0 and loader_seed(2, 0) == 2
    seeds = {loader_seed(seed, epoch) for seed in (0, 1, 2) for epoch in (0, 12, 25)}
    assert len(seeds) == 9  # every (seed, resume point) pair differs
    order = [
        torch.randperm(20, generator=torch.Generator().manual_seed(loader_seed(0, e))).tolist()
        for e in (0, 12)
    ]
    assert order[0] != order[1]
