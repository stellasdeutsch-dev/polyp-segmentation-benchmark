"""Tests for loading the released PraNet model (skipped until `make pranet` has run)."""

from __future__ import annotations

import pytest
import torch

from seg.pranet import CODE_DIR, WEIGHTS, load_pranet, predict

HAS_PRANET = WEIGHTS.exists() and (CODE_DIR / "lib" / "PraNet_Res2Net.py").exists()


@pytest.mark.skipif(not HAS_PRANET, reason="needs `make pranet`")
def test_released_weights_load_strictly_and_predict():
    model = load_pranet(torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) == 32_547_319
    out = predict(model, torch.zeros(1, 3, 352, 352))
    assert out.shape == (1, 1, 352, 352) and torch.isfinite(out).all()


def test_missing_files_give_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="make pranet"):
        load_pranet(torch.device("cpu"), weights=tmp_path / "nope.pth")
