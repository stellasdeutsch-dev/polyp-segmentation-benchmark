"""Load the released PraNet model (reference only; decision D12/D13).

The model code is fetched by `make pranet` into third_party/PraNet/ at commit 95f12bd and is not
part of this repository. Two adaptations, neither touching its code:
  * PraNet's constructor loads its ImageNet backbone from a hard-coded path on the authors' machine
    (Res2Net_v1b.py:194). We build the backbone with pretrained=False instead; PraNet-19.pth
    contains every weight, backbone included.
  * It is run on whatever device we have (the original assumes CUDA).
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CODE_DIR = REPO / "third_party" / "PraNet"
WEIGHTS = REPO / "checkpoints" / "pranet" / "PraNet-19.pth"
TEST_SIZE = 352  # MyTest.py:10


def load_pranet(device: torch.device, weights: Path = WEIGHTS) -> torch.nn.Module:
    if not (CODE_DIR / "lib" / "PraNet_Res2Net.py").exists() or not weights.exists():
        raise FileNotFoundError("PraNet code or weights missing; run `make pranet` first")
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    import lib.PraNet_Res2Net as pranet_module  # noqa: PLC0415 (third-party code fetched at run time)

    original = pranet_module.res2net50_v1b_26w_4s

    @functools.wraps(original)
    def without_pretrained(*args, **kwargs):
        # PraNet calls this with pretrained=True explicitly, so a functools.partial default would be
        # overridden; force it off here (found in 3.1).
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    pranet_module.res2net50_v1b_26w_4s = without_pretrained
    try:
        model = pranet_module.PraNet()
    finally:
        pranet_module.res2net50_v1b_26w_4s = original
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)  # every key must match, in both directions
    return model.to(device).eval()


@torch.inference_mode()
def predict(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Logits of PraNet's final output (lateral_map_2, as in MyTest.py:33-34) for a (B, 3, 352, 352) batch."""
    return model(image)[3]
