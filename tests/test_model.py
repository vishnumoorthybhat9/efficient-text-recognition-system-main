"""Quick checks: python tests/test_model.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr import config
from ocr.model import CRNN, MambaBlock, load_model


def test_mamba_matrix_form_equals_step_by_step():
    torch.manual_seed(0)
    blk = MambaBlock(32, d_state=4, head_dim=16)
    bsz, t = 2, 11
    x, dt = torch.randn(bsz, t, blk.h, blk.p), torch.rand(bsz, t, blk.h)
    b, c = torch.randn(bsz, t, blk.n), torch.randn(bsz, t, blk.n)
    a = torch.exp(dt * -blk.A_log.exp())                         # B T h
    h, ys = torch.zeros(bsz, blk.h, blk.p, blk.n), []
    for i in range(t):                                           # the recurrence, one column at a time
        h = a[:, i, :, None, None] * h + (dt[:, i, :, None] * x[:, i])[..., None] * b[:, i, None, None]
        ys.append((h * c[:, i, None, None]).sum(-1))
    assert torch.allclose(blk.scan(x, dt, b, c), torch.stack(ys, 1), atol=1e-5)


def test_every_arch_runs_and_mamba_sees_both_sides():
    x = torch.zeros(2, 1, config.IMG_HEIGHT, 64)
    for arch in config.ARCHS:
        net = CRNN(arch).eval()
        y0 = net(x)
        assert y0.shape == (2, 16, len(config.CHARSET) + 1), (arch, y0.shape)
    x2 = x.clone()
    x2[..., :8] = 1                                             # change only the left edge
    assert (net(x2)[0, -1] - y0[0, -1]).abs().max() > 0         # ...and the right end notices


def test_shipped_model_loads_as_lstm():
    model, info = load_model(config.MODEL_PATHS["lstm"])
    assert info["arch"] == "lstm" and model.arch == "lstm"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
