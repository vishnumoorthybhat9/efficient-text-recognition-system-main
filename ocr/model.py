"""Line recognizer: a small CNN reads the line image, a sequence model reads the
column features left-to-right, and CTC aligns the output with the text.

The CNN, CTC loss, data and decoding are the same for every variant; only the
sequence model changes, so the variants can be compared fairly:

    lstm         2-layer bidirectional LSTM (the original CRNN)
    transformer  4-layer Transformer encoder (self-attention over all columns)
    mamba        6-layer Mamba-2 (selective state-space model), alternating direction
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import CHARSET, IMG_HEIGHT


def conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def sine_positions(t: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(1e4) / d))
    return torch.cat([torch.sin(pos * div), torch.cos(pos * div)], 1)


class TransformerSeq(nn.Module):
    def __init__(self, din: int, d: int = 192, layers: int = 4, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(din, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True,
                                           norm_first=True)      # pre-norm: stable from scratch
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + sine_positions(x.size(1), x.size(2), x.device).to(x.dtype)
        return self.norm(self.enc(x))


class MambaBlock(nn.Module):
    """One Mamba-2 block (selective state-space model, Dao & Gu 2024) in plain
    PyTorch, so it runs on CPU and Windows without the mamba-ssm CUDA kernels.

    The state update h[t] = exp(a[t]) * h[t-1] + dt[t] * B[t] * x[t],  y[t] = C[t] . h[t]
    is computed in its exact matrix form, y = (C B^T * decay) x, which is one
    batched matrix product instead of a Python loop over the columns."""

    def __init__(self, d: int, d_state: int = 16, expand: int = 2, head_dim: int = 48, d_conv: int = 4):
        super().__init__()
        self.di, self.n, self.p = expand * d, d_state, head_dim
        self.h = self.di // head_dim
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * self.di + 2 * d_state + self.h, bias=False)   # z, x, B, C, dt
        width = self.di + 2 * d_state
        self.conv = nn.Conv1d(width, width, d_conv, padding=d_conv - 1, groups=width)
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # start with steps in [0.001, 0.1]
        self.A_log = nn.Parameter(torch.log(torch.empty(self.h).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.h))
        self.out_proj = nn.Linear(self.di, d, bias=False)

    def scan(self, x, dt, b, c):
        """x: B T h p, dt: B T h, b/c: B T n -> y: B T h p (all float32)."""
        t = x.size(1)
        cs = (dt * -self.A_log.exp()).cumsum(1).transpose(1, 2)       # B h T: summed log-decay
        seg = cs.unsqueeze(-1) - cs.unsqueeze(-2)                     # decay from column s to l
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        # shortcut: whole-line T x T matrix (fine for text lines); use chunked SSD / mamba-ssm for very long inputs
        w = torch.einsum("bln,bsn->bls", c, b).unsqueeze(1) * seg.masked_fill(~causal, -math.inf).exp()
        return torch.einsum("bhls,bshp->blhp", w, x * dt.unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # B x T x d
        bsz, t, _ = x.shape
        z, xbc, dt = self.in_proj(self.norm(x)).split([self.di, self.di + 2 * self.n, self.h], -1)
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[..., :t].transpose(1, 2))
        u, b, c = xbc.float().split([self.di, self.n, self.n], -1)
        u = u.view(bsz, t, self.h, self.p)
        dt = F.softplus(dt.float() + self.dt_bias)
        y = self.scan(u, dt, b, c) + u * self.D[:, None]
        return x + self.out_proj((y.reshape(bsz, t, self.di) * F.silu(z.float())).to(x.dtype))


class MambaSeq(nn.Module):
    def __init__(self, din: int, d: int = 192, layers: int = 6):
        super().__init__()
        self.proj = nn.Linear(din, d)
        self.blocks = nn.ModuleList(MambaBlock(d) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for i, block in enumerate(self.blocks):
            # Mamba reads one way; flipping every other layer lets each column see both sides
            x = block(x) if i % 2 == 0 else block(x.flip(1)).flip(1)
        return self.norm(x)


class CRNN(nn.Module):
    def __init__(self, arch: str = "lstm", num_classes: int = len(CHARSET) + 1, hidden: int = 192):
        super().__init__()
        self.arch = arch
        self.cnn = nn.Sequential(                      # input 1 x 40 x W
            conv(1, 32), nn.MaxPool2d(2),              # 20 x W/2
            conv(32, 64), nn.MaxPool2d(2),             # 10 x W/4
            conv(64, 96), nn.MaxPool2d((2, 1)),        # 5 x W/4
            conv(96, 128), conv(128, 192), nn.MaxPool2d((2, 1)),   # 2 x W/4
            nn.Dropout2d(0.1),
        )
        feat = 192 * (IMG_HEIGHT // 16)
        if arch == "lstm":
            self.rnn = nn.LSTM(feat, hidden, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=0.3)
            out = hidden * 2
        elif arch == "transformer":
            self.rnn, out = TransformerSeq(feat, hidden), hidden
        elif arch == "mamba":
            self.rnn, out = MambaSeq(feat, hidden), hidden
        else:
            raise ValueError(f"unknown architecture {arch!r}")
        self.head = nn.Linear(out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                                # B x C x H' x T
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)
        out = self.rnn(f)
        if self.arch == "lstm":
            out = out[0]
        return self.head(out)                          # B x T x classes (logits)


def greedy_decode(logits: torch.Tensor) -> list[tuple[str, float]]:
    """CTC best-path decoding -> (text, mean character confidence) per item."""
    probs = logits.softmax(-1)
    conf, ids = probs.max(-1)
    results = []
    for row_ids, row_conf in zip(ids.tolist(), conf.tolist()):
        chars, confs, prev = [], [], 0
        for i, p in zip(row_ids, row_conf):
            if i != prev and i != 0:
                chars.append(CHARSET[i - 1])
                confs.append(p)
            prev = i
        results.append(("".join(chars), sum(confs) / len(confs) if confs else 0.0))
    return results


def save_model(model: CRNN, path, **info) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"state": state, "charset": CHARSET, "height": IMG_HEIGHT, "arch": model.arch, **info}, path)


def load_model(path, device: str = "cpu") -> tuple[CRNN, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("charset") != CHARSET or ckpt.get("height") != IMG_HEIGHT:
        raise RuntimeError("Saved model does not match ocr/config.py - retrain it.")
    ckpt.setdefault("arch", "lstm")                    # models saved before the arch option
    model = CRNN(ckpt["arch"])
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                           for k, v in ckpt.pop("state").items()})
    return model.to(device).eval(), ckpt
