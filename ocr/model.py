"""CRNN: a small CNN reads the line image, a BiLSTM models the character
sequence, and CTC aligns the two.  ~2.3 M parameters, trained from scratch."""
from __future__ import annotations

import torch
from torch import nn

from .config import CHARSET, IMG_HEIGHT


def conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, num_classes: int = len(CHARSET) + 1, hidden: int = 192):
        super().__init__()
        self.cnn = nn.Sequential(                      # input 1 x 40 x W
            conv(1, 32), nn.MaxPool2d(2),              # 20 x W/2
            conv(32, 64), nn.MaxPool2d(2),             # 10 x W/4
            conv(64, 96), nn.MaxPool2d((2, 1)),        # 5 x W/4
            conv(96, 128), conv(128, 192), nn.MaxPool2d((2, 1)),   # 2 x W/4
            nn.Dropout2d(0.1),
        )
        self.rnn = nn.LSTM(192 * (IMG_HEIGHT // 16), hidden, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                                # B x C x H' x T
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)
        out, _ = self.rnn(f)
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


def save_model(model: nn.Module, path, **info) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"state": state, "charset": CHARSET, "height": IMG_HEIGHT, **info}, path)


def load_model(path, device: str = "cpu") -> tuple[CRNN, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("charset") != CHARSET or ckpt.get("height") != IMG_HEIGHT:
        raise RuntimeError("Saved model does not match ocr/config.py - retrain it.")
    model = CRNN()
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                           for k, v in ckpt.pop("state").items()})
    return model.to(device).eval(), ckpt
