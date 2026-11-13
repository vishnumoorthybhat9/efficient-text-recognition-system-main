"""Read text from an image: clean it, split it into lines, run the CRNN on each."""
from __future__ import annotations

import io
import time

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from .dataset import normalize_line, to_tensor
from .model import greedy_decode

MAX_SIDE = 2200


def open_image(data: bytes | str) -> Image.Image:
    img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):                    # flatten transparency on white
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(white, img)
    img = img.convert("L")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def clean(img: Image.Image) -> Image.Image:
    """Return dark text on an even white background (handles shadows, dark themes)."""
    a = np.asarray(img, dtype=np.float32)
    if np.median(a) < 110:                                 # light text on dark paper
        img, a = ImageOps.invert(img), 255.0 - a
    # estimate the paper brightness around every pixel and divide it out
    small = img.resize((max(8, img.width // 8), max(8, img.height // 8)), Image.BILINEAR)
    paper = small.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(4))
    paper = np.asarray(paper.resize(img.size, Image.BILINEAR), dtype=np.float32)
    a = np.clip(a / np.maximum(paper, 1.0), 0, 1)
    lo = np.percentile(a, 1)
    a = np.clip((a - lo) / max(1e-3, 1.0 - lo), 0, 1)       # stretch ink to black
    return Image.fromarray((a * 255).astype(np.uint8))


def otsu(a: np.ndarray) -> int:
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total, levels = hist.sum(), np.arange(256)
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * levels)
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m0[-1] * w0 - m0 * total) ** 2 / (w0 * w1)
    return int(np.nanargmax(between))


def find_lines(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Split a cleaned page into text-line boxes (left, top, right, bottom)."""
    a = np.asarray(img)
    ink = a < min(otsu(a), 200)
    profile = ink.sum(1).astype(np.float32)
    if profile.max() == 0:
        return []
    profile = np.convolve(profile, np.ones(3) / 3, mode="same")
    busy = profile > 0.08 * np.percentile(profile[profile > 0], 90)

    runs, start = [], None                                  # consecutive busy rows
    for y, on in enumerate(np.append(busy, False)):
        if on and start is None:
            start = y
        elif not on and start is not None:
            runs.append([start, y])
            start = None
    if not runs:
        return []
    typical = float(np.median([b - t for t, b in runs]))
    merged: list[list[int]] = []
    for t, b in runs:                                       # dots, accents, specks
        if merged and (b - t < 0.4 * typical) and t - merged[-1][1] < 0.4 * typical:
            merged[-1][1] = b
        elif b - t >= 0.4 * typical or len(runs) == 1:
            merged.append([t, b])

    boxes = []
    for i, (t, b) in enumerate(merged):                     # give tails some room
        grow = int(0.3 * (b - t)) + 2
        top = max(t - grow, (merged[i - 1][1] + t) // 2 if i else 0)
        bottom = min(b + grow, (merged[i + 1][0] + b + 1) // 2 if i + 1 < len(merged) else a.shape[0])
        cols = np.flatnonzero(ink[top:bottom].any(0))
        if len(cols) and bottom - top >= 6:
            boxes.append((int(cols[0]), int(top), int(cols[-1]) + 1, int(bottom)))
    return boxes


@torch.no_grad()
def recognize(model, data: bytes | str, device: str = "cpu") -> dict:
    started = time.time()
    page = clean(open_image(data))
    lines = []
    for box in find_lines(page):
        x = to_tensor(normalize_line(page.crop(box)))
        x = torch.nn.functional.pad(x, (0, -x.shape[-1] % 4)).unsqueeze(0).to(device)
        text, conf = greedy_decode(model(x))[0]
        if text.strip():
            lines.append({"text": text.strip(), "confidence": round(conf, 3), "box": box})
    return {
        "text": "\n".join(l["text"] for l in lines),
        "lines": lines,
        "width": page.width,
        "height": page.height,
        "seconds": round(time.time() - started, 3),
    }
