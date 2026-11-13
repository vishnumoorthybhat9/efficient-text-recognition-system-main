"""Dataset: unzip once, then serve augmented text-line batches for training."""
from __future__ import annotations

import csv
import random
import zipfile
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageFilter

from .config import BATCH_SIZE, CHARSET, DATASET_DIR, DATASET_ZIP, IMG_HEIGHT, MAX_WIDTH

CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARSET)}


def ensure_extracted() -> None:
    """Extract data/dataset.zip into dataset/ the first time it is needed."""
    marker = DATASET_DIR / "labels.csv"
    if marker.exists():
        return
    if not DATASET_ZIP.exists():
        raise SystemExit(f"Dataset archive not found: {DATASET_ZIP}")
    print(f"Extracting {DATASET_ZIP.name} -> {DATASET_DIR.name}/ ...", flush=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP) as z:
        names = [n for n in z.namelist() if n != "labels.csv"]
        z.extractall(DATASET_DIR, names)
        z.extract("labels.csv", DATASET_DIR)      # last, so it marks completion
    print(f"  {len(names)} files extracted.", flush=True)


def load_labels() -> list[dict]:
    with open(DATASET_DIR / "labels.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def encode(text: str) -> list[int]:
    return [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]


def normalize_line(img: Image.Image, pad: int = 2) -> Image.Image:
    """Crop a dark-on-light line image to its ink and resize it to IMG_HEIGHT."""
    a = np.asarray(img)
    ink = a < 140
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    if len(rows) and len(cols):
        img = img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    inner = IMG_HEIGHT - 2 * pad
    w = max(4, round(img.width * inner / img.height))
    img = img.resize((w, inner), Image.BILINEAR)
    out = Image.new("L", (w + 2 * pad, IMG_HEIGHT), 255)
    out.paste(img, (pad, pad))
    return out


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Greyscale PIL image -> 1 x H x W float tensor, ink = 1, paper = 0."""
    return torch.from_numpy(1.0 - np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Cheap distortions applied to a white-background line before normalising."""
    w, h = img.size
    if rng.random() < 0.6:                                   # slant + small rotation
        shear, rot = rng.uniform(-0.3, 0.3), rng.uniform(-0.02, 0.02)
        img = img.transform((w, h), Image.AFFINE,
                            (1, shear, -shear * h / 2, rot, 1, -rot * w / 2),
                            Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.5:                                   # stretch / squeeze
        img = img.resize((max(8, int(w * rng.uniform(0.8, 1.25))), h), Image.BILINEAR)
    r = rng.random()
    if r < 0.15:
        img = img.filter(ImageFilter.MinFilter(3))           # thicker strokes
    elif r < 0.25:
        img = img.filter(ImageFilter.MaxFilter(3))           # thinner strokes
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return img


def photometric(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Fade ink, grey the paper and add sensor noise (x: ink = 1)."""
    if rng.random() < 0.5:
        x = x * rng.uniform(0.55, 1.0) + rng.uniform(0.0, 0.15)
    if rng.random() < 0.5:
        x = x + torch.randn_like(x) * rng.uniform(0.02, 0.10)
    return x.clamp_(0, 1)


class LineBatches(torch.utils.data.Dataset):
    """Each item is one ready batch (images, targets, target_lengths, texts).

    Handwritten samples are whole IAM lines.  Printed samples are built on the
    fly by joining several word images of the same font, so 17k words turn
    into a practically unlimited number of different text lines.
    """

    def __init__(self, split: str, batches: int, batch_size: int = BATCH_SIZE, seed: int = 0):
        self.train = split == "train"
        self.batches, self.batch_size, self.seed = batches, batch_size, seed
        self.hand: list[tuple[str, str]] = []
        self.fonts: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in load_labels():
            if row["split"] != split:
                continue
            if row["kind"] == "printed":
                self.fonts[row["font"]].append((row["file"], row["text"]))
            else:
                self.hand.append((row["file"], row["text"]))
        self.font_ids = sorted(self.fonts)
        self.cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return self.batches

    def _open(self, name: str) -> Image.Image:
        img = self.cache.get(name)
        if img is None:
            img = self.cache[name] = Image.open(DATASET_DIR / name).convert("L")
        return img

    def _printed(self, rng: random.Random, n_words: int) -> tuple[Image.Image, str]:
        words = self.fonts[rng.choice(self.font_ids)]
        picks = [rng.choice(words) for _ in range(n_words)]
        imgs = [self._open(f) for f, _ in picks]
        gap = rng.randint(9, 24)
        h = imgs[0].height
        line = Image.new("L", (sum(i.width for i in imgs) + gap * (n_words - 1), h), 255)
        x = 0
        for im in imgs:
            line.paste(im, (x, 0))
            x += im.width + gap
        return line, " ".join(t for _, t in picks)

    def __getitem__(self, index: int):
        rng = random.Random(self.seed * 1_000_003 + index if not self.train else None)
        kind = index % 3 if not self.train else rng.choice((0, 0, 1, 2))
        items: list[tuple[Image.Image, str]] = []
        for k in range(self.batch_size):
            if kind == 0 and self.hand:                      # real handwriting
                i = (index * self.batch_size + k) % len(self.hand) if not self.train \
                    else rng.randrange(len(self.hand))
                name, text = self.hand[i]
                items.append((self._open(name), text))
            elif kind == 1:                                  # long printed lines
                items.append(self._printed(rng, rng.randint(4, 8)))
            else:                                            # short printed lines
                items.append(self._printed(rng, rng.randint(1, 3)))

        tensors, targets, texts = [], [], []
        for img, text in items:
            if self.train:
                img = augment(img, rng)
            img = normalize_line(img, pad=rng.randint(1, 5) if self.train else 2)
            # CTC needs at least one time step (4 px) per character, two for doubles
            need = 4 * (2 * len(text) + 2)
            if img.width < need or img.width > MAX_WIDTH:
                img = img.resize((min(max(img.width, need), MAX_WIDTH), IMG_HEIGHT), Image.BILINEAR)
            x = to_tensor(img)
            tensors.append(photometric(x, rng) if self.train else x)
            targets.append(torch.tensor(encode(text), dtype=torch.long))
            texts.append(text)

        width = max(t.shape[-1] for t in tensors)
        width += -width % 4
        batch = torch.zeros(len(tensors), 1, IMG_HEIGHT, width)
        for i, t in enumerate(tensors):
            batch[i, :, :, : t.shape[-1]] = t
        lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        return batch, torch.cat(targets), lengths, texts
