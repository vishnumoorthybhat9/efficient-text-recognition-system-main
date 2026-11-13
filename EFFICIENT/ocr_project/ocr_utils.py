"""
Torch-free helpers for the handwritten-name OCR project.

Contains: data-CSV loading, image preprocessing, light augmentation,
CTC greedy decoding, and the character error rate (CER) metric.
Only numpy, pandas and OpenCV are needed, so this file can be tested
without a deep-learning framework.
"""
import csv
import os
import random

import cv2
import numpy as np

IMG_H = 64      # every image is resized to this height
IMG_W = 384     # every image is padded/squeezed to this width (T = IMG_W // 4 = 96 CTC steps)
BLANK = 0       # CTC blank class; real characters use indices 1..K


# --------------------------------------------------------------------------- data
def read_split(csv_path, images_dir):
    """Return a list of (image_path, label) pairs, dropping rows with empty labels."""
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("IDENTITY")
            if label is None or not str(label).strip() or str(label).lower() == "nan":
                continue
            items.append((os.path.join(images_dir, row["FILENAME"]), str(label)))
    return items


def build_charset(labels):
    """Sorted list of every character seen in the labels. Class id of chars[i] is i + 1."""
    return sorted({c for s in labels for c in s})


# -------------------------------------------------------------- preprocessing
def load_gray(path_or_array):
    if isinstance(path_or_array, (str, os.PathLike)):
        img = cv2.imread(os.fspath(path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
        return img
    img = np.asarray(path_or_array)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def preprocess(path_or_array, augment=False):
    """
    Grayscale -> (optional augmentation) -> resize to height IMG_H keeping aspect
    ratio -> pad with white to IMG_W (or squeeze if wider) -> float32 in [-1, 1],
    shape (IMG_H, IMG_W).
    """
    img = load_gray(path_or_array)
    if augment:
        img = augment_gray(img)

    h, w = img.shape
    new_w = max(1, int(round(w * IMG_H / h)))
    if new_w > IMG_W:
        # too wide: squeeze horizontally so no text is cut off
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        new_w = IMG_W
    else:
        img = cv2.resize(img, (new_w, IMG_H), interpolation=cv2.INTER_AREA)

    canvas = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)   # white background
    canvas[:, :new_w] = img
    x = canvas.astype(np.float32) / 255.0
    return (x - 0.5) / 0.5


def augment_gray(img, rng=random):
    """Small random rotation, shear, scale and brightness/contrast jitter (training only)."""
    h, w = img.shape
    angle = rng.uniform(-2.0, 2.0)
    scale = rng.uniform(0.9, 1.1)
    shear = rng.uniform(-0.15, 0.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 1] += shear  # horizontal shear, the usual slant of handwriting
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    alpha = rng.uniform(0.8, 1.2)   # contrast
    beta = rng.uniform(-20, 20)     # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img


# ------------------------------------------------------------------ decoding
def decode_greedy(class_ids, idx2char):
    """
    CTC greedy decoding for one sequence of per-step argmax class ids:
    collapse repeated ids, then drop blanks.
    """
    out, prev = [], None
    for k in class_ids:
        k = int(k)
        if k != prev and k != BLANK:
            out.append(idx2char[k - 1])
        prev = k
    return "".join(out)


# ------------------------------------------------------------------- metrics
def edit_distance(a, b):
    """Levenshtein distance between two strings (iterative, O(len(a)*len(b)))."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds, refs):
    """Character error rate = total edit distance / total reference characters."""
    errors = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    total = sum(len(r) for r in refs)
    return errors / max(1, total)


def exact_match(preds, refs):
    return sum(p == r for p, r in zip(preds, refs)) / max(1, len(refs))
