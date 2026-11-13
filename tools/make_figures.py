"""Draw the explanatory figures in docs/figures/ from the real pipeline and model.

    pip install matplotlib        (only this script needs it)
    python tools/make_figures.py  (results.png scores every trained model, so run it after training)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ocr import config
from ocr.dataset import normalize_line, to_tensor
from ocr.model import load_model
from ocr.predict import clean, find_lines, open_image

OUT = ROOT / "docs" / "figures"
INK, PAPER, ORANGE, BLUE, GREY = "#16213A", "#F7F5F0", "#D9682B", "#2F6DB5", "#8A93A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
                     "axes.edgecolor": GREY, "figure.facecolor": PAPER, "savefig.facecolor": PAPER})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / name)


def show(ax, img, title: str = "") -> None:
    ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, loc="left", fontsize=14, color=INK)


def fig_clean() -> None:
    raw = open_image(str(config.SAMPLES_DIR / "printed_dark.png"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.6))
    show(axes[0], raw, "1. Photo as uploaded (light text on a dark, uneven background)")
    show(axes[1], clean(raw), "2. After clean-up: dark text on even white paper")
    save(fig, "step1_clean.png")


def fig_lines() -> Image.Image:
    page = clean(open_image(str(config.SAMPLES_DIR / "handwritten_paragraph.png")))
    boxes = find_lines(page)
    a = np.asarray(page)
    ink = (a < 128).sum(1)
    fig, (ax, ap) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [6, 1]})
    show(ax, page, f"Found {len(boxes)} text lines")
    for i, (l, t, r, b) in enumerate(boxes, 1):
        ax.add_patch(plt.Rectangle((l, t), r - l, b - t, fill=False, ec=ORANGE, lw=2))
        ax.text(l - 6, (t + b) / 2, str(i), color=ORANGE, ha="right", va="center", fontsize=13, weight="bold")
    ap.barh(np.arange(len(ink)), ink, color=BLUE, height=1.0)
    ap.set_ylim(len(ink), 0), ap.set_yticks([]), ap.set_xticks([])
    ap.set_title("ink per row", fontsize=12, color=GREY)
    for s in ("top", "right", "bottom"):
        ap.spines[s].set_visible(False)
    save(fig, "step2_lines.png")
    return page.crop(boxes[0])


def fig_normalize() -> None:
    page = clean(open_image(str(config.SAMPLES_DIR / "printed_serif.png")))
    line = page.crop(find_lines(page)[0])
    norm = normalize_line(line)
    fig, axes = plt.subplots(2, 1, figsize=(9, 2.6), gridspec_kw={"height_ratios": [line.height, norm.height * 1.4]})
    show(axes[0], line, f"Cropped line: {line.width} x {line.height} px")
    show(axes[1], norm, f"Resized to {config.IMG_HEIGHT} px high ({norm.width} x {norm.height}), grey 0-1")
    save(fig, "step3_normalize.png")


@torch.no_grad()
def fig_cnn(model, norm: Image.Image) -> None:
    x = to_tensor(norm)
    x = torch.nn.functional.pad(x, (0, -x.shape[-1] % 4)).unsqueeze(0)
    mid = model.cnn[:4](x)[0]                          # 64 maps, 10 x W/4
    out = model.cnn(x)[0]                              # 192 maps, 2 x W/4
    top = mid.flatten(1).var(1).argsort(descending=True)[:4]
    fig, axes = plt.subplots(6, 1, figsize=(10, 7.2), gridspec_kw={"height_ratios": [4, 1.4, 1.4, 1.4, 1.4, 4]})
    fig.subplots_adjust(hspace=0.45)
    show(axes[0], np.asarray(norm), "Input line (1 x 40 x W)")
    for ax, c in zip(axes[1:5], top.tolist()):
        ax.imshow(mid[c], cmap="magma", aspect="auto", interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(f"map {c}", rotation=0, ha="right", va="center", fontsize=11, color=GREY)
    axes[1].set_title("4 of 64 feature maps after two conv blocks: each lights up on a different stroke pattern",
                      loc="left", fontsize=13)
    seq = out.reshape(-1, out.shape[-1])
    axes[5].imshow(seq, cmap="viridis", aspect="auto", interpolation="nearest")
    axes[5].set_xticks([]), axes[5].set_yticks([])
    axes[5].set_title(f"CNN output: {seq.shape[1]} columns x {seq.shape[0]} numbers - "
                      "one feature vector per 4-pixel slice", loc="left", fontsize=13)
    save(fig, "step4_cnn.png")


@torch.no_grad()
def fig_ctc(model) -> None:
    page = clean(open_image(str(config.SAMPLES_DIR / "printed_dark.png")))
    norm = normalize_line(page.crop(find_lines(page)[0]))
    ink = (np.asarray(norm) < 128).any(0)
    gaps = np.flatnonzero(np.convolve(~ink, np.ones(10), "valid") == 10)   # first wide gap = end of word
    end = int(gaps[gaps > 20][0]) + 6 if len(gaps[gaps > 20]) else norm.width
    word = norm.crop((0, 0, end + (-end % 4), norm.height))
    x = to_tensor(word).unsqueeze(0)
    probs = model(x)[0].softmax(-1)
    conf, ids = probs.max(-1)
    t = len(ids)
    labels = ["·" if i == 0 else config.CHARSET[i - 1] for i in ids.tolist()]
    collapsed, prev = [], 0
    for i in ids.tolist():
        if i != prev and i != 0:
            collapsed.append(config.CHARSET[i - 1])
        prev = i
    fig, axes = plt.subplots(2, 1, figsize=(11, 3.4), gridspec_kw={"height_ratios": [3, 1.3]})
    axes[0].imshow(np.asarray(word), cmap="gray", vmin=0, vmax=255, extent=(0, t, 1, 0), aspect="auto")
    axes[0].set_xticks([]), axes[0].set_yticks([])
    axes[0].set_title(f"The model gives one guess per 4-pixel column ({t} columns)", loc="left", fontsize=14)
    ax = axes[1]
    for c, (lab, p) in enumerate(zip(labels, conf.tolist())):
        blank = lab == "·"
        ax.add_patch(plt.Rectangle((c, 0), 1, 1, color=GREY if blank else ORANGE, alpha=0.15 if blank else 0.25 + 0.75 * p))
        ax.text(c + 0.5, 0.5, lab, ha="center", va="center", fontsize=12,
                color=GREY if blank else INK, weight="normal" if blank else "bold")
    ax.set_xlim(0, t), ax.set_ylim(0, 1), ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(f"merge repeats, drop blanks (·)  →  “{''.join(collapsed)}”", fontsize=15, color=INK, labelpad=10)
    save(fig, "step6_ctc.png")


def fig_results() -> None:
    from ocr.train import compare
    rows = compare("cuda" if torch.cuda.is_available() else "cpu")
    names = [r["name"].replace("CNN + ", "") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in zip(axes, ("hand_cer", "print_cer"), ("Handwriting", "Printed text")):
        vals = [100 * r[key] for r in rows]
        best = min(vals)
        bars = ax.bar(names, vals, color=[ORANGE if v == best else BLUE for v in vals], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%" + ("  best" if v == best else ""),
                    ha="center", va="bottom", fontsize=13, color=INK)
        ax.set_title(f"{title}: wrong characters (lower is better)", loc="left", fontsize=14)
        ax.set_facecolor(PAPER)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    save(fig, "results.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(config.MODEL_PATHS["lstm"])
    fig_clean()
    fig_normalize()
    fig_cnn(model, normalize_line(fig_lines()))
    fig_ctc(model)
    fig_results()
